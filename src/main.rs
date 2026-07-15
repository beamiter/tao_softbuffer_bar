use anyhow::{Context as _, Result};
use cairo::{Context as CairoContext, Format, ImageSurface};
use log::warn;
use pango::FontDescription;
use std::env;
use std::num::NonZeroU32;
use std::os::fd::AsRawFd;
use std::process::Command;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tao::platform::run_return::EventLoopExtRunReturn;
use tao::{
    dpi::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy},
    monitor::MonitorHandle,
    window::{Window, WindowBuilder},
};

use xbar_core::{
    BarEffect, BarRuntime, ModelConfig, RuntimeUpdate, SharedEventNotifier, SharedTransport,
    logging::init as initialize_logging,
    presentation::{Point, PointerAction, PresentationConfig, Size},
    render::cairo::CairoBar,
};

const TRANSPORT_RETRY_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone)]
enum UserEvent {
    Tick,
    SharedUpdated(Arc<AtomicBool>),
}

struct EventForwarder {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for EventForwarder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take()
            && let Err(payload) = worker.join()
        {
            warn!("event forwarding thread panicked: {payload:?}");
        }
    }
}

fn spawn_tick_thread(proxy: EventLoopProxy<UserEvent>) -> EventForwarder {
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let worker = thread::spawn(move || {
        while !worker_stop.load(Ordering::Acquire) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0));
            let sleep_duration =
                Duration::from_nanos(1_000_000_000_u64.saturating_sub(now.subsec_nanos().into()));
            thread::sleep(sleep_duration);
            if worker_stop.load(Ordering::Acquire) || proxy.send_event(UserEvent::Tick).is_err() {
                break;
            }
        }
    });
    EventForwarder {
        stop,
        worker: Some(worker),
    }
}

fn spawn_shared_thread(
    proxy: EventLoopProxy<UserEvent>,
    notifier: Option<SharedEventNotifier>,
) -> Option<EventForwarder> {
    notifier.map(|notifier| {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        // The event-loop handler clears this only after it has drained the
        // transport, so at most one shared update can be queued at a time.
        let worker_pending = Arc::new(AtomicBool::new(false));
        let worker = thread::spawn(move || {
            let mut poll_fd = libc::pollfd {
                fd: notifier.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            while !worker_stop.load(Ordering::Acquire) {
                poll_fd.revents = 0;
                let poll_result = unsafe { libc::poll(&mut poll_fd, 1, 250) };
                if poll_result < 0 {
                    let error = std::io::Error::last_os_error();
                    if error.raw_os_error() == Some(libc::EINTR) {
                        continue;
                    }
                    warn!("shared notifier poll failed: {error}");
                    break;
                }
                if poll_result == 0 {
                    continue;
                }
                if poll_fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                    warn!(
                        "shared notifier became unusable: poll revents={}",
                        poll_fd.revents
                    );
                    break;
                }
                if poll_fd.revents & libc::POLLIN != 0 {
                    match notifier.drain() {
                        Ok(0) => {}
                        Ok(_) => {
                            if worker_pending
                                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                                .is_ok()
                            {
                                let event = UserEvent::SharedUpdated(Arc::clone(&worker_pending));
                                if proxy.send_event(event).is_err() {
                                    worker_pending.store(false, Ordering::Release);
                                    break;
                                }
                            }
                            while worker_pending.load(Ordering::Acquire)
                                && !worker_stop.load(Ordering::Acquire)
                            {
                                thread::sleep(Duration::from_millis(10));
                            }
                        }
                        Err(error) => {
                            warn!("failed to drain shared notifier: {error}");
                            break;
                        }
                    }
                }
            }
        });
        EventForwarder {
            stop,
            worker: Some(worker),
        }
    })
}

struct CairoBackBuffer {
    width: u32,
    height: u32,
    image: ImageSurface,
}

impl CairoBackBuffer {
    fn new(width: u32, height: u32) -> Result<Self> {
        let image = ImageSurface::create(
            Format::ARgb32,
            i32::try_from(width)?,
            i32::try_from(height)?,
        )?;
        Ok(Self {
            width,
            height,
            image,
        })
    }

    fn ensure_size(&mut self, width: u32, height: u32) -> Result<()> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.image = ImageSurface::create(
            Format::ARgb32,
            i32::try_from(width)?,
            i32::try_from(height)?,
        )?;
        self.width = width;
        self.height = height;
        Ok(())
    }
}

struct App {
    window: Rc<Window>,
    back: CairoBackBuffer,
    soft_surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    bar: CairoBar,
    scale_factor: f64,
    logical_size: LogicalSize<f64>,
    default_logical_size: LogicalSize<f64>,
    last_physical_size: PhysicalSize<u32>,
    last_cursor_pos: Option<Point>,
    shared_path: String,
    last_transport_attempt: Instant,
}

impl App {
    fn new(
        window: Rc<Window>,
        bar: CairoBar,
        logical_size: LogicalSize<f64>,
        scale_factor: f64,
        shared_path: String,
    ) -> Result<Self> {
        let physical_size = window.inner_size();
        let soft_context = softbuffer::Context::new(Rc::clone(&window))
            .map_err(|error| anyhow::anyhow!("failed to create softbuffer context: {error}"))?;
        let mut soft_surface = softbuffer::Surface::new(&soft_context, Rc::clone(&window))
            .map_err(|error| anyhow::anyhow!("failed to create softbuffer surface: {error}"))?;
        resize_soft_surface(&mut soft_surface, physical_size)?;

        Ok(Self {
            window,
            back: CairoBackBuffer::new(physical_size.width, physical_size.height)?,
            soft_surface,
            bar,
            scale_factor,
            logical_size,
            default_logical_size: logical_size,
            last_physical_size: physical_size,
            last_cursor_pos: None,
            shared_path,
            last_transport_attempt: Instant::now(),
        })
    }

    fn redraw(&mut self) -> Result<()> {
        let PhysicalSize { width, height } = self.last_physical_size;
        if width == 0 || height == 0 {
            return Ok(());
        }
        self.back.ensure_size(width, height)?;

        {
            let context = CairoContext::new(&self.back.image)?;
            context.scale(self.scale_factor, self.scale_factor);
            self.bar.render(
                &context,
                Size::new(
                    self.logical_size.width as f32,
                    self.logical_size.height as f32,
                ),
            )?;
        }

        self.back.image.flush();
        let stride = usize::try_from(self.back.image.stride())?;
        let data = self.back.image.data()?;
        let width = width as usize;
        let height = height as usize;
        let mut buffer = self
            .soft_surface
            .buffer_mut()
            .map_err(|error| anyhow::anyhow!("failed to acquire softbuffer frame: {error}"))?;

        if stride == width * 4 {
            let source: &[u32] = bytemuck::cast_slice(&data[..height * stride]);
            buffer[..width * height].copy_from_slice(source);
        } else {
            for y in 0..height {
                let row = &data[y * stride..y * stride + width * 4];
                let source: &[u32] = bytemuck::cast_slice(row);
                buffer[y * width..(y + 1) * width].copy_from_slice(source);
            }
        }
        buffer
            .present()
            .map_err(|error| anyhow::anyhow!("failed to present softbuffer frame: {error}"))?;
        Ok(())
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }

    fn resize(&mut self, physical_size: PhysicalSize<u32>) {
        self.last_physical_size = physical_size;
        self.logical_size = physical_size.to_logical(self.scale_factor);
        if let Err(error) = resize_soft_surface(&mut self.soft_surface, physical_size) {
            warn!("failed to resize softbuffer surface: {error:#}");
        }
        self.request_redraw();
    }

    fn update_hover(&mut self, point: Point) {
        if self.bar.pointer_motion(point) {
            self.request_redraw();
        }
    }

    fn handle_pointer_action(&mut self, point: Point, action: PointerAction) {
        let update = self.bar.pointer_action(point, action);
        self.handle_runtime_update(update);
    }

    fn handle_runtime_update(&mut self, update: RuntimeUpdate) {
        let RuntimeUpdate {
            changes,
            platform_effects,
            issues,
        } = update;

        for issue in issues {
            warn!("xbar runtime issue: {issue:?}");
        }
        for effect in platform_effects {
            self.handle_platform_effect(effect);
        }
        if !changes.is_empty() {
            self.request_redraw();
        }
    }

    fn tick_and_poll(&mut self) {
        if !self.shared_path.is_empty()
            && self.bar.runtime().transport().is_none()
            && self.last_transport_attempt.elapsed() >= TRANSPORT_RETRY_INTERVAL
        {
            self.last_transport_attempt = Instant::now();
            match SharedTransport::open(&self.shared_path) {
                Ok(transport) => {
                    self.bar.runtime_mut().set_transport(Some(transport));
                    log::debug!("reconnected WM transport at {}", self.shared_path);
                }
                Err(error) => log::debug!("WM transport is still unavailable: {error}"),
            }
        }

        let mut update = self.bar.tick();
        update.merge(self.bar.poll_transport());
        self.handle_runtime_update(update);
    }

    fn handle_platform_effect(&mut self, effect: BarEffect) {
        match effect {
            BarEffect::ApplyMonitorGeometry(geometry) => self.apply_monitor_geometry(geometry),
            BarEffect::ClearMonitorGeometry => {
                self.window
                    .set_outer_position(LogicalPosition::new(0.0, 0.0));
                self.window.set_inner_size(self.default_logical_size);
            }
            BarEffect::Screenshot => spawn_program("flameshot", &["gui"]),
            BarEffect::OpenAudioControl => spawn_program("pavucontrol", &[]),
            BarEffect::WindowManager(_)
            | BarEffect::ToggleMute
            | BarEffect::AdjustVolume(_)
            | BarEffect::AdjustBrightness(_)
            | BarEffect::RefreshBattery => {
                warn!("no frontend adapter handled platform effect: {effect:?}");
            }
        }
    }

    fn apply_monitor_geometry(&self, geometry: xbar_core::MonitorGeometry) {
        let height = (f64::from(self.bar.config().bar_height) * self.scale_factor)
            .round()
            .clamp(1.0, f64::from(u32::MAX)) as u32;
        self.window
            .set_outer_position(PhysicalPosition::new(geometry.x, geometry.y));
        self.window
            .set_inner_size(PhysicalSize::new(geometry.width, height));
    }
}

fn resize_soft_surface(
    surface: &mut softbuffer::Surface<Rc<Window>, Rc<Window>>,
    size: PhysicalSize<u32>,
) -> Result<()> {
    let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
    else {
        return Ok(());
    };
    surface
        .resize(width, height)
        .map_err(|error| anyhow::anyhow!("softbuffer resize failed: {error}"))
}

fn spawn_program(program: &str, args: &[&str]) {
    let program = program.to_owned();
    let args = args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>();
    thread::spawn(move || {
        if let Err(error) = Command::new(&program).args(&args).status() {
            warn!("failed to run {program}: {error}");
        }
    });
}

fn main() -> Result<()> {
    let shared_path = env::args().skip(1).last().unwrap_or_default();
    initialize_logging("tao_softbuffer_bar", &shared_path)?;

    let transport = if shared_path.is_empty() {
        None
    } else {
        Some(
            SharedTransport::open(&shared_path)
                .with_context(|| format!("failed to open shared transport {shared_path}"))?,
        )
    };
    let notifier = transport
        .as_ref()
        .map(|transport| transport.notifier(true))
        .transpose()
        .context("failed to start shared transport notifier")?;

    let runtime = BarRuntime::with_transport(ModelConfig::default(), transport)?;
    let presentation = PresentationConfig {
        bar_height: 38.0,
        ..PresentationConfig::default()
    };
    let font = FontDescription::from_string(
        &env::var("XBAR_FONT").unwrap_or_else(|_| "monospace 11".to_owned()),
    );
    let bar = CairoBar::new(runtime, presentation, font);

    let mut event_loop: EventLoop<UserEvent> = EventLoopBuilder::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let _tick_forwarder = spawn_tick_thread(proxy.clone());
    let _shared_forwarder = spawn_shared_thread(proxy, notifier);

    let primary_monitor: Option<MonitorHandle> = event_loop
        .primary_monitor()
        .or_else(|| event_loop.available_monitors().next());
    let scale_factor = primary_monitor
        .as_ref()
        .map(MonitorHandle::scale_factor)
        .unwrap_or(1.0);
    let screen_size = primary_monitor
        .as_ref()
        .map(MonitorHandle::size)
        .unwrap_or(PhysicalSize::new(1920, 1080));
    let logical_size = LogicalSize::new(screen_size.width as f64 / scale_factor, 38.0);

    let window = Rc::new(
        WindowBuilder::new()
            .with_title("tao_softbuffer_bar")
            .with_inner_size(logical_size)
            .with_decorations(false)
            .with_resizable(true)
            .with_visible(true)
            .with_transparent(false)
            .build(&event_loop)
            .context("failed to build tao window")?,
    );
    let mut app = App::new(window, bar, logical_size, scale_factor, shared_path)?;

    let update = app.bar.tick();
    app.handle_runtime_update(update);
    let update = app.bar.poll_transport();
    app.handle_runtime_update(update);
    app.request_redraw();

    let exit_code = event_loop.run_return(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::UserEvent(UserEvent::Tick) => {
                app.tick_and_poll();
            }
            Event::UserEvent(UserEvent::SharedUpdated(pending)) => {
                let update = app.bar.poll_transport();
                app.handle_runtime_update(update);
                pending.store(false, Ordering::Release);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(size) => app.resize(size),
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                } => {
                    app.scale_factor = scale_factor;
                    app.resize(*new_inner_size);
                    if let Some(geometry) = app.bar.runtime().view().geometry {
                        app.apply_monitor_geometry(geometry);
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let logical = position.to_logical::<f64>(app.scale_factor);
                    let point = Point::new(logical.x as f32, logical.y as f32);
                    app.last_cursor_pos = Some(point);
                    app.update_hover(point);
                }
                WindowEvent::CursorLeft { .. } => {
                    app.last_cursor_pos = None;
                    if app.bar.pointer_leave() {
                        app.request_redraw();
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    use tao::event::{ElementState, MouseButton};
                    if state == ElementState::Pressed
                        && let Some(point) = app.last_cursor_pos
                    {
                        let action = match button {
                            MouseButton::Left => Some(PointerAction::Primary),
                            MouseButton::Right => Some(PointerAction::Secondary),
                            MouseButton::Middle | MouseButton::Other(_) => None,
                            _ => None,
                        };
                        if let Some(action) = action {
                            app.handle_pointer_action(point, action);
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    use tao::event::MouseScrollDelta;
                    if let Some(point) = app.last_cursor_pos {
                        let y = match delta {
                            MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                            MouseScrollDelta::PixelDelta(position) => position.y,
                            _ => 0.0,
                        };
                        let action = if y > 0.0 {
                            Some(PointerAction::ScrollUp)
                        } else if y < 0.0 {
                            Some(PointerAction::ScrollDown)
                        } else {
                            None
                        };
                        if let Some(action) = action {
                            app.handle_pointer_action(point, action);
                        }
                    }
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                if let Err(error) = app.redraw() {
                    warn!("redraw failed: {error:#}");
                }
            }
            _ => {}
        }
    });

    if exit_code == 0 {
        Ok(())
    } else {
        anyhow::bail!("tao event loop exited with status {exit_code}")
    }
}
