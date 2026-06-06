use anyhow::Result;
use log::warn;
use shared_structures::SharedRingBuffer;
use std::env;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tao::{
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy},
    monitor::MonitorHandle,
    window::{Window, WindowBuilder},
};

use xbar_core::{
    AppState, BarConfig, Color, Rect, ShapeStyle, ThemeMode,
    cairo::{Context, Format, ImageSurface},
    colors_for_theme, draw_bar, initialize_logging,
    pango::FontDescription,
    spawn_shared_eventfd_notifier,
};

fn tuned_colors_for_theme(mode: ThemeMode) -> xbar_core::Colors {
    let mut c = colors_for_theme(mode);
    match mode {
        ThemeMode::Dark => {
            c.bg = Color::rgb(13, 16, 23);
            c.text = Color::rgb(235, 238, 245);
            c.gray = Color::rgb(45, 55, 72);
            c.time = Color::rgb(9, 41, 64);
            c.accent = Color::rgb(8, 145, 178);
            c.accent_light = Color::rgb(34, 211, 238);
            c.dim = Color::rgb(81, 90, 104);
        }
        ThemeMode::Light => {
            c.bg = Color::rgb(246, 247, 250);
            c.text = Color::rgb(22, 24, 28);
            c.gray = Color::rgb(203, 213, 225);
            c.time = Color::rgb(224, 242, 254);
            c.accent = Color::rgb(59, 130, 246);
            c.accent_light = Color::rgb(96, 165, 250);
            c.dim = Color::rgb(100, 116, 139);
        }
    }
    c
}

fn background_opacity_from_env() -> f64 {
    env::var("XBAR_BG_OPACITY")
        .or_else(|_| env::var("XBAR_OPACITY"))
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

fn set_argb32_pixel_opacity(pixel: &mut [u8], opacity: f64) {
    let alpha = (opacity.clamp(0.0, 1.0) * 255.0).round() as u16;

    #[cfg(target_endian = "little")]
    let channels = [0usize, 1, 2];
    #[cfg(target_endian = "big")]
    let channels = [1usize, 2, 3];

    for channel in channels {
        pixel[channel] = ((pixel[channel] as u16 * alpha + 127) / 255) as u8;
    }

    #[cfg(target_endian = "little")]
    {
        pixel[3] = alpha as u8;
    }
    #[cfg(target_endian = "big")]
    {
        pixel[0] = alpha as u8;
    }
}

fn is_inside_round_rect(px: usize, py: usize, rect: &Rect, radius: f64) -> bool {
    if rect.w == 0 || rect.h == 0 {
        return false;
    }

    let x0 = rect.x as f64;
    let y0 = rect.y as f64;
    let x1 = x0 + rect.w as f64;
    let y1 = y0 + rect.h as f64;
    let x = px as f64 + 0.5;
    let y = py as f64 + 0.5;

    if x < x0 || x >= x1 || y < y0 || y >= y1 {
        return false;
    }

    let radius = radius.min(rect.w as f64 / 2.0).min(rect.h as f64 / 2.0) + 1.0;
    if radius <= 1.0 {
        return true;
    }

    let inner_x0 = x0 + radius;
    let inner_y0 = y0 + radius;
    let inner_x1 = x1 - radius;
    let inner_y1 = y1 - radius;

    if (x >= inner_x0 && x < inner_x1) || (y >= inner_y0 && y < inner_y1) {
        return true;
    }

    let nearest_x = x.clamp(inner_x0, inner_x1);
    let nearest_y = y.clamp(inner_y0, inner_y1);
    let dx = x - nearest_x;
    let dy = y - nearest_y;
    dx * dx + dy * dy <= radius * radius
}

fn collect_widget_rects(state: &AppState) -> Vec<Rect> {
    let mut rects = Vec::with_capacity(19);
    rects.extend(state.tag_rects.iter().copied());
    rects.push(state.layout_button_rect);
    rects.extend(state.layout_option_rects.iter().copied());
    rects.push(state.ss_rect);
    rects.push(state.time_rect);
    rects.push(state.audio_rect);
    rects.push(state.theme_rect);
    rects.push(state.mem_rect);
    rects.push(state.cpu_rect);
    rects.push(state.mon_rect);
    rects.retain(|rect| rect.w > 0 && rect.h > 0);
    rects
}

fn apply_background_opacity(
    data: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    opacity: f64,
    widget_radius: f64,
    widget_rects: &[Rect],
) {
    if opacity >= 0.999 {
        return;
    }

    for y in 0..height {
        for x in 0..width {
            if widget_rects
                .iter()
                .any(|rect| is_inside_round_rect(x, y, rect, widget_radius))
            {
                continue;
            }

            let offset = y * stride + x * 4;
            set_argb32_pixel_opacity(&mut data[offset..offset + 4], opacity);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    Tick,
    SharedUpdated,
}

struct CairoBackBuffer {
    width: u32,
    height: u32,
    image: ImageSurface,
}

impl CairoBackBuffer {
    fn new(width: u32, height: u32) -> Result<Self> {
        let image = ImageSurface::create(Format::ARgb32, width as i32, height as i32)?;
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
        self.image = ImageSurface::create(Format::ARgb32, width as i32, height as i32)?;
        self.width = width;
        self.height = height;
        Ok(())
    }
}

fn spawn_tick_thread(proxy: EventLoopProxy<UserEvent>) {
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(1000));
            let _ = proxy.send_event(UserEvent::Tick);
        }
    });
}

fn spawn_shared_thread(proxy: EventLoopProxy<UserEvent>, shared_efd: Option<i32>) {
    if let Some(efd) = shared_efd {
        thread::spawn(move || {
            let mut buf8 = [0u8; 8];
            let mut pfd = libc::pollfd {
                fd: efd,
                events: libc::POLLIN,
                revents: 0,
            };
            loop {
                let pr = unsafe { libc::poll(&mut pfd as *mut libc::pollfd, 1, -1) };
                if pr < 0 {
                    let err = std::io::Error::last_os_error();
                    if let Some(code) = err.raw_os_error() {
                        if code == libc::EINTR {
                            continue;
                        }
                    }
                    warn!("[shared-thread] poll error: {}", err);
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }
                if (pfd.revents & libc::POLLIN) != 0 {
                    let r = unsafe { libc::read(efd, buf8.as_mut_ptr() as *mut _, buf8.len()) };
                    if r == 8 {
                        let _ = proxy.send_event(UserEvent::SharedUpdated);
                    } else if r < 0 {
                        let err = std::io::Error::last_os_error();
                        if let Some(code) = err.raw_os_error() {
                            if code == libc::EINTR {
                                continue;
                            }
                        }
                        warn!("[shared-thread] eventfd read error: {}", err);
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
        });
    }
}

struct App {
    window: Rc<Window>,
    back: CairoBackBuffer,
    soft_surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,

    colors: xbar_core::Colors,
    cfg: BarConfig,
    font: FontDescription,
    background_opacity: f64,
    state: AppState,

    scale_factor: f64,
    logical_size: LogicalSize<f64>,

    last_clock_update: Instant,
    last_monitor_update: Instant,
    last_cursor_pos_px: Option<(i32, i32)>,
}

impl App {
    fn new(
        window: Rc<Window>,
        logical_size: LogicalSize<f64>,
        scale: f64,
        shared_buffer: Option<Arc<SharedRingBuffer>>,
    ) -> Result<Self> {
        let cfg = BarConfig {
            bar_height: 38,
            padding_x: 10.0,
            padding_y: 6.0,
            tag_spacing: 6.0,
            pill_hpadding: 10.0,
            pill_radius: 12.0,
            shape_style: ShapeStyle::Pill,
            time_icon: "🕐",
            screenshot_label: "📸",
            tag_labels: ["🖥", "🌐", "📁", "💬", "📝", "🎵", "⚙", "📊", "🏠"],
            theme_dark_label: "🌙",
            theme_light_label: "☀️",
            monitor_labels: ["🥇", "🥈", "🥉", "❔"],
            volume_label: "🔊",
            mute_label: "🔇",
            show_audio: true,
            show_theme_toggle: true,
            volume_step: 5,
        };

        let font_str = env::var("XBAR_FONT").unwrap_or_else(|_| "monospace 11".to_string());
        let font = FontDescription::from_string(&font_str);
        let background_opacity = background_opacity_from_env();

        let mut state = AppState::new(shared_buffer);
        state.theme_mode = ThemeMode::Dark;
        let colors = tuned_colors_for_theme(state.theme_mode);

        let width_px = (logical_size.width * scale).round() as u32;
        let height_px = (logical_size.height * scale).round() as u32;

        let soft_ctx = softbuffer::Context::new(window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer::Context::new: {}", e))?;
        let mut soft_surface = softbuffer::Surface::new(&soft_ctx, window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer::Surface::new: {}", e))?;

        if let (Some(w), Some(h)) = (NonZeroU32::new(width_px), NonZeroU32::new(height_px)) {
            let _ = soft_surface.resize(w, h);
        }

        let back = CairoBackBuffer::new(width_px, height_px)?;

        Ok(Self {
            window,
            back,
            colors,
            cfg,
            font,
            background_opacity,
            state,
            scale_factor: scale,
            logical_size,
            last_clock_update: Instant::now(),
            last_monitor_update: Instant::now(),
            last_cursor_pos_px: None,
            soft_surface,
        })
    }

    fn redraw(&mut self) -> Result<()> {
        let width_px = (self.logical_size.width * self.scale_factor).round() as u32;
        let height_px = (self.logical_size.height * self.scale_factor).round() as u32;

        self.back.ensure_size(width_px, height_px)?;
        let back = &mut self.back;

        {
            let cr = Context::new(&back.image)?;
            cr.save()?;
            cr.set_source_rgba(0.0, 0.0, 0.0, 0.0);
            cr.set_operator(xbar_core::cairo::Operator::Source);
            cr.paint()?;
            cr.restore()?;

            let w_u16 = width_px.min(u16::MAX as u32) as u16;
            let h_u16 = height_px.min(u16::MAX as u32) as u16;
            draw_bar(
                &cr,
                w_u16,
                h_u16,
                &self.colors,
                &mut self.state,
                &self.font,
                &self.cfg,
            )?;
        }

        back.image.flush();
        let stride = back.image.stride() as usize;
        let mut data = back.image.data()?;
        let w = width_px as usize;
        let h = height_px as usize;
        let widget_rects = collect_widget_rects(&self.state);
        apply_background_opacity(
            &mut data[..],
            stride,
            w,
            h,
            self.background_opacity,
            self.cfg.pill_radius,
            &widget_rects,
        );

        use bytemuck::cast_slice;
        let mut buf = self
            .soft_surface
            .buffer_mut()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        if stride == w * 4 {
            let src_u32: &[u32] = cast_slice(&data[..h * stride]);
            buf[..w * h].copy_from_slice(src_u32);
        } else {
            for y in 0..h {
                let row = &data[y * stride..y * stride + w * 4];
                let src_u32: &[u32] = cast_slice(row);
                let dst_row = &mut buf[y * w..(y + 1) * w];
                dst_row.copy_from_slice(src_u32);
            }
        }
        buf.present().map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok(())
    }

    fn update_hover_and_redraw(&mut self, px: i32, py: i32) {
        if self.state.update_hover(px as i16, py as i16) {
            if let Err(e) = self.redraw() {
                warn!("redraw error (hover): {}", e);
            }
        }
    }

    fn handle_button(&mut self, px: i32, py: i32, button_id: u8) {
        let prev_theme = self.state.theme_mode;
        if self.state.handle_buttons(px as i16, py as i16, button_id) {
            if self.state.theme_mode != prev_theme {
                self.colors = tuned_colors_for_theme(self.state.theme_mode);
            }
            if let Err(e) = self.redraw() {
                warn!("redraw error (button): {}", e);
            }
        }
    }

    fn resize(&mut self, new_size: LogicalSize<f64>) {
        self.logical_size = new_size;
        let w = (self.logical_size.width * self.scale_factor).round() as u32;
        let h = (self.logical_size.height * self.scale_factor).round() as u32;
        if let (Some(wnz), Some(hnz)) = (NonZeroU32::new(w), NonZeroU32::new(h)) {
            let _ = self.soft_surface.resize(wnz, hnz);
        }
        if let Err(e) = self.redraw() {
            warn!("redraw error (resize): {}", e);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let shared_path = args.iter().skip(1).last().cloned().unwrap_or_default();

    if let Err(e) = initialize_logging("tao_softbuffer_bar", &shared_path) {
        eprintln!("Failed to initialize logging: {}", e);
        std::process::exit(1);
    }

    let shared_buffer = SharedRingBuffer::create_shared_ring_buffer_aux(&shared_path).map(Arc::new);
    let shared_efd = spawn_shared_eventfd_notifier(shared_buffer.clone(), false);

    // 修复 1: 使用 EventLoopBuilder 创建带用户事件的 EventLoop
    let event_loop: EventLoop<UserEvent> = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();

    spawn_tick_thread(proxy.clone());
    spawn_shared_thread(proxy.clone(), shared_efd);

    // 修复 2: 显式标注 primary_monitor 的类型为 Option<MonitorHandle>
    let primary_monitor: Option<MonitorHandle> = event_loop.primary_monitor();
    let scale_factor = primary_monitor
        .as_ref()
        .map(|m: &MonitorHandle| m.scale_factor())
        .unwrap_or(1.0);
    let screen_size: PhysicalSize<u32> = primary_monitor
        .as_ref()
        .map(|m: &MonitorHandle| m.size())
        .unwrap_or(PhysicalSize::new(1920, 1080));

    let height_px = 38u32;
    let logical_size = LogicalSize::new(
        screen_size.width as f64 / scale_factor,
        height_px as f64 / scale_factor,
    );

    let window = WindowBuilder::new()
        .with_title("tao_softbuffer_bar")
        .with_inner_size(logical_size)
        .with_decorations(false)
        .with_resizable(true)
        .with_visible(true)
        .with_transparent(true)
        .build(&event_loop)
        .map(Rc::new)
        .expect("Failed to build window"); // 修复 3: main 不再返回 Result，改用 expect

    let mut app = App::new(window.clone(), logical_size, scale_factor, shared_buffer)
        .expect("Failed to initialize App");

    if let Err(e) = app.redraw() {
        warn!("Initial redraw error: {}", e);
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::UserEvent(UserEvent::Tick) => {
                if app.last_clock_update.elapsed() >= Duration::from_secs(1) {
                    app.last_clock_update = Instant::now();
                    app.window.request_redraw();
                }
                if app.last_monitor_update.elapsed() >= Duration::from_secs(2) {
                    app.state.system_monitor.update_if_needed();
                    app.state.audio_manager.update_if_needed();
                    app.last_monitor_update = Instant::now();
                    app.window.request_redraw();
                }
            }
            Event::UserEvent(UserEvent::SharedUpdated) => {
                app.window.request_redraw();
                if let Some(buf_arc) = app.state.shared_buffer.as_ref().cloned() {
                    match buf_arc.try_read_latest_message() {
                        Ok(Some(msg)) => {
                            app.state.update_from_shared(msg);
                        }
                        Ok(None) => {}
                        Err(e) => warn!("Shared try_read_latest_message failed: {}", e),
                    }
                }
            }
            Event::WindowEvent {
                event: window_event,
                ..
            } => match window_event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(new_size) => {
                    let logical = new_size.to_logical::<f64>(app.scale_factor);
                    app.resize(logical);
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                } => {
                    app.scale_factor = scale_factor;
                    let logical = new_inner_size.to_logical::<f64>(app.scale_factor);
                    app.resize(logical);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let px = position.x.round() as i32;
                    let py = position.y.round() as i32;
                    app.last_cursor_pos_px = Some((px, py));
                    app.update_hover_and_redraw(px, py);
                }
                WindowEvent::CursorLeft { .. } => {
                    app.state.clear_hover();
                    app.window.request_redraw();
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    use tao::event::MouseScrollDelta;
                    if let Some((px, py)) = app.last_cursor_pos_px {
                        let dy = match delta {
                            MouseScrollDelta::LineDelta(_x, y) => y as f64,
                            MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => y,
                            _ => 0.0,
                        };
                        let button_id = if dy > 0.0 {
                            4
                        } else if dy < 0.0 {
                            5
                        } else {
                            0
                        };
                        if button_id != 0 {
                            app.handle_button(px, py, button_id);
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    use tao::event::{ElementState, MouseButton};
                    if state == ElementState::Pressed {
                        if let Some((px, py)) = app.last_cursor_pos_px {
                            let button_id = match button {
                                MouseButton::Left => 1,
                                MouseButton::Middle => 2,
                                MouseButton::Right => 3,
                                MouseButton::Other(n) => n as u8,
                                _ => 0,
                            };
                            app.handle_button(px, py, button_id);
                        }
                    }
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                if let Err(e) = app.redraw() {
                    warn!("redraw error (RedrawRequested): {}", e);
                }
            }
            _ => {}
        }
    });
}
