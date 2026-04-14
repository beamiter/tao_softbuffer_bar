use anyhow::Result;
use cairo::{Context, Format, ImageSurface};
use log::warn;
use pango::FontDescription;
use shared_structures::SharedRingBuffer;
use std::env;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tao::event_loop::EventLoopBuilder;

use tao::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopProxy},
    window::{Window, WindowBuilder, WindowId},
};

use xbar_core::{
    AppState, BarConfig, Color, ShapeStyle, ThemeMode, colors_for_theme, draw_bar,
    initialize_logging, spawn_shared_eventfd_notifier,
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

type SbSurface = softbuffer::Surface<Rc<Window>, Rc<Window>>;

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    SharedUpdated,
    Tick,
}

// Cairo 后备缓冲：ImageSurface + Context
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
    #[allow(dead_code)]
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
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

// 每秒对齐 tick 线程：按秒对齐，发送 UserEvent::Tick
fn spawn_tick_thread(proxy: EventLoopProxy<UserEvent>) {
    thread::spawn(move || {
        loop {
            // 当前系统时间的纳秒偏移
            let now_sys = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0));
            let subns = now_sys.subsec_nanos() as u64;

            // 睡到下一秒边界，避免频繁唤醒
            let remain_ns = 1_000_000_000u64.saturating_sub(subns).max(1);
            // 注意：标准库的 sleep 最小精度依赖平台，这里无需完全精确
            thread::sleep(Duration::from_nanos(remain_ns));

            // 发送 Tick 事件
            let _ = proxy.send_event(UserEvent::Tick);
        }
    });
}

struct App {
    // 运行期资源
    window: Option<Rc<Window>>,
    window_id: Option<WindowId>,
    back: Option<CairoBackBuffer>,

    // 配置与状态
    colors: xbar_core::Colors,
    cfg: BarConfig,
    font: FontDescription,
    state: AppState,

    // DPI/尺寸
    scale_factor: f64,
    logical_size: LogicalSize<f64>,

    // 更新时间控制
    last_monitor_update: Instant,

    // 记录最近一次鼠标物理坐标（像素）
    last_cursor_pos_px: Option<(i32, i32)>,

    soft_surface: Option<SbSurface>,
}

impl App {
    fn new(
        shared_buffer: Option<Arc<SharedRingBuffer>>,
        logical_size: LogicalSize<f64>,
        scale: f64,
    ) -> Self {
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

        // 字体（尽量不依赖 Nerd Font；可用 XBAR_FONT 覆盖）
        let font_str = env::var("XBAR_FONT").unwrap_or_else(|_| "monospace 11".to_string());
        let font = FontDescription::from_string(&font_str);

        let mut state = AppState::new(shared_buffer);
        state.theme_mode = ThemeMode::Dark;
        let colors = tuned_colors_for_theme(state.theme_mode);
        Self {
            window: None,
            window_id: None,
            back: None,
            colors,
            cfg,
            font,
            state,
            scale_factor: scale,
            logical_size,
            last_monitor_update: Instant::now(),
            last_cursor_pos_px: None,
            soft_surface: None,
        }
    }

    fn ensure_init_window(&mut self, target: &tao::event_loop::EventLoopWindowTarget<UserEvent>) {
        if self.window.is_some() {
            return;
        }

        // 初始尺寸：以主显示器宽度 + bar 高度
        let primary = target
            .primary_monitor()
            .or_else(|| target.available_monitors().next());
        let scale = primary.as_ref().map(|m| m.scale_factor()).unwrap_or(1.0);
        self.scale_factor = scale;

        let screen_size: PhysicalSize<u32> = primary
            .as_ref()
            .map(|m| m.size())
            .unwrap_or(PhysicalSize::new(1920, 1080));
        let width_px = screen_size.width;
        let height_px = self.cfg.bar_height as u32;

        self.logical_size = LogicalSize::new(
            width_px as f64 / self.scale_factor,
            height_px as f64 / self.scale_factor,
        );

        let window = WindowBuilder::new()
            .with_title("tao_softbuffer_bar")
            .with_inner_size(self.logical_size)
            .with_decorations(false)
            .with_resizable(true)
            .with_visible(true)
            .with_transparent(false)
            .build(target)
            .expect("create window failed");
        let window = Rc::new(window);

        // softbuffer Context 与 Surface
        let soft_ctx = softbuffer::Context::new(window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer::Context::new: {}", e))
            .expect("softbuffer context");
        let mut soft_surface = softbuffer::Surface::new(&soft_ctx, window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer::Surface::new: {}", e))
            .expect("softbuffer surface");

        let width_px = (self.logical_size.width * self.scale_factor).round() as u32;
        let height_px = (self.logical_size.height * self.scale_factor).round() as u32;
        if let (Some(w), Some(h)) = (NonZeroU32::new(width_px), NonZeroU32::new(height_px)) {
            let _ = soft_surface.resize(w, h);
        }

        // Cairo back buffer
        let back = CairoBackBuffer::new(width_px, height_px).expect("cairo back buffer failed");

        self.window_id = Some(window.id());
        self.window = Some(window);
        self.back = Some(back);
        self.soft_surface = Some(soft_surface);

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn redraw(&mut self) -> Result<()> {
        if self.window.is_none() {
            return Ok(());
        }
        let width_px = (self.logical_size.width * self.scale_factor).round() as u32;
        let height_px = (self.logical_size.height * self.scale_factor).round() as u32;

        // back buffer 尺寸保证
        if self.back.is_none() {
            self.back = Some(CairoBackBuffer::new(width_px, height_px)?);
        } else {
            self.back
                .as_mut()
                .unwrap()
                .ensure_size(width_px, height_px)?;
        }
        let back = self.back.as_mut().unwrap();

        // Cairo 绘制
        {
            let cr = Context::new(&back.image)?;
            cr.save()?;
            cr.set_source_rgba(0.0, 0.0, 0.0, 1.0);
            cr.set_operator(cairo::Operator::Source);
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

        // 像素提交
        back.image.flush();
        let stride = back.image.stride() as usize;
        let data = back.image.data()?; // &[u8]
        let w = width_px as usize;
        let h = height_px as usize;

        let surface = match self.soft_surface.as_mut() {
            Some(s) => s,
            None => return Ok(()),
        };

        use bytemuck::cast_slice;
        let mut buf = surface.buffer_mut().map_err(|e| anyhow::anyhow!("{}", e))?;
        if stride == w * 4 {
            let src_u32: &[u32] = cast_slice(&data[..h * stride]); // BGRA 小端
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
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    fn handle_button(&mut self, px: i32, py: i32, button_id: u8) {
        let prev_theme = self.state.theme_mode;
        if self.state.handle_buttons(px as i16, py as i16, button_id) {
            if self.state.theme_mode != prev_theme {
                self.colors = tuned_colors_for_theme(self.state.theme_mode);
            }
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }
}

fn main() -> Result<()> {
    // 参数
    let args: Vec<String> = env::args().collect();
    let shared_path = args.get(1).cloned().unwrap_or_default();

    // 日志
    if let Err(e) = initialize_logging("tao_softbuffer_bar", &shared_path) {
        eprintln!("Failed to initialize logging: {}", e);
        std::process::exit(1);
    }

    // 共享内存与通知
    let shared_buffer = SharedRingBuffer::create_shared_ring_buffer_aux(&shared_path).map(Arc::new);
    let shared_efd = spawn_shared_eventfd_notifier(shared_buffer.clone(), false);

    // 事件循环与代理（tao）
    let event_loop: EventLoop<UserEvent> = EventLoopBuilder::with_user_event().build();
    let proxy = event_loop.create_proxy();

    // 后台线程：SharedUpdated 和 Tick
    spawn_shared_thread(proxy.clone(), shared_efd);
    spawn_tick_thread(proxy.clone());

    // 初始逻辑尺寸，实际初始化在 NewEvents::Init 中完成
    let logical_size = LogicalSize::new(800.0, 40.0);
    let mut app = App::new(shared_buffer, logical_size, 1.0);

    // 运行（闭包式）
    event_loop.run(move |event, target, control_flow| {
        // 始终等待事件（共享通知 + Tick + 窗口事件）
        *control_flow = ControlFlow::Wait;

        match event {
            Event::NewEvents(StartCause::Init) => {
                // 首次进入事件循环时创建窗口与渲染资源
                app.ensure_init_window(target);
            }

            Event::Resumed => {
                // 某些平台用 Resumed 作为初始化契机
                app.ensure_init_window(target);
            }

            Event::UserEvent(ue) => match ue {
                UserEvent::SharedUpdated => {
                    let mut need_redraw = false;
                    if let Some(buf_arc) = app.state.shared_buffer.as_ref().cloned() {
                        match buf_arc.try_read_latest_message() {
                            Ok(Some(msg)) => {
                                log::trace!("redraw by msg: {:?}", msg);
                                app.state.update_from_shared(msg);
                                need_redraw = true;
                            }
                            Ok(None) => { /* 没有消息 */ }
                            Err(e) => {
                                warn!("Shared try_read_latest_message failed: {}", e);
                            }
                        }
                    }
                    if need_redraw {
                        if let Some(w) = &app.window {
                            w.request_redraw();
                        }
                    }
                }
                UserEvent::Tick => {
                    let mut need_redraw = false;

                    // 1. 检查时间字符串是否变化
                    let new_time = app.state.format_time();
                    if new_time != app.state.last_time_string {
                        app.state.last_time_string = new_time;
                        log::trace!("redraw by time update");
                        need_redraw = true;
                    }

                    // 2. 检查系统监控数据及音频数据是否更新
                    if app.last_monitor_update.elapsed() >= Duration::from_secs(2) {
                        need_redraw |= app.state.system_monitor.update_if_needed();
                        need_redraw |= app.state.audio_manager.update_if_needed();
                        app.last_monitor_update = Instant::now();
                        if need_redraw {
                            log::trace!("redraw by system/audio update");
                        }
                    }

                    if need_redraw {
                        if let Some(w) = &app.window {
                            w.request_redraw();
                        }
                    }
                }
            },

            Event::WindowEvent {
                window_id, event, ..
            } => {
                if Some(window_id) != app.window_id {
                    return;
                }
                let window = match &app.window {
                    Some(w) => w,
                    None => return,
                };

                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Resized(new_size) => {
                        app.scale_factor = window.scale_factor();
                        app.logical_size = new_size.to_logical::<f64>(app.scale_factor);
                        if let Some(surface) = app.soft_surface.as_mut() {
                            let w = (app.logical_size.width * app.scale_factor).round() as u32;
                            let h = (app.logical_size.height * app.scale_factor).round() as u32;
                            if let (Some(wnz), Some(hnz)) = (NonZeroU32::new(w), NonZeroU32::new(h))
                            {
                                let _ = surface.resize(wnz, hnz);
                            }
                        }
                        if let Err(e) = app.redraw() {
                            warn!("redraw error (Resized): {}", e);
                        }
                    }
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        app.scale_factor = scale_factor;
                        app.logical_size = window.inner_size().to_logical::<f64>(app.scale_factor);
                        if let Some(surface) = app.soft_surface.as_mut() {
                            let w = (app.logical_size.width * app.scale_factor).round() as u32;
                            let h = (app.logical_size.height * app.scale_factor).round() as u32;
                            if let (Some(wnz), Some(hnz)) = (NonZeroU32::new(w), NonZeroU32::new(h))
                            {
                                let _ = surface.resize(wnz, hnz);
                            }
                        }
                        if let Err(e) = app.redraw() {
                            warn!("redraw error (ScaleFactorChanged): {}", e);
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let px = position.x.round() as i32;
                        let py = position.y.round() as i32;
                        app.last_cursor_pos_px = Some((px, py));
                        app.update_hover_and_redraw(px, py);
                        log::trace!("cursor px={}, py={}", px, py);
                    }
                    WindowEvent::CursorLeft { .. } => {
                        app.state.clear_hover();
                        if let Some(w) = &app.window {
                            w.request_redraw();
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        use tao::event::MouseScrollDelta;
                        if let Some((px, py)) = app.last_cursor_pos_px {
                            let dy = match delta {
                                MouseScrollDelta::LineDelta(_x, y) => y as f64,
                                MouseScrollDelta::PixelDelta(pos) => pos.y,
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
                                    _ => todo!(),
                                };
                                app.handle_button(px, py, button_id);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                log::trace!("[RedrawRequested]");
                if let Err(e) = app.redraw() {
                    warn!("redraw error (RedrawRequested): {}", e);
                }
            }

            Event::LoopDestroyed => {
                // 资源在 Drop 时释放，这里无需处理
            }

            // 旧的 StartCause::ResumeTimeReached 已不再使用
            _ => {}
        }
    });
}
