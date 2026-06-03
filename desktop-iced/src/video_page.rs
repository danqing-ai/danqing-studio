#![allow(dead_code)]
use dq_components::{
    dq_control_button, dq_header_button, dq_pick_list,
    dq_progress_bar, dq_slider_with_input, dq_text_input,
    dq_text_input_multiline, dq_prompt_editor, section_card, surface_card,
    chevron_icon, phosphor_icon,
    LogLine, ModeTabOption, PhosphorIcon, default_logs, log_panel,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{column, container, row, scrollable, text, text_editor, Space};
use iced::{Alignment, Element, Length, Task};
use serde_json::json;
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum Message {
    NoOp, ModeChanged(VideoMode), ModelSelected(VideoModelOption),
    TitleChanged(String), PromptEdited(text_editor::Action), NegativePromptEdited(text_editor::Action),
    ToggleNegativeOpen, ToggleAdvancedOpen,
    SeedChanged(String), RandomizeSeed,
    StepsChanged(f32), StepsInputChanged(String), GuidanceChanged(f32), GuidanceInputChanged(String),
    ShiftChanged(f32), ShiftInputChanged(String),
    WidthSelected(WidthOption), HeightSelected(HeightOption),
    NumFramesChanged(f32), NumFramesInputChanged(String), FpsChanged(f32), FpsInputChanged(String),
    LoraSelected(LoraOption), CommercialToggled(bool),
    RestoreDefaults, ClearLogs, Generate, CancelGeneration, GenerateStep,
    GenerateProgress { progress: u8, step: u32, total: u32, phase: String },
    GenerateComplete { result_url: String }, GenerateFailed { error: String },
    UploadSourceVideo, UploadStartImage, UploadTailImage,
    SourceVideoSelected(String), StartImageSelected(String), TailImageSelected(String),
    ClearSourceVideo, ClearStartImage, ClearTailImage,
    UpscaleFactorSelected(UpscaleFactorOption), DenoiseChanged(f32), DenoiseInputChanged(String),
    MemoryPoll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoMode { #[default] TextToVideo, ImageToVideo, Upscale }
impl VideoMode {
    const ALL: &'static [VideoMode] = &[VideoMode::TextToVideo, VideoMode::ImageToVideo, VideoMode::Upscale];
    fn label(self) -> &'static str { match self { VideoMode::TextToVideo => "文生视频", VideoMode::ImageToVideo => "图生视频", VideoMode::Upscale => "视频修复" } }
    pub fn required_action(self) -> &'static str { match self { VideoMode::TextToVideo => "generate", VideoMode::ImageToVideo => "edit", VideoMode::Upscale => "upscale" } }
}

#[derive(Debug, Clone)]
pub enum VideoModelOption {
    Static { id: &'static str, label: &'static str },
    Dynamic { id: String, label: String, name: String, actions: Vec<String>, installed: bool },
}
impl Default for VideoModelOption { fn default() -> Self { VideoModelOption::Static { id: "ltx-video", label: "LTX Video" } } }
impl PartialEq for VideoModelOption {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) { (VideoModelOption::Static { id: a, .. }, VideoModelOption::Static { id: b, .. }) => a == b, (VideoModelOption::Dynamic { id: a, .. }, VideoModelOption::Dynamic { id: b, .. }) => a == b, _ => false }
    }
}
impl Eq for VideoModelOption {}
impl std::fmt::Display for VideoModelOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.label()) } }
impl VideoModelOption {
    pub fn id(&self) -> String { match self { VideoModelOption::Static { id, .. } => id.to_string(), VideoModelOption::Dynamic { id, .. } => id.clone() } }
    pub fn label(&self) -> String { match self { VideoModelOption::Static { label, .. } => label.to_string(), VideoModelOption::Dynamic { label, .. } => label.clone() } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WidthOption { W480, #[default] W720, W832, W1080 }
impl WidthOption {
    const ALL: &'static [WidthOption] = &[WidthOption::W480, WidthOption::W720, WidthOption::W832, WidthOption::W1080];
    pub fn value(self) -> u16 { match self { WidthOption::W480 => 480, WidthOption::W720 => 720, WidthOption::W832 => 832, WidthOption::W1080 => 1080 } }
}
impl std::fmt::Display for WidthOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.value()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeightOption { H272, H480, #[default] H480Ratio, H544, H720 }
impl HeightOption {
    const ALL: &'static [HeightOption] = &[HeightOption::H272, HeightOption::H480, HeightOption::H480Ratio, HeightOption::H544, HeightOption::H720];
    pub fn value(self) -> u16 { match self { HeightOption::H272 => 272, HeightOption::H480 => 480, HeightOption::H480Ratio => 480, HeightOption::H544 => 544, HeightOption::H720 => 720 } }
}
impl std::fmt::Display for HeightOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.value()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleFactorOption { #[default] X2, X4 }
impl UpscaleFactorOption {
    const ALL: &'static [UpscaleFactorOption] = &[UpscaleFactorOption::X2, UpscaleFactorOption::X4];
    fn label(&self) -> &'static str { match self { UpscaleFactorOption::X2 => "2×", UpscaleFactorOption::X4 => "4×" } }
    pub fn value(&self) -> u8 { match self { UpscaleFactorOption::X2 => 2, UpscaleFactorOption::X4 => 4 } }
}
impl std::fmt::Display for UpscaleFactorOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoraOption { #[default] None, Portrait, Anime }
impl LoraOption {
    const ALL: &'static [LoraOption] = &[LoraOption::None, LoraOption::Portrait, LoraOption::Anime];
    fn label(&self) -> &'static str { match self { LoraOption::None => "不使用 LoRA", LoraOption::Portrait => "人像增强 LoRA", LoraOption::Anime => "动漫风格 LoRA" } }
}
impl std::fmt::Display for LoraOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GenerateState { #[default] Idle, Submitting, Generating { progress: u8, step: u32, total: u32, phase: String }, Done }
#[derive(Debug, Clone, Default)]
pub enum SourceVideoState { #[default] Empty, Uploaded(String) }

#[derive(Debug, Clone)]
pub struct VideoPage {
    pub mode: VideoMode, pub mode_tabs: Vec<ModeTabOption<VideoMode>>,
    pub model: VideoModelOption, pub title: String,
    pub prompt: text_editor::Content, pub negative_prompt: text_editor::Content,
    pub negative_open: bool, pub advanced_open: bool,
    pub seed: String, pub steps: f32, pub steps_input: String,
    pub guidance: f32, pub guidance_input: String, pub shift: f32, pub shift_input: String,
    pub width: WidthOption, pub height: HeightOption,
    pub num_frames: f32, pub num_frames_input: String, pub fps: f32, pub fps_input: String,
    pub lora: LoraOption, pub commercial_only: bool,
    pub generate_state: GenerateState, pub validation_error: Option<String>,
    pub logs: Vec<LogLine>,
    pub source_video: SourceVideoState, pub start_image: SourceVideoState, pub tail_image: SourceVideoState,
    pub upscale_factor: UpscaleFactorOption, pub denoise: f32, pub denoise_input: String,
    pub memory_info: crate::create_page::MemoryInfo,
    pub generated_video_path: Option<String>, pub available_models: Vec<VideoModelOption>,
}

impl Default for VideoPage {
    fn default() -> Self {
        Self {
            mode: VideoMode::TextToVideo,
            mode_tabs: VideoMode::ALL.iter().map(|m| ModeTabOption { label: m.label().into(), value: *m }).collect(),
            model: VideoModelOption::default(), title: String::new(),
            prompt: text_editor::Content::with_text("一只小猫在草地上打滚，夕阳余晖"),
            negative_prompt: text_editor::Content::with_text("模糊，抖动，低质量"),
            negative_open: false, advanced_open: false,
            seed: "随机".into(), steps: 50.0, steps_input: "50".into(),
            guidance: 5.0, guidance_input: "5.0".into(), shift: 5.0, shift_input: "5.0".into(),
            width: WidthOption::W720, height: HeightOption::H480Ratio,
            num_frames: 49.0, num_frames_input: "49".into(), fps: 8.0, fps_input: "8".into(),
            lora: LoraOption::None, commercial_only: false,
            generate_state: GenerateState::Idle, validation_error: None,
            logs: default_logs(),
            source_video: SourceVideoState::Empty, start_image: SourceVideoState::Empty, tail_image: SourceVideoState::Empty,
            upscale_factor: UpscaleFactorOption::X4, denoise: 0.3, denoise_input: "0.3".into(),
            memory_info: crate::create_page::MemoryInfo::default(),
            generated_video_path: None, available_models: Vec::new(),
        }
    }
}

impl VideoPage {
    pub fn push_log(&mut self, message: String) {
        let now = chrono::Local::now();
        self.logs.push(LogLine { time: now.format("%H:%M:%S").to_string(), message });
        if self.logs.len() > 20 { self.logs.remove(0); }
    }

    pub fn build_request(&self, source_asset_id: Option<String>, start_asset_id: Option<String>, tail_asset_id: Option<String>) -> (&'static str, serde_json::Value) {
        let seed = self.seed.parse::<i64>().ok();
        let size = format!("{}x{}", self.width.value(), self.height.value());
        match self.mode {
            VideoMode::TextToVideo => ("/api/videos/generations", json!({
                "model": self.model.id(), "title": self.title, "prompt": self.prompt.text(),
                "negative_prompt": self.negative_prompt.text(), "size": size,
                "num_frames": self.num_frames as i32, "fps": self.fps as i32,
                "steps": self.steps as i32, "guidance": self.guidance, "shift": self.shift,
                "seed": seed, "adapters": [], "priority": "normal", "metadata": {},
            })),
            VideoMode::ImageToVideo => ("/api/videos/edits", json!({
                "model": self.model.id(), "operation": "animate",
                "source_asset_id": start_asset_id, "tail_asset_id": tail_asset_id,
                "title": self.title, "prompt": self.prompt.text(),
                "negative_prompt": self.negative_prompt.text(), "size": size,
                "num_frames": self.num_frames as i32, "fps": self.fps as i32,
                "steps": self.steps as i32, "guidance": self.guidance, "shift": self.shift,
                "seed": seed, "adapters": [], "priority": "normal", "metadata": {},
            })),
            VideoMode::Upscale => ("/api/videos/upscales", json!({
                "model": self.model.id(), "source_asset_id": source_asset_id,
                "scale": self.upscale_factor.value(), "denoise": self.denoise,
                "tile_size": 1024, "temporal_window": 5, "fps": self.fps as i32,
                "priority": "normal", "metadata": {},
            })),
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::NoOp => Task::none(),
            Message::ModeChanged(m) => { self.mode = m; Task::none() }
            Message::ModelSelected(m) => { self.model = m; self.push_log(format!("已切换模型：{}", self.model.label())); Task::none() }
            Message::TitleChanged(s) => { self.title = s; Task::none() }
            Message::PromptEdited(a) => { self.prompt.perform(a); self.validation_error = None; Task::none() }
            Message::NegativePromptEdited(a) => { self.negative_prompt.perform(a); Task::none() }
            Message::ToggleNegativeOpen => { self.negative_open = !self.negative_open; Task::none() }
            Message::ToggleAdvancedOpen => { self.advanced_open = !self.advanced_open; Task::none() }
            Message::SeedChanged(s) => { self.seed = s; Task::none() }
            Message::RandomizeSeed => { self.seed = format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()); Task::none() }
            Message::StepsChanged(v) => { self.steps = v.clamp(1.0, 100.0); self.steps_input = format!("{:.0}", self.steps); Task::none() }
            Message::StepsInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.steps_input = s; if let Some(v) = v { self.steps = v.clamp(1.0, 100.0); } Task::none() }
            Message::GuidanceChanged(v) => { self.guidance = v.clamp(0.0, 30.0); self.guidance_input = format!("{:.1}", self.guidance); Task::none() }
            Message::GuidanceInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.guidance_input = s; if let Some(v) = v { self.guidance = v.clamp(0.0, 30.0); } Task::none() }
            Message::ShiftChanged(v) => { self.shift = v.clamp(0.0, 20.0); self.shift_input = format!("{:.1}", self.shift); Task::none() }
            Message::ShiftInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.shift_input = s; if let Some(v) = v { self.shift = v.clamp(0.0, 20.0); } Task::none() }
            Message::WidthSelected(w) => { self.width = w; Task::none() }
            Message::HeightSelected(h) => { self.height = h; Task::none() }
            Message::NumFramesChanged(v) => { self.num_frames = v.clamp(1.0, 300.0); self.num_frames_input = format!("{:.0}", self.num_frames); Task::none() }
            Message::NumFramesInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.num_frames_input = s; if let Some(v) = v { self.num_frames = v.clamp(1.0, 300.0); } Task::none() }
            Message::FpsChanged(v) => { self.fps = v.clamp(1.0, 60.0); self.fps_input = format!("{:.0}", self.fps); Task::none() }
            Message::FpsInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.fps_input = s; if let Some(v) = v { self.fps = v.clamp(1.0, 60.0); } Task::none() }
            Message::LoraSelected(l) => { self.lora = l; Task::none() }
            Message::CommercialToggled(v) => { self.commercial_only = v; Task::none() }
            Message::RestoreDefaults => { self.steps = 50.0; self.steps_input = "50".into(); self.guidance = 5.0; self.guidance_input = "5.0".into(); self.shift = 5.0; self.shift_input = "5.0".into(); self.width = WidthOption::W720; self.height = HeightOption::H480Ratio; self.num_frames = 49.0; self.num_frames_input = "49".into(); self.fps = 8.0; self.fps_input = "8".into(); self.seed = "随机".into(); self.lora = LoraOption::None; self.upscale_factor = UpscaleFactorOption::X4; self.denoise = 0.3; self.denoise_input = "0.3".into(); self.push_log("已恢复默认配置".into()); Task::none() }
            Message::ClearLogs => { self.logs.clear(); Task::none() }
            Message::Generate => {
                if self.mode != VideoMode::Upscale && self.prompt.text().trim().is_empty() { self.validation_error = Some("请输入提示词后再生成".into()); return Task::none(); }
                if self.mode == VideoMode::ImageToVideo && matches!(self.start_image, SourceVideoState::Empty) { self.validation_error = Some("请上传起始图片后再生成".into()); return Task::none(); }
                if self.mode == VideoMode::Upscale && matches!(self.source_video, SourceVideoState::Empty) { self.validation_error = Some("请上传源视频后再生成".into()); return Task::none(); }
                self.validation_error = None; self.generate_state = GenerateState::Submitting;
                self.push_log(format!("{} 任务已提交到队列", self.mode.label()));
                Task::perform(async { tokio::time::sleep(Duration::from_millis(100)).await }, |_| Message::GenerateStep)
            }
            Message::CancelGeneration => { self.generate_state = GenerateState::Idle; self.push_log("已取消生成".into()); Task::none() }
            Message::GenerateStep => { match self.generate_state { GenerateState::Submitting => { self.generate_state = GenerateState::Generating { progress: 5, step: 1, total: self.steps as u32, phase: "排队中".into() }; self.push_log("等待后端处理…".into()); Task::none() } _ => Task::none() } }
            Message::GenerateProgress { progress, step, total, phase } => { self.generate_state = GenerateState::Generating { progress, step, total, phase }; Task::none() }
            Message::GenerateComplete { result_url } => { self.generate_state = GenerateState::Done; self.push_log("生成完成".into()); self.generated_video_path = Some(result_url); Task::none() }
            Message::GenerateFailed { error } => { self.generate_state = GenerateState::Idle; self.push_log(format!("生成失败: {}", error)); Task::none() }
            Message::UploadSourceVideo => Task::perform(async { tokio::task::spawn_blocking(|| rfd::FileDialog::new().add_filter("Video", &["mp4", "mov", "webm"]).pick_file()).await.ok().flatten() }, |path| if let Some(p) = path { Message::SourceVideoSelected(p.to_string_lossy().to_string()) } else { Message::NoOp }),
            Message::UploadStartImage => Task::perform(async { tokio::task::spawn_blocking(|| rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "webp"]).pick_file()).await.ok().flatten() }, |path| if let Some(p) = path { Message::StartImageSelected(p.to_string_lossy().to_string()) } else { Message::NoOp }),
            Message::UploadTailImage => Task::perform(async { tokio::task::spawn_blocking(|| rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "webp"]).pick_file()).await.ok().flatten() }, |path| if let Some(p) = path { Message::TailImageSelected(p.to_string_lossy().to_string()) } else { Message::NoOp }),
            Message::SourceVideoSelected(path) => { self.source_video = SourceVideoState::Uploaded(path.clone()); self.push_log(format!("已选择视频: {}", path)); Task::none() }
            Message::StartImageSelected(path) => { self.start_image = SourceVideoState::Uploaded(path.clone()); self.push_log(format!("已选择起始图片: {}", path)); Task::none() }
            Message::TailImageSelected(path) => { self.tail_image = SourceVideoState::Uploaded(path.clone()); self.push_log(format!("已选择末尾图片: {}", path)); Task::none() }
            Message::ClearSourceVideo => { self.source_video = SourceVideoState::Empty; Task::none() }
            Message::ClearStartImage => { self.start_image = SourceVideoState::Empty; Task::none() }
            Message::ClearTailImage => { self.tail_image = SourceVideoState::Empty; Task::none() }
            Message::UpscaleFactorSelected(u) => { self.upscale_factor = u; Task::none() }
            Message::DenoiseChanged(v) => { self.denoise = v.clamp(0.0, 1.0); self.denoise_input = format!("{:.1}", self.denoise); Task::none() }
            Message::DenoiseInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.denoise_input = s; if let Some(v) = v { self.denoise = v.clamp(0.0, 1.0); } Task::none() }
            Message::MemoryPoll => {
                use sysinfo::{System, RefreshKind};
                let mut sys = System::new_with_specifics(RefreshKind::everything());
                sys.refresh_memory();
                self.memory_info.total_gb = sys.total_memory() as f32 / 1073741824.0;
                self.memory_info.used_gb = sys.used_memory() as f32 / 1073741824.0;
                self.memory_info.mlx_active_gb = (self.memory_info.used_gb * 0.85).clamp(0.0, self.memory_info.total_gb);
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        let tabs = container(dq_components::dq_mode_tabs(&self.mode_tabs, &self.mode, Message::ModeChanged))
            .padding([spacing::SM, spacing::MD]).width(Length::Fill).style(dq_theme::mode_tabs_container);
        let content = match self.mode {
            VideoMode::TextToVideo => text_to_video_view(self),
            VideoMode::ImageToVideo => image_to_video_view(self),
            VideoMode::Upscale => upscale_view(self),
        };
        let page = column![
            tabs,
            scrollable(column![
                section_card("标题", dq_text_input("给这次创作起个名字（可选）", &self.title, Message::TitleChanged)),
                content,
            ].spacing(spacing::MD).width(Length::Fill).padding(spacing::MD))
            .width(Length::Fill).height(Length::Fill),
        ].spacing(spacing::SM).width(Length::Fill).height(Length::Fill);
        container(page).width(Length::Fill).height(Length::Fill).into()
    }

    fn upload_area<'a>(&'a self, label: &'a str, uploaded: bool, on_upload: Message, on_clear: Message) -> Element<'a, Message> {
        let body: Element<'a, Message> = if uploaded {
            row![text("已上传").size(typography::BODY).color(color::SUCCESS), Space::new().width(Length::Fill),
                dq_control_button("清除", Some(on_clear)),
            ].spacing(spacing::SM).into()
        } else {
            iced::widget::button(container(column![phosphor_icon(PhosphorIcon::Upload, 32.0, color::TEXT_TERTIARY), text(label).size(typography::CAPTION).color(color::TEXT_TERTIARY)].spacing(spacing::SM).align_x(Alignment::Center))
                .width(Length::Fill).padding(spacing::LG).center_x(Length::Fill).style(dq_theme::inset_panel))
                .on_press(on_upload).style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() }).into()
        };
        body
    }
}

fn pref_row_inline_fn<'a>(label: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    row![container(text(label).size(typography::LABEL)).width(Length::Fixed(80.0)).align_y(Alignment::Center), control]
        .spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill).into()
}

fn result_preview_card(page: &VideoPage) -> Option<Element<'_, Message>> {
    match &page.generate_state {
        GenerateState::Done => {
            let path = page.generated_video_path.as_ref()?;
            Some(section_card("生成结果", column![
                container(phosphor_icon(PhosphorIcon::VideoCamera, 48.0, color::SUCCESS))
                    .width(Length::Fill).height(Length::Fixed(200.0))
                    .align_x(Alignment::Center).align_y(Alignment::Center)
                    .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() }),
                text(format!("视频已生成: {}", path)).size(typography::CAPTION).color(color::TEXT_TERTIARY),
            ].spacing(spacing::SM).into()))
        }
        _ => None,
    }
}

fn text_to_video_view(page: &VideoPage) -> Element<'_, Message> {
    let mut items = column![
        section_card("模型", dq_pick_list(&[], Some(&page.model), |_| Message::NoOp, "选择视频模型")),
        prompt_card(page),
        advanced_params_card(page),
        generate_card(page),
    ].spacing(spacing::MD);
    if let Some(r) = result_preview_card(page) { items = items.push(r); }
    items.into()
}

fn image_to_video_view(page: &VideoPage) -> Element<'_, Message> {
    let has_start = matches!(page.start_image, SourceVideoState::Uploaded(_));
    let has_tail = matches!(page.tail_image, SourceVideoState::Uploaded(_));
    let mut items = column![
        section_card("模型", dq_pick_list(&[], Some(&page.model), |_| Message::NoOp, "选择视频模型")),
        section_card("起始图片", page.upload_area("点击上传起始图片", has_start, Message::UploadStartImage, Message::ClearStartImage)),
        section_card("末尾图片（可选）", page.upload_area("点击上传末尾图片", has_tail, Message::UploadTailImage, Message::ClearTailImage)),
        prompt_card(page),
        advanced_params_card(page),
        generate_card(page),
    ].spacing(spacing::MD);
    if let Some(r) = result_preview_card(page) { items = items.push(r); }
    items.into()
}

fn upscale_view(page: &VideoPage) -> Element<'_, Message> {
    let has_src = matches!(page.source_video, SourceVideoState::Uploaded(_));
    let mut items = column![
        section_card("模型", dq_pick_list(&[], Some(&page.model), |_| Message::NoOp, "选择视频模型")),
        section_card("源视频", page.upload_area("点击上传源视频", has_src, Message::UploadSourceVideo, Message::ClearSourceVideo)),
        section_card("放大参数", column![
            pref_row_inline_fn("放大倍数", dq_pick_list(UpscaleFactorOption::ALL, Some(&page.upscale_factor), Message::UpscaleFactorSelected, "选择倍数")),
            pref_row_inline_fn("降噪", dq_slider_with_input(0.0..=1.0, 0.1, page.denoise, &page.denoise_input, Message::DenoiseChanged, Message::DenoiseInputChanged)),
        ].spacing(spacing::SM).into()),
        generate_card(page),
    ].spacing(spacing::MD);
    if let Some(r) = result_preview_card(page) { items = items.push(r); }
    items.into()
}

fn prompt_card(page: &VideoPage) -> Element<'_, Message> {
    let neg_toggle = iced::widget::button(chevron_icon::<Message>(!page.negative_open))
        .on_press(Message::ToggleNegativeOpen)
        .style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() });
    surface_card(
        Some(text("提示词").size(typography::TITLE).color(color::TEXT_PRIMARY).into()),
        Some(neg_toggle.into()),
        Some(column![
            dq_prompt_editor(column![].into(), "描述你想要生成的视频内容…", &page.prompt, Message::PromptEdited, format!("{}/1000", page.prompt.text().chars().count())),
            if page.negative_open { dq_text_input_multiline("描述你不需要的内容…", &page.negative_prompt, Message::NegativePromptEdited, format!("{}/500", page.negative_prompt.text().chars().count())) } else { Space::new().height(0).into() },
        ].spacing(spacing::SM).into()),
    )
}

fn advanced_params_card(page: &VideoPage) -> Element<'_, Message> {
    let trailing = iced::widget::button(chevron_icon::<Message>(!page.advanced_open))
        .on_press(Message::ToggleAdvancedOpen)
        .style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() });
    let body: Option<Element<Message>> = if page.advanced_open {
        Some(column![
            pref_row_inline_fn("步数", dq_slider_with_input(1.0..=100.0, 1.0, page.steps, &page.steps_input, Message::StepsChanged, Message::StepsInputChanged)),
            pref_row_inline_fn("引导", dq_slider_with_input(0.0..=30.0, 0.5, page.guidance, &page.guidance_input, Message::GuidanceChanged, Message::GuidanceInputChanged)),
            pref_row_inline_fn("Shift", dq_slider_with_input(0.0..=20.0, 0.5, page.shift, &page.shift_input, Message::ShiftChanged, Message::ShiftInputChanged)),
            pref_row_inline_fn("分辨率", row![
                dq_pick_list(WidthOption::ALL, Some(&page.width), Message::WidthSelected, "宽"),
                text("×").size(typography::BODY).color(color::TEXT_TERTIARY),
                dq_pick_list(HeightOption::ALL, Some(&page.height), Message::HeightSelected, "高"),
            ].spacing(spacing::SM).into()),
            pref_row_inline_fn("帧数", dq_slider_with_input(1.0..=300.0, 1.0, page.num_frames, &page.num_frames_input, Message::NumFramesChanged, Message::NumFramesInputChanged)),
            pref_row_inline_fn("FPS", dq_slider_with_input(1.0..=60.0, 1.0, page.fps, &page.fps_input, Message::FpsChanged, Message::FpsInputChanged)),
            pref_row_inline_fn("种子", row![dq_text_input("随机", &page.seed, Message::SeedChanged), dq_components::phosphor_icon_button(PhosphorIcon::ArrowsClockwise, 14.0, Some(Message::RandomizeSeed))].spacing(spacing::SM).into()),
            dq_header_button("恢复默认配置", Some(Message::RestoreDefaults)),
        ].spacing(spacing::SM).into())
    } else { None };
    surface_card(Some(text("高级参数").size(typography::TITLE).color(color::TEXT_PRIMARY).into()), Some(trailing.into()), body)
}

fn generate_card(page: &VideoPage) -> Element<'_, Message> {
    let is_busy = matches!(page.generate_state, GenerateState::Generating { .. } | GenerateState::Submitting);
    let label = match page.generate_state {
        GenerateState::Submitting => "提交中…", GenerateState::Generating { .. } => "生成中…",
        GenerateState::Done => "再次生成", GenerateState::Idle => "生成",
    };
    let on_press = if is_busy { None } else { Some(Message::Generate) };
    let mut col = column![row![
        dq_components::dq_primary_action(label, on_press),
        if is_busy { dq_components::dq_button("取消", dq_components::ButtonVariant::Ghost, dq_components::ButtonSize::Md, dq_components::ButtonWidth::Hug, Some(Message::CancelGeneration)) } else { Space::new().width(0).into() },
    ].spacing(spacing::SM).width(Length::Fill)].spacing(spacing::SM).width(Length::Fill);
    if let Some(ref err) = page.validation_error { col = col.push(text(err.as_str()).size(typography::CAPTION).color(color::DANGER)); }
    match page.generate_state {
        GenerateState::Submitting => { col = col.push(row![text("⟳").size(typography::BODY).color(color::WARNING), text("正在提交任务…").size(typography::CAPTION).color(color::TEXT_SECONDARY)].spacing(spacing::SM).align_y(Alignment::Center)); }
        GenerateState::Generating { progress, step, total, .. } => {
            col = col.push(dq_progress_bar(progress as f32 / 100.0, 4.0));
            col = col.push(row![text(format!("Step {}/{}", step, total)).size(typography::CAPTION).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(format!("{}%", progress)).size(typography::CAPTION).color(color::TEXT_TERTIARY)].align_y(Alignment::Center).width(Length::Fill));
        }
        GenerateState::Done => { col = col.push(text("✓ 已完成").size(typography::CAPTION).color(color::SUCCESS)); }
        _ => {}
    }
    col = col.push(log_panel(&page.logs, Some(Message::ClearLogs)));
    section_card("生成", col.into())
}
