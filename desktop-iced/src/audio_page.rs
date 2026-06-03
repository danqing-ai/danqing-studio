#![allow(dead_code)]
use dq_components::{
    dq_control_button, dq_header_button, dq_pick_list,
    dq_progress_bar, dq_slider_with_input, dq_text_input,
    dq_text_input_multiline, dq_prompt_editor, section_card, surface_card, tag,
    chevron_icon, phosphor_icon, TagType,
    LogLine, ModeTabOption, PhosphorIcon, default_logs, log_panel,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{column, container, row, scrollable, text, text_editor, Space, toggler};
use iced::{Alignment, Element, Length, Task};
use serde_json::json;

const PROMPT_MAX: usize = 1000;
const LYRICS_MAX: usize = 5000;

#[derive(Debug, Clone)]
pub enum Message {
    NoOp, ModeChanged(AudioMode), ModelSelected(AudioModelOption),
    TitleChanged(String), PromptEdited(text_editor::Action), NegativePromptEdited(text_editor::Action),
    LyricsEdited(text_editor::Action), ToggleNegativeOpen, ToggleAdvancedOpen,
    ToggleInstrumental(bool),
    DurationChanged(f32), DurationInputChanged(String), BpmChanged(String),
    KeySelected(KeyOption), TimeSignatureSelected(TimeSignatureOption),
    VocalTypeSelected(VocalTypeOption), VocalLanguageSelected(VocalLanguageOption),
    StepsChanged(f32), StepsInputChanged(String), GuidanceChanged(f32), GuidanceInputChanged(String),
    TemperatureChanged(f32), TemperatureInputChanged(String), TopKChanged(f32), TopKInputChanged(String),
    SeedChanged(String), RandomizeSeed, BatchCountChanged(String), FormatSelected(AudioFormatOption),
    RestoreDefaults, ClearLogs, Generate, CancelGeneration, GenerateStep,
    GenerateProgress { progress: u8, status: String }, GenerateComplete { result_url: String }, GenerateFailed(String),
    ApiAudioSubmitted(Result<serde_json::Value, String>),
    SourceFidelityChanged(f32), SourceFidelityInputChanged(String),
    UploadCoverSource, CoverSourceSelected(String), ClearCoverSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioMode { #[default] Create, Cover }
impl AudioMode {
    const ALL: &'static [AudioMode] = &[AudioMode::Create, AudioMode::Cover];
    fn label(self) -> &'static str { match self { AudioMode::Create => "创作", AudioMode::Cover => "翻唱" } }
}

#[derive(Debug, Clone)]
pub enum AudioModelOption {
    Static { id: &'static str, label: &'static str },
    Dynamic { id: String, label: String, name: String },
}
impl Default for AudioModelOption {
    fn default() -> Self { AudioModelOption::Static { id: "ace-step-xl-sft", label: "ACE-Step XL SFT" } }
}
impl PartialEq for AudioModelOption {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) { (AudioModelOption::Static { id: a, .. }, AudioModelOption::Static { id: b, .. }) => a == b, (AudioModelOption::Dynamic { id: a, .. }, AudioModelOption::Dynamic { id: b, .. }) => a == b, _ => false }
    }
}
impl Eq for AudioModelOption {}
impl std::fmt::Display for AudioModelOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.label()) }
}
impl AudioModelOption {
    pub fn id(&self) -> String { match self { AudioModelOption::Static { id, .. } => id.to_string(), AudioModelOption::Dynamic { id, .. } => id.clone() } }
    pub fn label(&self) -> String { match self { AudioModelOption::Static { label, .. } => label.to_string(), AudioModelOption::Dynamic { label, .. } => label.clone() } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyOption {
    #[default] None, C, CSharp, D, DSharp, E, F, FSharp, G, GSharp, A, ASharp, B,
    Cm, CSharpM, Dm, DSharpM, Em, Fm, FSharpM, Gm, GSharpM, Am, ASharpM, Bm,
}
impl KeyOption {
    const ALL: &'static [KeyOption] = &[KeyOption::None, KeyOption::C, KeyOption::CSharp, KeyOption::D, KeyOption::DSharp,
        KeyOption::E, KeyOption::F, KeyOption::FSharp, KeyOption::G, KeyOption::GSharp, KeyOption::A, KeyOption::ASharp, KeyOption::B,
        KeyOption::Cm, KeyOption::CSharpM, KeyOption::Dm, KeyOption::DSharpM, KeyOption::Em, KeyOption::Fm, KeyOption::FSharpM,
        KeyOption::Gm, KeyOption::GSharpM, KeyOption::Am, KeyOption::ASharpM, KeyOption::Bm];
    fn label(&self) -> &'static str { match self {
        KeyOption::None => "自动", KeyOption::C => "C", KeyOption::CSharp => "C#", KeyOption::D => "D", KeyOption::DSharp => "D#",
        KeyOption::E => "E", KeyOption::F => "F", KeyOption::FSharp => "F#", KeyOption::G => "G", KeyOption::GSharp => "G#",
        KeyOption::A => "A", KeyOption::ASharp => "A#", KeyOption::B => "B",
        KeyOption::Cm => "Cm", KeyOption::CSharpM => "C#m", KeyOption::Dm => "Dm", KeyOption::DSharpM => "D#m",
        KeyOption::Em => "Em", KeyOption::Fm => "Fm", KeyOption::FSharpM => "F#m", KeyOption::Gm => "Gm",
        KeyOption::GSharpM => "G#m", KeyOption::Am => "Am", KeyOption::ASharpM => "A#m", KeyOption::Bm => "Bm",
    }}
}
impl std::fmt::Display for KeyOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeSignatureOption { #[default] N4, N3, N2 }
impl TimeSignatureOption {
    const ALL: &'static [TimeSignatureOption] = &[TimeSignatureOption::N4, TimeSignatureOption::N3, TimeSignatureOption::N2];
    fn label(&self) -> &'static str { match self { TimeSignatureOption::N4 => "4/4", TimeSignatureOption::N3 => "3/4", TimeSignatureOption::N2 => "2/4" } }
}
impl std::fmt::Display for TimeSignatureOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VocalTypeOption { #[default] Male, Female, Chorus, Duet }
impl VocalTypeOption {
    const ALL: &'static [VocalTypeOption] = &[VocalTypeOption::Male, VocalTypeOption::Female, VocalTypeOption::Chorus, VocalTypeOption::Duet];
    fn label(&self) -> &'static str { match self { VocalTypeOption::Male => "男声", VocalTypeOption::Female => "女声", VocalTypeOption::Chorus => "合唱", VocalTypeOption::Duet => "对唱" } }
}
impl std::fmt::Display for VocalTypeOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VocalLanguageOption { #[default] Auto, En, Zh, Ja, Ko, Fr, De, Es, Pt }
impl VocalLanguageOption {
    const ALL: &'static [VocalLanguageOption] = &[VocalLanguageOption::Auto, VocalLanguageOption::En, VocalLanguageOption::Zh,
        VocalLanguageOption::Ja, VocalLanguageOption::Ko, VocalLanguageOption::Fr, VocalLanguageOption::De, VocalLanguageOption::Es, VocalLanguageOption::Pt];
    fn label(&self) -> &'static str { match self {
        VocalLanguageOption::Auto => "自动", VocalLanguageOption::En => "英语", VocalLanguageOption::Zh => "中文",
        VocalLanguageOption::Ja => "日语", VocalLanguageOption::Ko => "韩语", VocalLanguageOption::Fr => "法语",
        VocalLanguageOption::De => "德语", VocalLanguageOption::Es => "西班牙语", VocalLanguageOption::Pt => "葡萄牙语",
    }}
}
impl std::fmt::Display for VocalLanguageOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioFormatOption { #[default] Wav, Mp3, Flac }
impl AudioFormatOption {
    const ALL: &'static [AudioFormatOption] = &[AudioFormatOption::Wav, AudioFormatOption::Mp3, AudioFormatOption::Flac];
    fn label(&self) -> &'static str { match self { AudioFormatOption::Wav => "WAV", AudioFormatOption::Mp3 => "MP3", AudioFormatOption::Flac => "FLAC" } }
    fn ext(&self) -> &'static str { match self { AudioFormatOption::Wav => "wav", AudioFormatOption::Mp3 => "mp3", AudioFormatOption::Flac => "flac" } }
}
impl std::fmt::Display for AudioFormatOption { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) } }

#[derive(Debug, Clone)]
pub enum GenerateState { Idle, Submitting, Generating { progress: u8, status: String }, Done }
impl Default for GenerateState { fn default() -> Self { GenerateState::Idle } }

#[derive(Debug, Clone, Default)]
pub enum SourceAudioState { #[default] Empty, Uploaded(String) }

#[derive(Debug, Clone)]
pub struct AudioCreatePage {
    pub mode: AudioMode, pub mode_tabs: Vec<ModeTabOption<AudioMode>>,
    pub model: AudioModelOption, pub title: String,
    pub prompt: text_editor::Content, pub negative_prompt: text_editor::Content, pub lyrics: text_editor::Content,
    pub negative_open: bool, pub advanced_open: bool, pub instrumental: bool,
    pub duration: f32, pub duration_input: String, pub bpm: String,
    pub key: KeyOption, pub time_signature: TimeSignatureOption,
    pub vocal_type: VocalTypeOption, pub vocal_language: VocalLanguageOption,
    pub steps: f32, pub steps_input: String, pub guidance: f32, pub guidance_input: String,
    pub temperature: f32, pub temperature_input: String, pub top_k: f32, pub top_k_input: String,
    pub seed: String, pub batch_count: String, pub format: AudioFormatOption,
    pub source_audio: SourceAudioState, pub source_fidelity: f32, pub source_fidelity_input: String,
    pub generate_state: GenerateState, pub validation_error: Option<String>,
    pub logs: Vec<LogLine>, pub generated_audio_path: Option<String>, pub available_models: Vec<AudioModelOption>,
}

impl Default for AudioCreatePage {
    fn default() -> Self {
        Self {
            mode: AudioMode::Create,
            mode_tabs: AudioMode::ALL.iter().map(|m| ModeTabOption { label: m.label().into(), value: *m }).collect(),
            model: AudioModelOption::default(), title: String::new(),
            prompt: text_editor::Content::with_text("轻快的电子流行音乐，温暖的和弦，充满希望的旋律"),
            negative_prompt: text_editor::Content::new(), lyrics: text_editor::Content::new(),
            negative_open: false, advanced_open: false, instrumental: false,
            duration: 30.0, duration_input: "30".into(), bpm: String::new(),
            key: KeyOption::None, time_signature: TimeSignatureOption::N4,
            vocal_type: VocalTypeOption::Female, vocal_language: VocalLanguageOption::Auto,
            steps: 50.0, steps_input: "50".into(), guidance: 5.0, guidance_input: "5.0".into(),
            temperature: 0.7, temperature_input: "0.7".into(), top_k: 50.0, top_k_input: "50".into(),
            seed: "随机".into(), batch_count: "1".into(), format: AudioFormatOption::Wav,
            source_audio: SourceAudioState::Empty, source_fidelity: 0.6, source_fidelity_input: "0.6".into(),
            generate_state: GenerateState::Idle, validation_error: None,
            logs: default_logs(), generated_audio_path: None, available_models: Vec::new(),
        }
    }
}

impl AudioCreatePage {
    pub fn new() -> Self { Self::default() }
    pub fn push_log(&mut self, message: String) {
        let now = chrono::Local::now();
        self.logs.push(LogLine { time: now.format("%H:%M:%S").to_string(), message });
        if self.logs.len() > 20 { self.logs.remove(0); }
    }

    pub fn build_request(&self) -> (&'static str, serde_json::Value) {
        match self.mode {
            AudioMode::Create => ("/api/audios/generations", json!({
                "model": self.model.id(), "title": self.title, "prompt": self.prompt.text(),
                "negative_prompt": self.negative_prompt.text(), "lyrics": self.lyrics.text(),
                "duration": self.duration as i32, "instrumental": self.instrumental,
                "bpm": self.bpm.parse::<i32>().ok(),
                "key": if matches!(self.key, KeyOption::None) { None } else { Some(self.key.label()) },
                "time_signature": self.time_signature.label(),
                "vocal_type": self.vocal_type.label(), "vocal_language": self.vocal_language.label(),
                "steps": self.steps as i32, "guidance": self.guidance, "temperature": self.temperature,
                "top_k": self.top_k as i32, "seed": self.seed.parse::<i64>().ok(),
                "batch_count": self.batch_count.parse::<i32>().unwrap_or(1), "format": self.format.ext(),
                "priority": "normal", "metadata": {},
            })),
            AudioMode::Cover => ("/api/audios/edits", json!({
                "model": self.model.id(), "title": self.title, "prompt": self.prompt.text(),
                "source_fidelity": self.source_fidelity, "duration": self.duration as i32,
                "priority": "normal", "metadata": {},
            })),
        }
    }

    pub fn update(&mut self, message: Message, _api_client: &Option<dq_api::ApiClient>) -> Task<Message> {
        match message {
            Message::NoOp => Task::none(),
            Message::ModeChanged(m) => { self.mode = m; Task::none() }
            Message::ModelSelected(m) => { self.model = m; Task::none() }
            Message::TitleChanged(s) => { self.title = s; Task::none() }
            Message::PromptEdited(a) => { self.prompt.perform(a); Task::none() }
            Message::NegativePromptEdited(a) => { self.negative_prompt.perform(a); Task::none() }
            Message::LyricsEdited(a) => { self.lyrics.perform(a); Task::none() }
            Message::ToggleNegativeOpen => { self.negative_open = !self.negative_open; Task::none() }
            Message::ToggleAdvancedOpen => { self.advanced_open = !self.advanced_open; Task::none() }
            Message::ToggleInstrumental(v) => { self.instrumental = v; Task::none() }
            Message::DurationChanged(v) => { self.duration = v.clamp(1.0, 300.0); self.duration_input = format!("{:.0}", self.duration); Task::none() }
            Message::DurationInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.duration_input = s; if let Some(v) = v { self.duration = v.clamp(1.0, 300.0); } Task::none() }
            Message::BpmChanged(s) => { self.bpm = s; Task::none() }
            Message::KeySelected(k) => { self.key = k; Task::none() }
            Message::TimeSignatureSelected(t) => { self.time_signature = t; Task::none() }
            Message::VocalTypeSelected(v) => { self.vocal_type = v; Task::none() }
            Message::VocalLanguageSelected(v) => { self.vocal_language = v; Task::none() }
            Message::StepsChanged(v) => { self.steps = v.clamp(1.0, 100.0); self.steps_input = format!("{:.0}", self.steps); Task::none() }
            Message::StepsInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.steps_input = s; if let Some(v) = v { self.steps = v.clamp(1.0, 100.0); } Task::none() }
            Message::GuidanceChanged(v) => { self.guidance = v.clamp(0.0, 30.0); self.guidance_input = format!("{:.1}", self.guidance); Task::none() }
            Message::GuidanceInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.guidance_input = s; if let Some(v) = v { self.guidance = v.clamp(0.0, 30.0); } Task::none() }
            Message::TemperatureChanged(v) => { self.temperature = v.clamp(0.0, 2.0); self.temperature_input = format!("{:.2}", self.temperature); Task::none() }
            Message::TemperatureInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.temperature_input = s; if let Some(v) = v { self.temperature = v.clamp(0.0, 2.0); } Task::none() }
            Message::TopKChanged(v) => { self.top_k = v.clamp(1.0, 100.0); self.top_k_input = format!("{:.0}", self.top_k); Task::none() }
            Message::TopKInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.top_k_input = s; if let Some(v) = v { self.top_k = v.clamp(1.0, 100.0); } Task::none() }
            Message::SeedChanged(s) => { self.seed = s; Task::none() }
            Message::RandomizeSeed => { self.seed = format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()); Task::none() }
            Message::BatchCountChanged(s) => { self.batch_count = s; Task::none() }
            Message::FormatSelected(f) => { self.format = f; Task::none() }
            Message::RestoreDefaults => { self.steps = 50.0; self.steps_input = "50".into(); self.guidance = 5.0; self.guidance_input = "5.0".into(); self.temperature = 0.7; self.temperature_input = "0.7".into(); self.top_k = 50.0; self.top_k_input = "50".into(); self.seed = "随机".into(); self.batch_count = "1".into(); self.format = AudioFormatOption::Wav; Task::none() }
            Message::ClearLogs => { self.logs.clear(); Task::none() }
            Message::Generate => {
                if self.prompt.text().trim().is_empty() { self.validation_error = Some("请输入提示词后再生成".into()); return Task::none(); }
                self.validation_error = None; self.generate_state = GenerateState::Submitting;
                self.push_log("音频生成任务已提交到队列".into());
                Task::perform(async { tokio::time::sleep(std::time::Duration::from_millis(100)).await }, |_| Message::GenerateStep)
            }
            Message::CancelGeneration => {
                self.generate_state = GenerateState::Idle; self.push_log("已取消生成".into()); Task::none()
            }
            Message::GenerateStep => {
                match self.generate_state { GenerateState::Submitting => { self.generate_state = GenerateState::Generating { progress: 5, status: "排队中".into() }; self.push_log("等待后端处理…".into()); Task::none() } _ => Task::none() }
            }
            Message::GenerateProgress { progress, status } => { self.generate_state = GenerateState::Generating { progress, status }; Task::none() }
            Message::GenerateComplete { result_url } => { self.generate_state = GenerateState::Done; self.push_log("生成完成".into()); self.generated_audio_path = Some(result_url); Task::none() }
            Message::GenerateFailed(error) => { self.generate_state = GenerateState::Idle; self.push_log(format!("生成失败: {}", error)); Task::none() }
            Message::ApiAudioSubmitted(result) => {
                match result {
                    Ok(response) => {
                        let task_id = response.get("task").and_then(|t| t.get("id")).and_then(|v| v.as_str()).map(|s| s.to_string())
                            .or_else(|| response.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()));
                        if let Some(tid) = task_id { self.push_log(format!("任务已提交: {}", tid)); }
                    }
                    Err(e) => { self.push_log(format!("提交失败: {}", e)); self.generate_state = GenerateState::Idle; }
                }
                Task::none()
            }
            Message::SourceFidelityChanged(v) => { self.source_fidelity = v.clamp(0.0, 1.0); self.source_fidelity_input = format!("{:.2}", self.source_fidelity); Task::none() }
            Message::SourceFidelityInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.source_fidelity_input = s; if let Some(v) = v { self.source_fidelity = v.clamp(0.0, 1.0); } Task::none() }
            Message::UploadCoverSource => Task::perform(async { tokio::task::spawn_blocking(|| rfd::FileDialog::new().add_filter("Audio", &["mp3", "wav", "flac", "ogg", "m4a"]).pick_file()).await.ok().flatten() }, |path| if let Some(p) = path { Message::CoverSourceSelected(p.to_string_lossy().to_string()) } else { Message::NoOp }),
            Message::CoverSourceSelected(path) => { self.source_audio = SourceAudioState::Uploaded(path.clone()); self.push_log(format!("已选择音频文件: {}", path)); Task::none() }
            Message::ClearCoverSource => { self.source_audio = SourceAudioState::Empty; Task::none() }
        }
    }

    pub fn view(&self, _api_client: &Option<dq_api::ApiClient>) -> Element<'_, Message> {
        let tabs = container(dq_components::dq_mode_tabs(&self.mode_tabs, &self.mode, Message::ModeChanged))
            .padding([spacing::SM, spacing::MD]).width(Length::Fill).style(dq_theme::mode_tabs_container);

        let content: Element<Message> = if self.mode == AudioMode::Create { self.create_view() } else { self.cover_view() };
        let page = column![tabs, scrollable(content).width(Length::Fill).height(Length::Fill)]
            .spacing(spacing::SM).width(Length::Fill).height(Length::Fill);
        container(page).width(Length::Fill).height(Length::Fill).into()
    }

    fn model_section(&self) -> Element<'_, Message> {
        if !self.available_models.is_empty() {
            let selected = if self.available_models.iter().any(|m| m == &self.model) { Some(&self.model) } else { self.available_models.first() };
            section_card("模型", dq_pick_list(&self.available_models[..], selected, Message::ModelSelected, "选择音频模型"))
        } else {
            text("音频模型暂不可用").size(typography::BODY).color(color::TEXT_SECONDARY).into()
        }
    }

    fn create_view(&self) -> Element<'_, Message> {
        let mut items = column![
            self.model_section(),
            section_card("标题", dq_text_input("给这次创作起个名字（可选）", &self.title, Message::TitleChanged)),
            prompt_card(self),
            lyrics_card(self),
            music_params_card(self),
            advanced_card(self),
            instrumental_toggle(self),
            generate_and_log_section(self),
        ].spacing(spacing::MD).width(Length::Fill).padding(spacing::MD);
        if matches!(self.generate_state, GenerateState::Done) {
            if let Some(ref path) = self.generated_audio_path {
                items = items.push(result_audio_card(path));
            }
        }
        items.into()
    }

    fn cover_view(&self) -> Element<'_, Message> {
        let source_body: Element<Message> = match &self.source_audio {
            SourceAudioState::Uploaded(path) => column![
                tag(path, TagType::Info),
                row![dq_control_button("更换文件", Some(Message::UploadCoverSource)), dq_control_button("清除", Some(Message::ClearCoverSource))].spacing(spacing::SM),
            ].spacing(spacing::XS).into(),
            SourceAudioState::Empty => {
                let upload_btn = container(column![phosphor_icon(PhosphorIcon::Upload, 32.0, color::TEXT_TERTIARY), text("点击上传源音频文件").size(typography::CAPTION).color(color::TEXT_TERTIARY)].spacing(spacing::SM).align_x(Alignment::Center))
                    .width(Length::Fill).padding(spacing::LG).center_x(Length::Fill).style(dq_theme::inset_panel);
                iced::widget::button(upload_btn).on_press(Message::UploadCoverSource).style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() }).into()
            }
        };
        column![
            self.model_section(),
            section_card("源音频", source_body),
            section_card("提示词", dq_text_input_multiline("描述翻唱风格（可选）…", &self.prompt, Message::PromptEdited, format!("{}/{}", self.prompt.text().chars().count(), PROMPT_MAX))),
            section_card("参数", column![
                dq_slider_with_input(0.0..=1.0, 0.05, self.source_fidelity, &self.source_fidelity_input, Message::SourceFidelityChanged, Message::SourceFidelityInputChanged),
                dq_slider_with_input(1.0..=300.0, 1.0, self.duration, &self.duration_input, Message::DurationChanged, Message::DurationInputChanged),
            ].spacing(spacing::SM).into()),
            generate_and_log_section(self),
        ].spacing(spacing::MD).width(Length::Fill).padding(spacing::MD).into()
    }
}

fn prompt_card(page: &AudioCreatePage) -> Element<'_, Message> {
    let _neg_toggle = iced::widget::button(chevron_icon::<Message>(!page.negative_open))
        .on_press(Message::ToggleNegativeOpen)
        .style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() });
    section_card("提示词", column![
        dq_prompt_editor(column![].into(), "描述你想要的音乐风格和情绪…", &page.prompt, Message::PromptEdited, format!("{}/{}", page.prompt.text().chars().count(), PROMPT_MAX)),
        if page.negative_open { dq_text_input_multiline("描述你不需要的内容…", &page.negative_prompt, Message::NegativePromptEdited, format!("{}/500", page.negative_prompt.text().chars().count())) } else { Space::new().height(0).into() },
    ].spacing(spacing::SM).into())
}

fn lyrics_card(page: &AudioCreatePage) -> Element<'_, Message> {
    if page.instrumental { return Space::new().height(0).into(); }
    section_card("歌词", column![
        dq_text_input_multiline("输入歌词（可选）…", &page.lyrics, Message::LyricsEdited, format!("{}/{}", page.lyrics.text().chars().count(), LYRICS_MAX)),
        pref_row_inline("人声类型", dq_pick_list(VocalTypeOption::ALL, Some(&page.vocal_type), Message::VocalTypeSelected, "选择人声类型")),
        pref_row_inline("人声语言", dq_pick_list(VocalLanguageOption::ALL, Some(&page.vocal_language), Message::VocalLanguageSelected, "选择语言")),
    ].spacing(spacing::SM).into())
}

fn music_params_card(page: &AudioCreatePage) -> Element<'_, Message> {
    let pills = row![].spacing(spacing::XS);
    let mut duration_row = pills;
    for &d in &[30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0] {
        let active = (page.duration - d).abs() < 0.5;
        let label = format!("{}s", d);
        let pill = container(text(label).size(typography::LABEL).color(if active { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY }))
            .padding([4.0, 8.0]).style(move |_theme: &iced::Theme| container::Style {
                background: if active { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None },
                border: iced::Border { color: if active { color::BORDER_SUBTLE } else { iced::Color::TRANSPARENT }, width: 1.0, radius: spacing::RADIUS_SM.into() },
                ..Default::default()
            });
        duration_row = duration_row.push(pill);
    }
    section_card("音乐参数", column![
        row![text("时长").size(typography::LABEL).color(color::TEXT_SECONDARY).width(Length::Fixed(80.0)), column![duration_row, dq_slider_with_input(1.0..=300.0, 1.0, page.duration, &page.duration_input, Message::DurationChanged, Message::DurationInputChanged)].spacing(spacing::XS).width(Length::Fill)].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
        pref_row_inline("BPM", dq_text_input("30-300", &page.bpm, Message::BpmChanged)),
        pref_row_inline("调性", dq_pick_list(KeyOption::ALL, Some(&page.key), Message::KeySelected, "选择调性")),
        pref_row_inline("拍号", dq_pick_list(TimeSignatureOption::ALL, Some(&page.time_signature), Message::TimeSignatureSelected, "选择拍号")),
    ].spacing(spacing::SM).into())
}

fn advanced_card(page: &AudioCreatePage) -> Element<'_, Message> {
    let trailing = iced::widget::button(chevron_icon::<Message>(!page.advanced_open))
        .on_press(Message::ToggleAdvancedOpen)
        .style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() });
    let body: Option<Element<Message>> = if page.advanced_open {
        Some(column![
            pref_row_inline("步数", dq_slider_with_input(1.0..=100.0, 1.0, page.steps, &page.steps_input, Message::StepsChanged, Message::StepsInputChanged)),
            pref_row_inline("引导", dq_slider_with_input(0.0..=30.0, 0.5, page.guidance, &page.guidance_input, Message::GuidanceChanged, Message::GuidanceInputChanged)),
            pref_row_inline("温度", dq_slider_with_input(0.0..=2.0, 0.05, page.temperature, &page.temperature_input, Message::TemperatureChanged, Message::TemperatureInputChanged)),
            pref_row_inline("Top-K", dq_slider_with_input(1.0..=100.0, 1.0, page.top_k, &page.top_k_input, Message::TopKChanged, Message::TopKInputChanged)),
            pref_row_inline("种子", row![dq_text_input("随机", &page.seed, Message::SeedChanged), dq_components::phosphor_icon_button(PhosphorIcon::ArrowsClockwise, 14.0, Some(Message::RandomizeSeed))].spacing(spacing::SM).into()),
            pref_row_inline("批次数", dq_text_input("1-8", &page.batch_count, Message::BatchCountChanged)),
            pref_row_inline("格式", dq_pick_list(AudioFormatOption::ALL, Some(&page.format), Message::FormatSelected, "选择格式")),
            dq_header_button("恢复默认配置", Some(Message::RestoreDefaults)),
        ].spacing(spacing::SM).into())
    } else { None };
    surface_card(Some(text("高级参数").size(typography::TITLE).color(color::TEXT_PRIMARY).into()), Some(trailing.into()), body)
}

fn instrumental_toggle(page: &AudioCreatePage) -> Element<'_, Message> {
    row![toggler(page.instrumental).label("纯音乐（无人声）").text_size(typography::LABEL).spacing(6.0).on_toggle(Message::ToggleInstrumental), Space::new().width(Length::Fill)].spacing(spacing::SM).width(Length::Fill).into()
}

fn result_audio_card(path: &str) -> Element<'_, Message> {
    section_card("生成结果", column![
        container(phosphor_icon(PhosphorIcon::SpeakerHigh, 48.0, color::SUCCESS))
            .width(Length::Fill).height(Length::Fixed(120.0))
            .align_x(Alignment::Center).align_y(Alignment::Center)
            .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() }),
        text(format!("音频已生成: {}", path)).size(typography::CAPTION).color(color::TEXT_TERTIARY),
    ].spacing(spacing::SM).into())
}

fn pref_row_inline<'a>(label: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    row![container(text(label).size(typography::LABEL)).width(Length::Fixed(80.0)).align_y(Alignment::Center), control].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill).into()
}

fn generate_and_log_section(page: &AudioCreatePage) -> Element<'_, Message> {
    let is_busy = matches!(page.generate_state, GenerateState::Generating { .. } | GenerateState::Submitting);
    let label = match page.generate_state {
        GenerateState::Submitting => "提交中…", GenerateState::Generating { .. } => "生成中…",
        GenerateState::Done => "再次生成", GenerateState::Idle => if page.mode == AudioMode::Cover { "生成翻唱" } else { "生成音乐" },
    };
    let on_press = if is_busy { None } else { Some(Message::Generate) };
    let mut col = column![row![
        dq_components::dq_primary_action(label, on_press),
        if is_busy { dq_components::dq_button("取消", dq_components::ButtonVariant::Ghost, dq_components::ButtonSize::Md, dq_components::ButtonWidth::Hug, Some(Message::CancelGeneration)) } else { Space::new().width(0).into() },
    ].spacing(spacing::SM).width(Length::Fill)].spacing(spacing::SM).width(Length::Fill);
    if let Some(ref err) = page.validation_error { col = col.push(text(err.as_str()).size(typography::CAPTION).color(color::DANGER)); }
    match page.generate_state {
        GenerateState::Submitting => { col = col.push(row![text("⟳").size(typography::BODY).color(color::WARNING), text("正在提交任务…").size(typography::CAPTION).color(color::TEXT_SECONDARY)].spacing(spacing::SM).align_y(Alignment::Center)); }
        GenerateState::Generating { progress, .. } => { col = col.push(dq_progress_bar(progress as f32 / 100.0, 4.0)); col = col.push(text(format!("{}%", progress)).size(typography::CAPTION).color(color::TEXT_TERTIARY)); }
        GenerateState::Done => { col = col.push(text("✓ 已完成").size(typography::CAPTION).color(color::SUCCESS)); }
        _ => {}
    }
    col = col.push(log_panel(&page.logs, Some(Message::ClearLogs)));
    col.into()
}
