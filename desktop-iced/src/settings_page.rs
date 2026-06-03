#![allow(dead_code)]
use dq_components::{
    dq_header_button, dq_button, dq_slider_with_input, dq_text_input,
    section_card, tag, pref_row_inline,
    TagType, ButtonSize, ButtonVariant, ButtonWidth, PhosphorIcon,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{button, column, container, row, scrollable, text, Space, toggler};
use iced::{Alignment, Element, Length, Task};

#[derive(Debug, Clone)]
pub enum Message {
    NoOp, LoadSystemInfo, SystemInfoLoaded(Result<serde_json::Value, String>),
    LoadMetrics, MetricsLoaded(Result<serde_json::Value, String>),
    SetTab(String),
    LanguageSelected(String), OutputFormatSelected(String),
    ThemeSelected(crate::app::ThemeId),
    MemoryLimitChanged(f32), MemoryLimitInputChanged(String),
    CacheTtlChanged(f32), CacheTtlInputChanged(String),
    QueueImageFirstToggled(bool), AutoSaveToggled(bool),
    WorkspacePathChanged(String),
    SaveSettings, SettingsSaved(Result<serde_json::Value, String>),
    LoadSettings, SettingsLoaded(Result<serde_json::Value, String>),
    // Preset templates
    AddPreset, EditPreset(usize), DeletePreset(usize), CancelPresetForm,
    PresetFormNameChanged(String), PresetFormPositiveChanged(String),
    PresetFormNegativeChanged(String), PresetFormMediaChanged(String),
    SavePresetForm, RestorePresets,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LangOption { Zh, En }
impl LangOption { const ALL: &'static [LangOption] = &[LangOption::Zh, LangOption::En]; }
impl std::fmt::Display for LangOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { LangOption::Zh => "中文", LangOption::En => "English" })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum FmtOption { Png, Jpeg, Webp }
impl FmtOption { const ALL: &'static [FmtOption] = &[FmtOption::Png, FmtOption::Jpeg, FmtOption::Webp]; }
impl std::fmt::Display for FmtOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { FmtOption::Png => "PNG", FmtOption::Jpeg => "JPEG", FmtOption::Webp => "WebP" })
    }
}

#[derive(Debug, Clone)]
pub struct PresetTemplate {
    pub name: String, pub positive: String, pub negative: String,
    pub media_scope: String, pub applies_to: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SettingsPage {
    pub health: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
    pub settings: Option<serde_json::Value>,
    pub loading: bool, pub saving: bool, pub logs: Vec<String>,
    pub active_tab: String,
    pub presets: Vec<PresetTemplate>,
    pub show_preset_form: bool,
    pub editing_preset: Option<usize>,
    pub preset_form_name: String,
    pub preset_form_positive: String,
    pub preset_form_negative: String,
    pub preset_form_media: String,
    // Form state
    pub language: String, pub output_format: String,
    pub mlx_memory_limit: f32, pub mlx_memory_limit_input: String,
    pub cache_ttl: f32, pub cache_ttl_input: String,
    pub queue_image_first: bool, pub auto_save_prompts: bool,
    pub workspace_dir: String,
    pub hf_token: String, pub civitai_token: String, pub nsfw_enabled: bool,
}

impl SettingsPage {
    pub fn new() -> Self { Self::default() }

    fn apply_settings(&mut self, data: &serde_json::Value) {
        if let Some(lang) = data.get("language").and_then(|v| v.as_str()) { self.language = lang.to_string(); }
        if let Some(fmt) = data.get("output_format").and_then(|v| v.as_str()) { self.output_format = fmt.to_string(); }
        if let Some(mlx) = data.get("mlx_memory_limit").and_then(|v| v.as_f64()) { self.mlx_memory_limit = mlx as f32; self.mlx_memory_limit_input = format!("{:.0}", mlx); }
        if let Some(ttl) = data.get("model_cache_ttl_minutes").and_then(|v| v.as_f64()) { self.cache_ttl = ttl as f32; self.cache_ttl_input = format!("{:.0}", ttl); }
        if let Some(q) = data.get("queue_image_first").and_then(|v| v.as_bool()) { self.queue_image_first = q; }
        if let Some(a) = data.get("auto_save_prompts").and_then(|v| v.as_bool()) { self.auto_save_prompts = a; }
        if let Some(w) = data.get("custom_workspace_dir").and_then(|v| v.as_str()) { self.workspace_dir = w.to_string(); }
        if let Some(hf) = data.get("huggingface_token").and_then(|v| v.as_str()) { self.hf_token = hf.to_string(); }
        if let Some(cc) = data.get("civitai_token").and_then(|v| v.as_str()) { self.civitai_token = cc.to_string(); }
        if let Some(n) = data.get("nsfw_enabled").and_then(|v| v.as_bool()) { self.nsfw_enabled = n; }
    }

    pub fn update(&mut self, message: Message, api_client: &Option<dq_api::ApiClient>) -> Task<Message> {
        match message {
            Message::NoOp => Task::none(),
            Message::LoadSystemInfo => {
                self.loading = true;
                if let Some(client) = api_client.clone() {
                    return Task::perform(async move {
                        match client.get::<serde_json::Value>("/api/system/health").await {
                            Ok(v) => Message::SystemInfoLoaded(Ok(v)), Err(e) => Message::SystemInfoLoaded(Err(e.to_string())),
                        }
                    }, |msg| msg);
                }
                self.push_log("API 客户端未初始化".into()); self.loading = false; Task::none()
            }
            Message::SystemInfoLoaded(result) => {
                self.loading = false;
                match result { Ok(data) => { self.health = Some(data); self.push_log("系统信息已更新".into()); } Err(e) => { self.push_log(format!("加载系统信息失败: {}", e)); } }
                Task::none()
            }
            Message::LoadSettings => {
                if let Some(client) = api_client.clone() {
                    return Task::perform(async move {
                        match client.get::<serde_json::Value>("/api/settings/system").await {
                            Ok(v) => Message::SettingsLoaded(Ok(v)), Err(e) => Message::SettingsLoaded(Err(e.to_string())),
                        }
                    }, |msg| msg);
                }
                Task::none()
            }
            Message::SettingsLoaded(result) => {
                match result {
                    Ok(data) => { self.settings = Some(data.clone()); self.apply_settings(&data); self.push_log("设置已加载".into()); }
                    Err(e) => { self.push_log(format!("加载设置失败: {}", e)); }
                }
                Task::none()
            }
            Message::SaveSettings => {
                self.saving = false;
                self.push_log("保存设置（待实现 API 集成）".into());
                Task::none()
            }
            Message::SettingsSaved(result) => {
                self.saving = false;
                match result { Ok(_) => self.push_log("设置已保存".into()), Err(e) => self.push_log(format!("保存失败: {}", e)), }
                Task::none()
            }
            Message::LoadMetrics => {
                if let Some(client) = api_client.clone() {
                    return Task::perform(async move {
                        match client.get::<serde_json::Value>("/api/system/metrics").await {
                            Ok(v) => Message::MetricsLoaded(Ok(v)), Err(e) => Message::MetricsLoaded(Err(e.to_string())),
                        }
                    }, |msg| msg);
                }
                Task::none()
            }
            Message::MetricsLoaded(result) => { match result { Ok(data) => { self.metrics = Some(data); } Err(e) => { self.push_log(format!("加载指标失败: {}", e)); } } Task::none() }
            Message::SetTab(t) => { self.active_tab = t; Task::none() }
            Message::LanguageSelected(s) => { self.language = s; Task::none() }
            Message::OutputFormatSelected(s) => { self.output_format = s; Task::none() }
            Message::ThemeSelected(_theme) => {
                // Theme change is handled by parent App; this page just emits the message
                Task::none()
            }
            Message::MemoryLimitChanged(v) => { self.mlx_memory_limit = v.clamp(32.0, 256.0); self.mlx_memory_limit_input = format!("{:.0}", self.mlx_memory_limit); Task::none() }
            Message::MemoryLimitInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.mlx_memory_limit_input = s; if let Some(v) = v { self.mlx_memory_limit = v.clamp(32.0, 256.0); } Task::none() }
            Message::CacheTtlChanged(v) => { self.cache_ttl = v.clamp(5.0, 120.0); self.cache_ttl_input = format!("{:.0}", self.cache_ttl); Task::none() }
            Message::CacheTtlInputChanged(s) => { let v: Option<f32> = s.parse().ok(); self.cache_ttl_input = s; if let Some(v) = v { self.cache_ttl = v.clamp(5.0, 120.0); } Task::none() }
            Message::QueueImageFirstToggled(v) => { self.queue_image_first = v; Task::none() }
            Message::AutoSaveToggled(v) => { self.auto_save_prompts = v; Task::none() }
            Message::WorkspacePathChanged(s) => { self.workspace_dir = s; Task::none() }
            // Preset templates
            Message::AddPreset => { self.show_preset_form = true; self.editing_preset = None; self.preset_form_name.clear(); self.preset_form_positive.clear(); self.preset_form_negative.clear(); self.preset_form_media = "image".into(); Task::none() }
            Message::EditPreset(idx) => {
                if let Some(p) = self.presets.get(idx) { self.preset_form_name = p.name.clone(); self.preset_form_positive = p.positive.clone(); self.preset_form_negative = p.negative.clone(); self.preset_form_media = p.media_scope.clone(); self.editing_preset = Some(idx); self.show_preset_form = true; }
                Task::none()
            }
            Message::DeletePreset(idx) => { if idx < self.presets.len() { self.presets.remove(idx); } Task::none() }
            Message::CancelPresetForm => { self.show_preset_form = false; self.editing_preset = None; Task::none() }
            Message::PresetFormNameChanged(s) => { self.preset_form_name = s; Task::none() }
            Message::PresetFormPositiveChanged(s) => { self.preset_form_positive = s; Task::none() }
            Message::PresetFormNegativeChanged(s) => { self.preset_form_negative = s; Task::none() }
            Message::PresetFormMediaChanged(s) => { self.preset_form_media = s; Task::none() }
            Message::SavePresetForm => {
                let tmpl = PresetTemplate { name: self.preset_form_name.clone(), positive: self.preset_form_positive.clone(), negative: self.preset_form_negative.clone(), media_scope: self.preset_form_media.clone(), applies_to: vec!["create".into()] };
                match self.editing_preset { Some(idx) => { if idx < self.presets.len() { self.presets[idx] = tmpl; } }, None => { self.presets.push(tmpl); } }
                self.show_preset_form = false; self.editing_preset = None;
                self.push_log(format!("预设已保存: {}", self.preset_form_name));
                Task::none()
            }
            Message::RestorePresets => { self.presets.clear(); self.push_log("预设已恢复为默认".into()); Task::none() }
        }
    }

    fn push_log(&mut self, message: String) {
        let now = chrono::Local::now();
        self.logs.push(format!("[{}] {}", now.format("%H:%M:%S"), message));
        if self.logs.len() > 20 { self.logs.remove(0); }
    }

    fn tab_btn<'a>(label: &'a str, active: bool, _on_press: Message) -> Element<'a, Message> {
        container(text(label).size(typography::LABEL).color(if active { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY }))
            .padding([6.0, 14.0])
            .style(move |_theme: &iced::Theme| container::Style {
                background: if active { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None },
                border: iced::Border { color: if active { color::BORDER_SUBTLE } else { iced::Color::TRANSPARENT }, width: 1.0, radius: spacing::RADIUS_SM.into() },
                ..Default::default()
            }).into()
    }

    pub fn view(&self, _api_client: &Option<dq_api::ApiClient>, current_theme: crate::app::ThemeId) -> Element<'_, Message> {
        let header = row![
            text("设置").size(typography::HEADING).color(color::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            dq_header_button("刷新", Some(Message::LoadSystemInfo)),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill);

        let tabs = row![
            button(Self::tab_btn("系统状态", self.active_tab == "status", Message::SetTab("status".into()))).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }).on_press(Message::SetTab("status".into())),
            button(Self::tab_btn("提示词模板", self.active_tab == "presets", Message::SetTab("presets".into()))).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }).on_press(Message::SetTab("presets".into())),
            button(Self::tab_btn("系统配置", self.active_tab == "config", Message::SetTab("config".into()))).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }).on_press(Message::SetTab("config".into())),
            button(Self::tab_btn("系统指标", self.active_tab == "metrics", Message::SetTab("metrics".into()))).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }).on_press(Message::SetTab("metrics".into())),
        ].spacing(spacing::XS);

        let content: Element<Message> = match self.active_tab.as_str() {
            "config" => self.config_view(current_theme),
            "presets" => self.presets_view(),
            "metrics" => self.metrics_view(),
            _ => self.status_view(),
        };

        scrollable(column![header, tabs, content].spacing(spacing::MD).width(Length::Fill).padding(spacing::MD))
            .width(Length::Fill).height(Length::Fill).into()
    }

    fn status_view(&self) -> Element<'_, Message> {
        if let Some(ref health) = self.health {
            let status = health.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            let backends = health.get("backends").and_then(|v| v.as_object());
            let engines = health.get("engines").and_then(|v| v.as_object());

            let mut backends_list = column![].spacing(spacing::XS).width(Length::Fill);
            if let Some(b) = backends {
                for (name, st) in b {
                    let st_str = st.as_str().unwrap_or("unknown");
                    backends_list = backends_list.push(row![text(name).size(typography::BODY).color(color::TEXT_PRIMARY), Space::new().width(Length::Fill), tag(st_str, if st_str == "ok" { TagType::Success } else { TagType::Warning })].spacing(spacing::SM));
                }
            }

            let mut engines_list = column![].spacing(spacing::XS).width(Length::Fill);
            if let Some(e) = engines {
                for (name, st) in e {
                    let st_str = st.as_str().unwrap_or("unknown");
                    engines_list = engines_list.push(row![text(name).size(typography::BODY).color(color::TEXT_PRIMARY), Space::new().width(Length::Fill), tag(st_str, TagType::Info)].spacing(spacing::SM));
                }
            }

            column![
                section_card("系统状态", row![tag(status, if status == "ok" { TagType::Success } else { TagType::Warning }), Space::new().width(Length::Fill)].spacing(spacing::SM).into()),
                section_card("后端服务", backends_list.into()),
                section_card("引擎", engines_list.into()),
            ].spacing(spacing::MD).width(Length::Fill).into()
        } else {
            dq_components::empty_state(Some(PhosphorIcon::Gear), "系统状态", "点击「刷新」加载系统信息", Some(dq_button("加载", ButtonVariant::Primary, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::LoadSystemInfo)))).into()
        }
    }

    fn metrics_view(&self) -> Element<'_, Message> {
        if let Some(ref metrics) = self.metrics {
            let cpu = metrics.get("cpu_percent").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let memory = metrics.get("memory").and_then(|v| v.as_object());

            let mut metric_rows = column![].spacing(spacing::MD).width(Length::Fill);
            metric_rows = metric_rows.push(row![text("CPU 使用率").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(format!("{:.1}%", cpu)).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).width(Length::Fill));

            if let Some(m) = memory {
                let total = m.get("total_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let used = m.get("used_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                metric_rows = metric_rows.push(row![text("内存").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(format!("{:.1} / {:.1} GB", used, total)).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).width(Length::Fill));
            }

            column![
                section_card("系统指标", metric_rows.into()),
                dq_header_button("刷新指标", Some(Message::LoadMetrics)),
            ].spacing(spacing::MD).width(Length::Fill).into()
        } else {
            dq_components::empty_state(Some(PhosphorIcon::Sliders), "系统指标", "点击「刷新」加载指标", Some(dq_button("加载", ButtonVariant::Primary, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::LoadMetrics)))).into()
        }
    }

    fn presets_view(&self) -> Element<'_, Message> {
        let header = row![
            text("提示词模板").size(typography::TITLE).color(color::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            dq_header_button("恢复默认", Some(Message::RestorePresets)),
            dq_button("添加模板", ButtonVariant::Primary, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::AddPreset)),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill);

        let mut list = column![].spacing(spacing::SM).width(Length::Fill);
        if self.presets.is_empty() {
            list = list.push(dq_components::empty_state(Some(PhosphorIcon::FileText), "暂无模板", "点击「添加模板」创建新的提示词预设", None));
        } else {
            for (idx, p) in self.presets.iter().enumerate() {
                let preview = p.positive.chars().take(80).collect::<String>();
                let card = section_card(&p.name, column![
                    row![tag(p.media_scope.as_str(), TagType::Info), Space::new().width(Length::Fill), dq_header_button("编辑", Some(Message::EditPreset(idx))), dq_components::dq_button("删除", dq_components::ButtonVariant::Ghost, dq_components::ButtonSize::Sm, dq_components::ButtonWidth::Hug, Some(Message::DeletePreset(idx)))].spacing(spacing::SM).width(Length::Fill),
                    text(preview).size(typography::CAPTION).color(color::TEXT_TERTIARY),
                ].spacing(spacing::XS).into());
                list = list.push(card);
            }
        }

        let form: Option<Element<Message>> = if self.show_preset_form {
            Some(section_card(if self.editing_preset.is_some() { "编辑模板" } else { "添加模板" }, column![
                dq_components::pref_row_inline("名称", dq_text_input("模板名称", &self.preset_form_name, Message::PresetFormNameChanged)),
                dq_components::pref_row_inline("媒体", dq_text_input("image/video", &self.preset_form_media, Message::PresetFormMediaChanged)),
                text("正面提示词").size(typography::LABEL).color(color::TEXT_SECONDARY),
                dq_text_input("正面提示词…", &self.preset_form_positive, Message::PresetFormPositiveChanged),
                text("负面提示词").size(typography::LABEL).color(color::TEXT_SECONDARY),
                dq_text_input("负面提示词…", &self.preset_form_negative, Message::PresetFormNegativeChanged),
                row![Space::new().width(Length::Fill), dq_components::dq_button("取消", dq_components::ButtonVariant::Ghost, dq_components::ButtonSize::Sm, dq_components::ButtonWidth::Hug, Some(Message::CancelPresetForm)), dq_components::dq_button("保存", dq_components::ButtonVariant::Primary, dq_components::ButtonSize::Sm, dq_components::ButtonWidth::Hug, Some(Message::SavePresetForm))].spacing(spacing::SM),
            ].spacing(spacing::MD).into()))
        } else { None };

        column![header, list, if let Some(f) = form { f } else { Space::new().height(0).into() }]
            .spacing(spacing::MD).width(Length::Fill).max_width(600.0).into()
    }

    fn config_view(&self, current_theme: crate::app::ThemeId) -> Element<'_, Message> {
        use crate::app::ThemeId;

        let theme_options = ThemeId::ALL.iter().map(|t| {
            let label = t.label();
            let is_active = *t == current_theme;
            row![
                text(label).size(typography::BODY).color(color::TEXT_PRIMARY),
                Space::new().width(Length::Fill),
                if is_active {
                    dq_components::tag("当前", dq_components::TagType::Success)
                } else {
                    dq_components::dq_button("切换", dq_components::ButtonVariant::Secondary, dq_components::ButtonSize::Sm, dq_components::ButtonWidth::Hug, Some(Message::ThemeSelected(*t)))
                },
            ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill).into()
        }).collect::<Vec<_>>();

        column![
            section_card("通用设置", column![
                row![text("语言").size(typography::LABEL).color(color::TEXT_SECONDARY).width(Length::Fixed(100.0)), text(match self.language.as_str() { "en" => "English", _ => "中文" }).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
                row![text("输出格式").size(typography::LABEL).color(color::TEXT_SECONDARY).width(Length::Fixed(100.0)), text(match self.output_format.as_str() { "jpeg" => "JPEG", "webp" => "WebP", _ => "PNG" }).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
            ].spacing(spacing::MD).into()),
            section_card("界面主题", column(theme_options).spacing(spacing::MD).into()),
            section_card("性能", column![
                pref_row_inline("内存限制", dq_slider_with_input(32.0..=256.0, 8.0, self.mlx_memory_limit, &self.mlx_memory_limit_input, Message::MemoryLimitChanged, Message::MemoryLimitInputChanged)),
                pref_row_inline("缓存 TTL", dq_slider_with_input(5.0..=120.0, 5.0, self.cache_ttl, &self.cache_ttl_input, Message::CacheTtlChanged, Message::CacheTtlInputChanged)),
                row![text("图片任务优先").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), toggler(self.queue_image_first).on_toggle(Message::QueueImageFirstToggled)].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
                row![text("自动保存提示词").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), toggler(self.auto_save_prompts).on_toggle(Message::AutoSaveToggled)].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
            ].spacing(spacing::MD).into()),
            section_card("工作空间", column![
                row![text("目录").size(typography::LABEL).color(color::TEXT_SECONDARY).width(Length::Fixed(100.0)), dq_text_input("自定义工作空间路径", &self.workspace_dir, Message::WorkspacePathChanged)].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
            ].spacing(spacing::MD).into()),
            dq_button("保存设置", ButtonVariant::Primary, ButtonSize::Md, ButtonWidth::Fill, if self.saving { None } else { Some(Message::SaveSettings) }),
        ].spacing(spacing::MD).width(Length::Fill).max_width(600.0).into()
    }
}
