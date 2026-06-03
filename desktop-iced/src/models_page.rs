#![allow(dead_code)]
use dq_components::{
    dq_button, dq_header_button,
    section_card, tag, alert, TagType, AlertType,
    ButtonSize, ButtonVariant, ButtonWidth,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{button, column, container, row, scrollable, text, Space};
use iced::{Alignment, Element, Length, Task};

#[derive(Debug, Clone)]
pub enum Message {
    NoOp, LoadModels,
    ModelsLoaded(Result<serde_json::Value, String>),
    InstallModel(String), InstallResult(Result<serde_json::Value, String>),
    DeleteModel(String), DeleteResult(Result<serde_json::Value, String>),
    SetCategory(String), SearchChanged(String),
}

#[derive(Debug, Clone)]
pub struct ModelItem {
    pub id: String, pub name: String, pub media: String,
    pub family: String, pub installed: bool,
    pub actions: Vec<String>, pub commercial_use_allowed: Option<bool>,
    pub size: Option<String>, pub source: Option<String>,
    pub recommended: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelsPage {
    pub models: Vec<ModelItem>, pub filtered: Vec<ModelItem>,
    pub loading: bool, pub error: Option<String>, pub logs: Vec<String>,
    pub category: String, pub search_query: String,
    pub show_installed_only: bool, pub show_commercial_only: bool,
}

static CATEGORIES: &[(&str, &str)] = &[
    ("all", "全部"), ("image", "图像模型"), ("video", "视频模型"),
    ("audio", "音频模型"), ("loras", "LoRA"), ("installed", "已安装"),
];

impl ModelsPage {
    pub fn new() -> Self { Self::default() }

    fn apply_filters(&mut self) {
        let mut items = self.models.clone();
        if self.category != "all" && self.category != "installed" {
            items.retain(|m| m.media == self.category || (self.category == "loras" && m.family == "lora"));
        }
        if self.category == "installed" { items.retain(|m| m.installed); }
        if !self.search_query.is_empty() {
            let q = self.search_query.to_lowercase();
            items.retain(|m| m.name.to_lowercase().contains(&q) || m.family.to_lowercase().contains(&q));
        }
        if self.show_installed_only { items.retain(|m| m.installed); }
        if self.show_commercial_only { items.retain(|m| m.commercial_use_allowed.unwrap_or(false)); }
        self.filtered = items;
    }

    pub fn update(&mut self, message: Message, api_client: &Option<dq_api::ApiClient>) -> Task<Message> {
        match message {
            Message::NoOp => Task::none(),
            Message::LoadModels => {
                self.loading = true; self.error = None;
                if let Some(client) = api_client.clone() {
                    return Task::perform(async move {
                        match client.get::<serde_json::Value>("/api/models").await {
                            Ok(v) => Message::ModelsLoaded(Ok(v)),
                            Err(e) => Message::ModelsLoaded(Err(e.to_string())),
                        }
                    }, |msg| msg);
                }
                self.push_log("API 客户端未初始化".into()); self.loading = false; Task::none()
            }
            Message::ModelsLoaded(result) => {
                self.loading = false;
                match result {
                    Ok(data) => {
                        self.models.clear();
                        if let Some(models) = data.get("models").and_then(|v| v.as_object()) {
                            for (id, info) in models {
                                let name = info.get("name").and_then(|v| v.as_str()).unwrap_or(id).to_string();
                                let media = info.get("media").and_then(|v| v.as_str()).unwrap_or("image").to_string();
                                let family = info.get("family").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let installed = info.get("installed").and_then(|v| v.as_bool()).unwrap_or(false);
                                let actions: Vec<String> = info.get("actions").and_then(|v| v.as_array()).map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
                                let commercial = info.get("commercial_use_allowed").and_then(|v| v.as_bool());
                                let recommended = info.get("recommended").and_then(|v| v.as_bool());
                                self.models.push(ModelItem { id: id.clone(), name, media, family, installed, actions, commercial_use_allowed: commercial, size: None, source: None, recommended });
                            }
                        }
                        self.push_log(format!("已加载 {} 个模型", self.models.len()));
                        self.apply_filters();
                    }
                    Err(e) => { self.error = Some(e.clone()); self.push_log(format!("加载模型失败: {}", e)); }
                }
                Task::none()
            }
            Message::InstallModel(model_id) => {
                self.push_log(format!("开始安装模型: {}", model_id));
                if let Some(client) = api_client.clone() {
                    let id = model_id.clone();
                    return Task::perform(async move {
                        match client.post::<serde_json::Value, _>(&format!("/api/models/{}/install", id), &serde_json::json!({})).await {
                            Ok(v) => Message::InstallResult(Ok(v)), Err(e) => Message::InstallResult(Err(e.to_string())),
                        }
                    }, |msg| msg);
                }
                Task::none()
            }
            Message::InstallResult(result) => {
                match result { Ok(response) => { let status = response.get("status").and_then(|v| v.as_str()).unwrap_or("unknown"); self.push_log(format!("安装结果: {}", status)); } Err(e) => { self.push_log(format!("安装失败: {}", e)); } }
                Task::none()
            }
            Message::DeleteModel(model_id) => { self.push_log(format!("删除模型: {}", model_id)); Task::none() }
            Message::DeleteResult(result) => { match result { Ok(_) => self.push_log("删除成功".into()), Err(e) => self.push_log(format!("删除失败: {}", e)), } Task::none() }
            Message::SetCategory(c) => { self.category = c; self.apply_filters(); Task::none() }
            Message::SearchChanged(s) => { self.search_query = s; self.apply_filters(); Task::none() }
        }
    }

    fn push_log(&mut self, message: String) {
        let now = chrono::Local::now();
        self.logs.push(format!("[{}] {}", now.format("%H:%M:%S"), message));
        if self.logs.len() > 20 { self.logs.remove(0); }
    }

    fn cat_pill<'a>(label: &'a str, active: bool, _on_press: Message) -> Element<'a, Message> {
        container(text(label).size(typography::CAPTION).color(if active { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY }))
            .padding([6.0, 12.0]).width(Length::Fill)
            .style(move |_theme: &iced::Theme| container::Style {
                background: if active { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None },
                border: iced::Border { color: if active { color::BORDER_SUBTLE } else { iced::Color::TRANSPARENT }, width: 1.0, radius: spacing::RADIUS_SM.into() },
                ..Default::default()
            })
            .into()
    }

    pub fn view(&self, _api_client: &Option<dq_api::ApiClient>) -> Element<'_, Message> {
        let sidebar = column![
            text("分类").size(typography::TITLE).color(color::TEXT_PRIMARY),
        ].spacing(spacing::XS).width(Length::Fixed(140.0));
        let mut sidebar = sidebar;
        for (key, label) in CATEGORIES {
            let _count = match *key {
                "all" => Some(self.models.len()),
                "installed" => Some(self.models.iter().filter(|m| m.installed).count()),
                _ => Some(self.models.iter().filter(|m| m.media == *key || (*key == "loras" && m.family == "lora")).count()),
            };
            let pill = button(Self::cat_pill(label, self.category == *key, Message::SetCategory(key.to_string())))
                .style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() })
                .on_press(Message::SetCategory(key.to_string()));
            sidebar = sidebar.push(pill);
        }

        // Main content
        let header = row![
            text(if self.category == "all" { "模型库" } else { CATEGORIES.iter().find(|(k,_)| *k == self.category).map(|(_,l)| *l).unwrap_or("模型") })
                .size(typography::HEADING).color(color::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            dq_header_button("刷新", Some(Message::LoadModels)),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill);

        let content: Element<Message> = if self.loading {
            container(text("加载中…").size(typography::BODY).color(color::TEXT_SECONDARY)).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).into()
        } else if let Some(ref e) = self.error {
            alert(e, AlertType::Danger, Some(dq_header_button("重试", Some(Message::LoadModels))))
        } else if self.filtered.is_empty() {
            container(column![
                text("暂无模型").size(typography::BODY).color(color::TEXT_SECONDARY),
                dq_button("加载模型列表", ButtonVariant::Primary, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::LoadModels)),
            ].spacing(spacing::MD).align_x(Alignment::Center)).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).into()
        } else {
            let mut list = column![].spacing(spacing::SM).width(Length::Fill);
            for m in &self.filtered {
                let status_tag = if m.installed { tag("已安装", TagType::Success) } else { tag("未安装", TagType::Default) };
                let install_btn: Element<Message> = if m.installed {
                    dq_button("已安装", ButtonVariant::Secondary, ButtonSize::Sm, ButtonWidth::Hug, None)
                } else {
                    dq_button("安装", ButtonVariant::Primary, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::InstallModel(m.id.clone())))
                };
                let card = section_card(&m.name, column![
                    row![
                        tag(m.family.as_str(), TagType::Info),
                        Space::new().width(Length::Fill),
                        status_tag,
                    ].spacing(spacing::XS).width(Length::Fill),
                    row![
                        text(format!("{} · {}", m.media, m.actions.join(", "))).size(typography::CAPTION).color(color::TEXT_TERTIARY),
                        Space::new().width(Length::Fill),
                        install_btn,
                    ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
                ].spacing(spacing::SM).into());
                list = list.push(card);
            }
            scrollable(list).width(Length::Fill).height(Length::Fill).into()
        };

        let main = column![header, content].spacing(spacing::MD).width(Length::Fill).height(Length::Fill);
        let body = row![
            container(sidebar).padding(spacing::MD).style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_PANEL)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_LG.into() }, ..Default::default() }),
            container(main).padding(spacing::MD).width(Length::Fill),
        ].spacing(0).width(Length::Fill).height(Length::Fill);
        container(body).width(Length::Fill).height(Length::Fill).into()
    }
}
