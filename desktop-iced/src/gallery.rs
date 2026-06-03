#![allow(dead_code)]
use dq_components::{
    alert, dq_button, dq_header_button, empty_state, loading_state, surface_card, tag,
    AlertType, TagType, ButtonSize, ButtonVariant, ButtonWidth,
    phosphor_icon, PhosphorIcon,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{button, column, container, row, scrollable, text, Space};
use iced::{Alignment, Element, Length, Task};

#[derive(Debug, Clone)]
pub enum Message {
    NoOp, LoadAssets,
    AssetsLoaded(Result<Vec<AssetItem>, String>),
    ThumbnailLoaded(String, Vec<u8>), // asset_id, image bytes
    AssetSelected(String), Refresh, ToggleView,
    SetFilter(MediaFilter), SetTimeFilter(TimeFilter),
    OpenDetail(String), CloseDetail,
    OpenLightbox(usize), CloseLightbox, LightboxPrev, LightboxNext,
    ToggleSelect(String), SelectAll, DeselectAll,
    BatchDownload, BatchDelete,
    GoToCreate,
    DownloadItem(String), DeleteItem(String),
    UseForImg2Img(String), CopyPrompt(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum MediaFilter { #[default] All, Image, Video, Audio }
impl MediaFilter {
    const ALL: &'static [MediaFilter] = &[MediaFilter::All, MediaFilter::Image, MediaFilter::Video, MediaFilter::Audio];
    fn label(&self) -> &'static str { match self { MediaFilter::All => "全部", MediaFilter::Image => "图片", MediaFilter::Video => "视频", MediaFilter::Audio => "音频" } }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TimeFilter { #[default] All, Today, ThisWeek, ThisMonth }
impl TimeFilter {
    const ALL: &'static [TimeFilter] = &[TimeFilter::All, TimeFilter::Today, TimeFilter::ThisWeek, TimeFilter::ThisMonth];
    fn label(&self) -> &'static str { match self { TimeFilter::All => "全部", TimeFilter::Today => "今天", TimeFilter::ThisWeek => "本周", TimeFilter::ThisMonth => "本月" } }
}

#[derive(Debug, Clone)]
pub struct AssetItem {
    pub id: String, pub name: String, pub media_type: String,
    pub thumbnail_url: String, pub file_url: String, pub created_at: String,
    pub width: Option<i32>, pub height: Option<i32>, pub duration_seconds: Option<f32>,
    pub model: Option<String>, pub prompt: Option<String>,
    pub steps: Option<i32>, pub guidance: Option<f32>, pub seed: Option<i64>, pub kind: String,
    pub thumbnail_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
pub struct GalleryPage {
    pub assets: Vec<AssetItem>, pub filtered_assets: Vec<AssetItem>,
    pub loading: bool, pub selected_asset: Option<String>,
    pub selected_ids: std::collections::HashSet<String>, pub error: Option<String>,
    pub grid_view: bool, pub media_filter: MediaFilter, pub time_filter: TimeFilter,
    pub detail_asset: Option<AssetItem>, pub show_detail: bool,
    pub lightbox_items: Vec<AssetItem>, pub lightbox_index: usize, pub show_lightbox: bool,
    pub page: usize, pub has_more: bool, pub loading_more: bool,
    pub api_available: bool,
}

impl GalleryPage {
    pub fn new() -> Self { Self::default() }

    fn apply_filters(&mut self) {
        let mut items = self.assets.clone();
        if self.media_filter != MediaFilter::All {
            let mtype = match self.media_filter { MediaFilter::Image => "image", MediaFilter::Video => "video", MediaFilter::Audio => "audio", _ => "" };
            items.retain(|a| a.media_type == mtype);
        }
        if self.time_filter != TimeFilter::All {
            let now = chrono::Local::now().format("%Y-%m-%d").to_string();
            items.retain(|a| {
                let d = if a.created_at.len() >= 10 { &a.created_at[..10] } else { return true; };
                match self.time_filter {
                    TimeFilter::Today => d == now,
                    TimeFilter::ThisWeek => d.get(0..7) == now.get(0..7),
                    TimeFilter::ThisMonth => d.get(0..7) == now.get(0..7),
                    _ => true,
                }
            });
        }
        self.filtered_assets = items;
    }

    /// Start async thumbnail loading for all assets
    fn load_thumbnails(&self, client: &dq_api::ApiClient) -> Task<Message> {
        let mut batch = Vec::new();
        for asset in &self.assets {
            if asset.thumbnail_data.is_some() { continue; }
            let client = client.clone();
            let id = asset.id.clone();
            // Use asset's thumbnail_url if available (may be relative), otherwise construct from id
            let url = if asset.thumbnail_url.starts_with("http") {
                asset.thumbnail_url.clone()
            } else {
                format!("{}{}", client.base_url(), asset.thumbnail_url)
            };
            let task = Task::perform(async move {
                match client.download_bytes(&url).await {
                    Ok(bytes) => {
                        Message::ThumbnailLoaded(id, bytes)
                    }
                    Err(_) => Message::NoOp,
                }
            }, |m| m);
            batch.push(task);
        }
        if batch.is_empty() { Task::none() } else { Task::batch(batch) }
    }

    pub fn update(&mut self, message: Message, api_client: &Option<dq_api::ApiClient>) -> Task<Message> {
        match message {
            Message::NoOp => Task::none(),
            Message::LoadAssets | Message::Refresh => {
                self.loading = true; self.error = None; self.page = 0; self.api_available = api_client.is_some();
                if let Some(client) = api_client.clone() {
                    return Task::batch([Task::perform(async move {
                        match client.get::<serde_json::Value>("/api/assets?limit=100").await {
                            Ok(v) => {
                                let items = v.get("items").and_then(|a| a.as_array()).cloned().unwrap_or_default()
                                    .into_iter().filter_map(|a| {
                                        let id_str = a.get("id")?.as_str()?.to_string();
                                        let thumb_url = a.get("thumbnail_url").and_then(|u| u.as_str()).map(|s| s.to_string())
                                            .unwrap_or_else(|| format!("{}/api/assets/{}/thumbnail", client.base_url(), &id_str));
                                        let f_url = format!("{}/api/assets/{}/file", client.base_url(), &id_str);
                                        let item_kind = a.get("kind")?.as_str().unwrap_or("image");
                                        let mime = a.get("mime_type").and_then(|v| v.as_str()).unwrap_or("image/png");
                                        let media_type = if mime.starts_with("video") { "video" } else if mime.starts_with("audio") { "audio" } else { item_kind };
                                        Some(AssetItem {
                                            id: id_str.clone(), name: id_str,
                                            media_type: media_type.to_string(),
                                            thumbnail_url: thumb_url,
                                            file_url: f_url,
                                            created_at: a.get("created_at")?.as_str().unwrap_or("").to_string(),
                                            width: a.get("width").and_then(|v| v.as_i64()).map(|v| v as i32),
                                            height: a.get("height").and_then(|v| v.as_i64()).map(|v| v as i32),
                                            duration_seconds: a.get("duration_seconds").and_then(|v| v.as_f64()).map(|v| v as f32),
                                            model: a.get("metadata").and_then(|m| m.get("model")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                                            prompt: a.get("metadata").and_then(|m| m.get("prompt")).and_then(|v| v.as_str()).map(|s| s.to_string()),
                                            steps: a.get("metadata").and_then(|m| m.get("steps")).and_then(|v| v.as_i64()).map(|v| v as i32),
                                            guidance: a.get("metadata").and_then(|m| m.get("guidance")).and_then(|v| v.as_f64()).map(|v| v as f32),
                                            seed: a.get("metadata").and_then(|m| m.get("seed")).and_then(|v| v.as_i64()),
                                            kind: item_kind.to_string(),
                                            thumbnail_data: None,
                                        })
                                    }).collect();
                                Message::AssetsLoaded(Ok(items))
                            }
                            Err(e) => Message::AssetsLoaded(Err(e.to_string())),
                        }
                    }, |m| m)]);
                }
                self.loading = false; Task::none()
            }
            Message::AssetsLoaded(result) => {
                self.loading = false; self.loading_more = false;
                match result {
                    Ok(assets) => {
                        self.assets = assets; self.has_more = self.assets.len() >= 100;
                        self.apply_filters();
                        // Start thumbnail loading
                        if let Some(client) = api_client { return self.load_thumbnails(client); }
                    }
                    Err(e) => { self.error = Some(format!("后端服务未就绪: {}", e)); }
                }
                Task::none()
            }
            Message::ThumbnailLoaded(id, bytes) => {
                if let Some(asset) = self.assets.iter_mut().find(|a| a.id == id) {
                    asset.thumbnail_data = Some(bytes.clone());
                }
                if let Some(asset) = self.filtered_assets.iter_mut().find(|a| a.id == id) {
                    asset.thumbnail_data = Some(bytes);
                }
                Task::none()
            }
            Message::AssetSelected(id) => { self.selected_asset = Some(id); Task::none() }
            Message::ToggleView => { self.grid_view = !self.grid_view; Task::none() }
            Message::SetFilter(f) => { self.media_filter = f; self.apply_filters(); Task::none() }
            Message::SetTimeFilter(f) => { self.time_filter = f; self.apply_filters(); Task::none() }
            Message::OpenDetail(id) => { if let Some(a) = self.assets.iter().find(|a| a.id == id).cloned() { self.detail_asset = Some(a); self.show_detail = true; } Task::none() }
            Message::CloseDetail => { self.show_detail = false; self.detail_asset = None; Task::none() }
            Message::OpenLightbox(idx) => { self.lightbox_items = self.filtered_assets.clone(); self.lightbox_index = idx.min(self.filtered_assets.len().saturating_sub(1)); self.show_lightbox = true; Task::none() }
            Message::CloseLightbox => { self.show_lightbox = false; self.lightbox_items.clear(); Task::none() }
            Message::LightboxPrev => { if self.lightbox_index > 0 { self.lightbox_index -= 1; } Task::none() }
            Message::LightboxNext => { if self.lightbox_index + 1 < self.lightbox_items.len() { self.lightbox_index += 1; } Task::none() }
            Message::ToggleSelect(id) => { if self.selected_ids.contains(&id) { self.selected_ids.remove(&id); } else { self.selected_ids.insert(id); } Task::none() }
            Message::SelectAll => { self.selected_ids = self.filtered_assets.iter().map(|a| a.id.clone()).collect(); Task::none() }
            Message::DeselectAll => { self.selected_ids.clear(); Task::none() }
            Message::GoToCreate | Message::BatchDownload | Message::DownloadItem(_) | Message::CopyPrompt(_) | Message::UseForImg2Img(_) => Task::none(),
            Message::BatchDelete => { self.assets.retain(|a| !self.selected_ids.contains(&a.id)); self.selected_ids.clear(); self.apply_filters(); Task::none() }
            Message::DeleteItem(id) => { self.assets.retain(|a| a.id != id); self.apply_filters(); Task::none() }
        }
    }

    pub fn view(&self, api_client: &Option<dq_api::ApiClient>) -> Element<'_, Message> {
        // API not available
        if api_client.is_none() {
            return container(empty_state(Some(PhosphorIcon::Images), "无法连接到后端服务", "请确保后端 API 正在运行（默认 http://127.0.0.1:7800）\n启动后点击「连接」并刷新", Some(dq_button("连接并刷新", ButtonVariant::Primary, ButtonSize::Md, ButtonWidth::Hug, Some(Message::Refresh)))))
                .width(Length::Fill).height(Length::Fill).into();
        }

        // Toolbar
        let mut toolbar = row![].spacing(spacing::XS).align_y(Alignment::Center).width(Length::Fill);
        for f in MediaFilter::ALL {
            let active = self.media_filter == *f;
            let pill: Element<Message> = container(text(f.label()).size(typography::LABEL).color(if active { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY })).padding([4.0, 10.0]).style(move |_theme: &iced::Theme| container::Style { background: if active { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None }, border: iced::Border { color: if active { color::BORDER_SUBTLE } else { iced::Color::TRANSPARENT }, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }).into();
            toolbar = toolbar.push(pill);
        }

        let view_toggle = button(phosphor_icon(if self.grid_view { PhosphorIcon::List } else { PhosphorIcon::SquaresFour }, 14.0, color::TEXT_SECONDARY))
            .on_press(Message::ToggleView).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() });
        let refresh_btn = button(phosphor_icon(PhosphorIcon::ArrowsClockwise, 14.0, color::TEXT_SECONDARY))
            .on_press(Message::Refresh).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() });

        // Content
        let content: Element<Message> = if self.loading {
            loading_state("加载作品列表…")
        } else if let Some(ref e) = self.error {
            alert(e, AlertType::Danger, Some(dq_header_button("重试", Some(Message::Refresh))))
        } else if self.filtered_assets.is_empty() && !self.assets.is_empty() {
            empty_state(Some(PhosphorIcon::Images), "无匹配结果", "尝试调整筛选条件以查看更多内容", Some(dq_button("显示全部", ButtonVariant::Ghost, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::SetFilter(MediaFilter::All)))))
        } else if self.assets.is_empty() {
            {
                let action: Element<Message> = column![
                    dq_button("去创作", ButtonVariant::Primary, ButtonSize::Md, ButtonWidth::Hug, Some(Message::GoToCreate)),
                    dq_button("刷新", ButtonVariant::Ghost, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::Refresh)),
                ].spacing(spacing::SM).align_x(Alignment::Center).into();
                empty_state(Some(PhosphorIcon::Sparkle), "暂无作品", "去「图像创作」页面生成第一张图片吧！完成后作品将自动显示在这里", Some(action))
            }
        } else if self.grid_view {
            grid_view(&self.filtered_assets, &self.selected_ids)
        } else {
            list_view(&self.filtered_assets, &self.selected_ids)
        };

        // Batch bar
        let batch_bar: Option<Element<Message>> = if !self.selected_ids.is_empty() {
            Some(container(row![text(format!("已选 {} 项", self.selected_ids.len())).size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), dq_header_button("全选", Some(Message::SelectAll)), dq_header_button("取消选择", Some(Message::DeselectAll)), dq_header_button("批量下载", Some(Message::BatchDownload)), container(text("批量删除").size(typography::CAPTION).color(color::DANGER)).padding([4.0, 8.0]).style(|_theme: &iced::Theme| container::Style { border: iced::Border { color: color::DANGER, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() })].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill)).padding(spacing::SM).width(Length::Fill).style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::FILL_SELECTED)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }).into())
        } else { None };

        let mut body = column![
            surface_card(Some(row![text("作品库").size(typography::HEADING).color(color::TEXT_PRIMARY), Space::new().width(Length::Fill), view_toggle, refresh_btn].spacing(spacing::SM).into()), None, None),
            toolbar,
        ].spacing(spacing::MD).width(Length::Fill).height(Length::Fill).padding(spacing::MD);
        if let Some(b) = batch_bar { body = body.push(b); }
        body = body.push(content);

        // Detail side panel
        if self.show_detail {
            let detail = detail_panel(&self.detail_asset);
            return container(row![container(body).width(Length::FillPortion(60)).height(Length::Fill), container(detail).width(Length::FillPortion(40)).height(Length::Fill)].width(Length::Fill).height(Length::Fill)).width(Length::Fill).height(Length::Fill).into();
        }

        // Lightbox overlay
        if self.show_lightbox && !self.lightbox_items.is_empty() {
            let lb = lightbox_view(&self.lightbox_items, self.lightbox_index);
            return container(column![body, lb].spacing(0).width(Length::Fill).height(Length::Fill)).width(Length::Fill).height(Length::Fill).into();
        }

        container(body).width(Length::Fill).height(Length::Fill).into()
    }
}

fn grid_view<'a>(assets: &'a [AssetItem], selected_ids: &'a std::collections::HashSet<String>) -> Element<'a, Message> where Message: 'a {
    let mut grid = column![].spacing(spacing::MD).width(Length::Fill);
    for (global_idx, chunk) in assets.chunks(4).enumerate() {
        let mut r = row![].spacing(spacing::MD).width(Length::Fill);
        for (local_idx, asset) in chunk.iter().enumerate() {
            let _idx = global_idx * 4 + local_idx;
            let _sel = selected_ids.contains(&asset.id);
            let thumb: Element<Message> = if let Some(ref data) = asset.thumbnail_data {
                let img = iced::widget::image(iced::widget::image::Handle::from_bytes(data.clone()))
                    .width(Length::Fixed(160.0)).height(Length::Fixed(120.0));
                container(img).width(Length::Fixed(160.0)).height(Length::Fixed(120.0)).into()
            } else {
                container(phosphor_icon(match asset.media_type.as_str() { "video" => PhosphorIcon::VideoCamera, "audio" => PhosphorIcon::SpeakerHigh, _ => PhosphorIcon::Image }, 40.0, color::TEXT_TERTIARY))
                    .width(Length::Fixed(160.0)).height(Length::Fixed(120.0)).center_x(Length::Fixed(160.0)).center_y(Length::Fixed(120.0))
                    .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_ELEVATED)), ..Default::default() }).into()
            };
            let card = surface_card(
                None,
                Some(button(phosphor_icon(PhosphorIcon::MagnifyingGlass, 12.0, color::TEXT_TERTIARY)).on_press(Message::OpenDetail(asset.id.clone())).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }).into()),
                Some(column![
                    thumb,
                    text(asset.name.chars().take(20).collect::<String>()).size(typography::CAPTION).color(color::TEXT_SECONDARY),
                    row![tag(asset.media_type.as_str(), TagType::Info), Space::new().width(Length::Fill)].spacing(spacing::XS),
                ].spacing(spacing::XS).into()),
            );
            r = r.push(card);
        }
        grid = grid.push(r);
    }
    scrollable(grid).width(Length::Fill).height(Length::Fill).into()
}

fn list_view<'a>(assets: &'a [AssetItem], selected_ids: &'a std::collections::HashSet<String>) -> Element<'a, Message> where Message: 'a {
    let mut list = column![].spacing(spacing::XS).width(Length::Fill);
    for asset in assets {
        let sel = selected_ids.contains(&asset.id);
        list = list.push(container(row![
            container(phosphor_icon(match asset.media_type.as_str() { "video" => PhosphorIcon::VideoCamera, "audio" => PhosphorIcon::SpeakerHigh, _ => PhosphorIcon::Image }, 20.0, color::TEXT_QUATERNARY)).width(Length::Fixed(40.0)).height(Length::Fixed(40.0)).align_x(Alignment::Center).align_y(Alignment::Center).style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }),
            column![text(asset.name.chars().take(25).collect::<String>()).size(typography::BODY).color(color::TEXT_PRIMARY), text(asset.model.as_deref().unwrap_or("unknown")).size(typography::CAPTION).color(color::TEXT_TERTIARY)].spacing(spacing::XS).width(Length::Fill),
            Space::new().width(Length::Fill),
            tag(asset.media_type.as_str(), TagType::Info),
            button(phosphor_icon(PhosphorIcon::Download, 12.0, color::TEXT_TERTIARY)).on_press(Message::DownloadItem(asset.id.clone())).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill)).padding(spacing::SM).width(Length::Fill).style(move |_theme: &iced::Theme| container::Style { background: if sel { Some(iced::Background::Color(color::FILL_SELECTED)) } else { Some(iced::Background::Color(color::BG_SURFACE)) }, border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }));
    }
    scrollable(list).width(Length::Fill).height(Length::Fill).into()
}

fn detail_panel(asset: &Option<AssetItem>) -> Element<'_, Message> {
    if let Some(ref asset) = asset {
        let resolution = asset.width.map(|w| format!("{}×{}", w, asset.height.unwrap_or(0))).unwrap_or_else(|| "-".to_string());
        let time = asset.created_at.chars().take(19).collect::<String>();
        let model = asset.model.as_deref().unwrap_or("-").to_string();

        surface_card(
            Some(row![text("详情").size(typography::TITLE).color(color::TEXT_PRIMARY), Space::new().width(Length::Fill), dq_header_button("关闭", Some(Message::CloseDetail))].spacing(spacing::SM).into()),
            None,
            Some(column![
                container(phosphor_icon(match asset.media_type.as_str() { "video" => PhosphorIcon::VideoCamera, "audio" => PhosphorIcon::SpeakerHigh, _ => PhosphorIcon::Image }, 48.0, color::TEXT_TERTIARY)).width(Length::Fill).height(Length::Fixed(160.0)).align_x(Alignment::Center).align_y(Alignment::Center).style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() }),
                row![text("模型").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(model).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).width(Length::Fill),
                row![text("分辨率").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(resolution).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).width(Length::Fill),
                row![text("时间").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(time).size(typography::BODY).color(color::TEXT_PRIMARY)].spacing(spacing::SM).width(Length::Fill),
                if let Some(ref p) = asset.prompt { let c: Element<Message> = column![text("提示词").size(typography::LABEL).color(color::TEXT_SECONDARY), text(p).size(typography::CAPTION).color(color::TEXT_TERTIARY)].spacing(spacing::XS).width(Length::Fill).into(); c } else { Space::new().height(0).into() },
                row![dq_button("下载", ButtonVariant::Secondary, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::DownloadItem(asset.id.clone()))), if asset.media_type == "image" { dq_button("用于图生图", ButtonVariant::Ghost, ButtonSize::Sm, ButtonWidth::Hug, Some(Message::UseForImg2Img(asset.id.clone()))) } else { Space::new().width(0).into() }].spacing(spacing::SM),
            ].spacing(spacing::MD).into()),
        )
    } else { Space::new().into() }
}

fn lightbox_view(items: &[AssetItem], index: usize) -> Element<'_, Message> {
    if items.is_empty() { return Space::new().into(); }
    let total = items.len();
    container(column![
        row![text(format!("{}/{}", index + 1, total)).size(typography::BODY).color(color::TEXT_PRIMARY), Space::new().width(Length::Fill),
            button(phosphor_icon(PhosphorIcon::Download, 14.0, color::TEXT_SECONDARY)).on_press(Message::DownloadItem(items[index].id.clone())).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
            button(phosphor_icon(PhosphorIcon::X, 14.0, color::TEXT_SECONDARY)).on_press(Message::CloseLightbox).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
        row![
            button(phosphor_icon(PhosphorIcon::ArrowLeft, 24.0, color::TEXT_SECONDARY)).on_press_maybe(if index > 0 { Some(Message::LightboxPrev) } else { None }).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
            container(text(items[index].name.chars().take(40).collect::<String>()).size(typography::BODY).color(color::TEXT_PRIMARY)).width(Length::Fill).align_x(Alignment::Center).align_y(Alignment::Center),
            button(phosphor_icon(PhosphorIcon::ArrowRight, 24.0, color::TEXT_SECONDARY)).on_press_maybe(if index + 1 < total { Some(Message::LightboxNext) } else { None }).style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
        ].spacing(spacing::MD).align_y(Alignment::Center).width(Length::Fill).height(Length::Fill),
    ].spacing(spacing::MD).width(Length::Fill).height(Length::Fill).padding(spacing::LG))
        .width(Length::Fill).height(Length::Fill)
        .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_OVERLAY)), ..Default::default() })
        .into()
}
