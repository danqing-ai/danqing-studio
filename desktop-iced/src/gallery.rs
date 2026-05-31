use dq_components::{phosphor_icon, PhosphorIcon};
use dq_tokens::{color, spacing, typography};
use iced::widget::{button, column, container, image, row, scrollable, text, Space};
use iced::{Alignment, Element, Length};

#[derive(Debug, Clone)]
pub enum Message {
    LoadAssets,
    AssetsLoaded(Result<Vec<AssetItem>, String>),
    AssetSelected(String),
    Refresh,
}

#[derive(Debug, Clone)]
pub struct AssetItem {
    pub id: String,
    pub name: String,
    pub media_type: String, // image, video, audio
    pub thumbnail_url: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Default)]
pub struct GalleryPage {
    pub assets: Vec<AssetItem>,
    pub loading: bool,
    pub selected_asset: Option<String>,
    pub error: Option<String>,
}

impl GalleryPage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self,
        message: Message,
        api_client: &Option<dq_api::ApiClient>,
    ) -> iced::Task<Message> {
        match message {
            Message::LoadAssets | Message::Refresh => {
                self.loading = true;
                self.error = None;
                if let Some(client) = api_client.clone() {
                    return iced::Task::perform(
                        async move {
                            match client.get::<serde_json::Value>("/api/assets?limit=50").await {
                                Ok(v) => {
                                    let items = v.get("assets")
                                        .and_then(|a| a.as_array())
                                        .cloned()
                                        .unwrap_or_default()
                                        .into_iter()
                                        .filter_map(|a| {
                                            let id = a.get("id")?.as_str()?.to_string();
                                            let name = a.get("name")?.as_str().unwrap_or("").to_string();
                                            let media_type = a.get("media_type")?.as_str().unwrap_or("image").to_string();
                                            let thumbnail_url = format!("{}/api/assets/{}/thumbnail", client.base_url(), id);
                                            let created_at = a.get("created_at")?.as_str().unwrap_or("").to_string();
                                            Some(AssetItem { id, name, media_type, thumbnail_url, created_at })
                                        })
                                        .collect();
                                    Ok(items)
                                }
                                Err(e) => Err(e.to_string()),
                            }
                        },
                        Message::AssetsLoaded,
                    );
                }
                self.loading = false;
                iced::Task::none()
            }
            Message::AssetsLoaded(result) => {
                self.loading = false;
                match result {
                    Ok(assets) => self.assets = assets,
                    Err(e) => self.error = Some(e),
                }
                iced::Task::none()
            }
            Message::AssetSelected(id) => {
                self.selected_asset = Some(id);
                iced::Task::none()
            }
        }
    }

    pub fn view(&self,
        api_client: &Option<dq_api::ApiClient>,
    ) -> Element<Message> {
        let header = row![
            text("作品库")
                .size(typography::BODY)
                .color(color::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            button(text("刷新").size(typography::CAPTION))
                .on_press(Message::Refresh)
                .style(|_theme, _status| button::Style {
                    background: Some(iced::Background::Color(color::BG_SURFACE)),
                    text_color: color::TEXT_SECONDARY,
                    border: iced::Border {
                        color: color::BORDER_SUBTLE,
                        width: 1.0,
                        radius: spacing::RADIUS_SM.into(),
                    },
                    ..Default::default()
                }),
        ]
        .spacing(spacing::SM)
        .align_y(Alignment::Center)
        .width(Length::Fill);

        let content: Element<Message> = if self.loading {
            container(
                column![
                    phosphor_icon(PhosphorIcon::CircleNotch, 32.0, color::TEXT_TERTIARY),
                    text("加载中...")
                        .size(typography::BODY)
                        .color(color::TEXT_SECONDARY),
                ]
                .spacing(spacing::SM)
                .align_x(Alignment::Center),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
        } else if let Some(ref error) = self.error {
            container(
                column![
                    phosphor_icon(PhosphorIcon::Warning, 32.0, color::DANGER),
                    text("加载失败")
                        .size(typography::BODY)
                        .color(color::TEXT_SECONDARY),
                    text(error)
                        .size(typography::CAPTION)
                        .color(color::TEXT_TERTIARY),
                ]
                .spacing(spacing::SM)
                .align_x(Alignment::Center),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
        } else if self.assets.is_empty() {
            container(
                column![
                    phosphor_icon(PhosphorIcon::Image, 32.0, color::TEXT_TERTIARY),
                    text("暂无作品")
                        .size(typography::BODY)
                        .color(color::TEXT_SECONDARY),
                    text("生成图片后将显示在这里")
                        .size(typography::CAPTION)
                        .color(color::TEXT_TERTIARY),
                ]
                .spacing(spacing::SM)
                .align_x(Alignment::Center),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
        } else {
            let mut grid = column![].spacing(spacing::MD).width(Length::Fill);
            for chunk in self.assets.chunks(4) {
                let mut row_items = row![].spacing(spacing::MD).width(Length::Fill);
                for asset in chunk {
                    let is_selected = self.selected_asset.as_ref() == Some(&asset.id);
                    let card = asset_card(asset, is_selected);
                    row_items = row_items.push(card);
                }
                grid = grid.push(row_items);
            }
            scrollable(grid)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        };

        column![header, content]
            .spacing(spacing::MD)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(spacing::LG)
            .into()
    }
}

fn asset_card(asset: &AssetItem, is_selected: bool) -> Element<Message> {
    let border_color = if is_selected { color::ACCENT } else { color::BORDER_SUBTLE };
    
    container(
        column![
            // Thumbnail placeholder
            container(
                phosphor_icon(PhosphorIcon::Image, 48.0, color::TEXT_TERTIARY)
            )
            .width(Length::Fixed(160.0))
            .height(Length::Fixed(120.0))
            .center_x(Length::Fixed(160.0))
            .center_y(Length::Fixed(120.0))
            .style(move |_theme| container::Style {
                background: Some(iced::Background::Color(color::BG_ELEVATED)),
                ..Default::default()
            }),
            text(asset.name.chars().take(20).collect::<String>())
                .size(typography::CAPTION)
                .color(color::TEXT_SECONDARY),
            text(&asset.media_type)
                .size(typography::CAPTION)
                .color(color::TEXT_TERTIARY),
        ]
        .spacing(spacing::XS)
        .align_x(Alignment::Center),
    )
    .width(Length::Fixed(160.0))
    .padding(spacing::SM)
    .style(move |_theme| container::Style {
        background: Some(iced::Background::Color(color::BG_SURFACE)),
        border: iced::Border {
            color: border_color,
            width: if is_selected { 2.0 } else { 1.0 },
            radius: spacing::RADIUS_MD.into(),
        },
        ..Default::default()
    })
    .into()
}
