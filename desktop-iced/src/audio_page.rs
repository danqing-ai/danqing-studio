use dq_tokens::{color, spacing, typography};
use iced::widget::{column, container, text};
use iced::{Alignment, Element, Length};

#[derive(Debug, Clone)]
pub enum Message {
    NoOp,
}

#[derive(Debug, Clone, Default)]
pub struct AudioCreatePage;

impl AudioCreatePage {
    pub fn update(
        &mut self,
        _message: Message,
        _api_client: &Option<dq_api::ApiClient>,
    ) -> iced::Task<Message> {
        iced::Task::none()
    }

    pub fn view(
        &self,
        _api_client: &Option<dq_api::ApiClient>,
    ) -> Element<Message> {
        container(
            column![
                text("音频创作")
                    .size(typography::HEADING)
                    .color(color::TEXT_PRIMARY),
                text("即将推出")
                    .size(typography::BODY)
                    .color(color::TEXT_SECONDARY),
            ]
            .spacing(spacing::MD)
            .align_x(Alignment::Center),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
    }
}
