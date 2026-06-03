mod api;
mod app;
mod audio_page;
mod create_page;
mod gallery;
mod models_page;
mod right_panel;
mod settings_page;
mod task_queue;
mod video_page;

use app::App;

fn main() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title("DanQing Studio v4")
        .theme(|app: &App| app.current_theme.to_iced_theme())
        .font(dq_theme::INTER)
        .font(dq_components::PHOSPHOR_REGULAR)
        .font(dq_components::PHOSPHOR_BOLD)
        .font(dq_components::PHOSPHOR_FILL)
        .default_font(dq_theme::inter())
        .subscription(App::subscription)
        .window(iced::window::Settings {
            size: iced::Size::new(1360.0, 880.0),
            ..Default::default()
        })
        .run()
}
