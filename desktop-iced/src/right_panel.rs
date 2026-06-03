#![allow(dead_code)]
use crate::create_page::{GenerateState, RecentGeneration, MemoryInfo};
use dq_components::{
    dq_button, dq_progress_bar, dq_progress_bar_muted,
    section_card, surface_card,
    ButtonSize, ButtonVariant, ButtonWidth,
    staging_area, StagedResult, StagingMessage,
    image_preview_with_meta,
    phosphor_icon, PhosphorIcon,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{column, container, row, scrollable, text, Space};
use iced::{Alignment, Element, Length};

pub fn right_panel<'a, Message: Clone + 'a>(
    generate_state: GenerateState,
    width: u16, height: u16, model_id: String, seed: &'a str,
    turbo_model: bool, recent: &'a [RecentGeneration],
    staged_results: &'a [StagedResult], memory_info: &'a MemoryInfo,
    enhance_visible: bool, enhance_label: &'a str,
    preview_path: Option<std::path::PathBuf>,
    on_download: Option<Message>, on_preview: Option<Message>,
    on_refresh: Message, on_enhance: Option<Message>,
    on_staging: impl Fn(StagingMessage) -> Message + Clone + 'a,
) -> Element<'a, Message> {
    let mem_warning = if memory_info.total_gb > 0.0 && (memory_info.used_gb / memory_info.total_gb) > 0.85 {
        Some(dq_components::alert("内存使用率超过 85%，建议关闭其他应用以释放内存", dq_components::AlertType::Warning, None))
    } else { None };

    column![
        resource_monitor(memory_info),
        if let Some(w) = mem_warning { w } else { iced::widget::Space::new().height(0).into() },
        if turbo_model { Some(turbo_hint()) } else { None },
        current_preview(generate_state, width, height, model_id, seed, preview_path, on_download, on_preview, enhance_visible, enhance_label, on_enhance),
        if !staged_results.is_empty() { Some(staging_area(staged_results, on_staging)) } else { None },
        recent_generations(recent, on_refresh),
    ].spacing(spacing::MD).width(Length::Fill).height(Length::Fill).into()
}

fn turbo_hint<'a, Message: Clone + 'a>() -> Element<'a, Message> {
    container(text("Turbo 模型已启用：步数与 CFG 已自动优化，适合快速预览。")
        .size(typography::CAPTION).color(color::TEXT_SECONDARY))
        .padding(spacing::SM).width(Length::Fill)
        .style(|_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(color::ACCENT_TINT)),
            border: iced::Border { color: color::ACCENT_MUTED, width: 1.0, radius: spacing::RADIUS_MD.into() },
            ..Default::default()
        }).into()
}

fn resource_monitor<'a, Message: Clone + 'a>(info: &MemoryInfo) -> Element<'a, Message> {
    let mem_ratio = if info.total_gb > 0.0 { (info.used_gb / info.total_gb).clamp(0.0, 1.0) } else { 0.0 };
    let mlx_ratio = if info.total_gb > 0.0 { (info.mlx_active_gb / info.total_gb).clamp(0.0, 1.0) } else { 0.0 };
    section_card("资源监控", column![
        column![
            row![text("内存").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(format!("{:.1} / {:.1} GB", info.used_gb, info.total_gb)).size(typography::CAPTION).color(color::TEXT_TERTIARY)].align_y(Alignment::Center).width(Length::Fill),
            dq_progress_bar(mem_ratio, 3.0),
        ].spacing(4.0).width(Length::Fill),
        column![
            row![text("模型占用").size(typography::LABEL).color(color::TEXT_SECONDARY), Space::new().width(Length::Fill), text(format!("{:.1} / 120 GB", info.mlx_active_gb)).size(typography::CAPTION).color(color::TEXT_TERTIARY)].align_y(Alignment::Center).width(Length::Fill),
            dq_progress_bar_muted(mlx_ratio, 3.0),
        ].spacing(4.0).width(Length::Fill),
    ].spacing(spacing::SM).into())
}

fn current_preview<'a, Message: Clone + 'a>(
    state: GenerateState, width: u16, height: u16, model_id: String, seed: &'a str,
    preview_path: Option<std::path::PathBuf>,
    on_download: Option<Message>, on_preview: Option<Message>,
    enhance_visible: bool, enhance_label: &'a str, on_enhance: Option<Message>,
) -> Element<'a, Message> {
    let meta = format!("{} · {}×{} · seed {}", model_id, width, height, seed);

    let preview: Element<'a, Message> = match state {
        GenerateState::Idle => {
            if let Some(ref path) = preview_path {
                image_preview_with_meta(Some(path), "", Length::Fill, Length::Fixed(340.0))
            } else {
                container(phosphor_icon(PhosphorIcon::Image, 48.0, color::TEXT_QUATERNARY))
                    .width(Length::Fill).height(Length::Fixed(340.0)).align_x(Alignment::Center).align_y(Alignment::Center)
                    .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() }).into()
            }
        }
        GenerateState::Submitting | GenerateState::Generating { .. } => {
            container(column![
                phosphor_icon(PhosphorIcon::CircleNotch, 32.0, color::ACCENT),
                text("生成中…").size(typography::BODY).color(color::TEXT_SECONDARY),
            ].spacing(spacing::SM).align_x(Alignment::Center)).width(Length::Fill).height(Length::Fixed(340.0)).align_x(Alignment::Center).align_y(Alignment::Center)
                .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() }).into()
        }
        GenerateState::Done => {
            if let Some(ref path) = preview_path {
                image_preview_with_meta(Some(path), "", Length::Fill, Length::Fixed(340.0))
            } else {
                container(phosphor_icon(PhosphorIcon::Image, 48.0, color::SUCCESS))
                    .width(Length::Fill).height(Length::Fixed(340.0)).align_x(Alignment::Center).align_y(Alignment::Center)
                    .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() }).into()
            }
        }
    };

    let mut body = column![
        container(preview).width(Length::Fill).height(Length::Fixed(340.0)).max_width(340.0).align_x(Alignment::Center),
        text(meta).size(typography::CAPTION).color(color::TEXT_QUATERNARY),
    ].spacing(spacing::SM).width(Length::Fill).align_x(Alignment::Center);

    if matches!(state, GenerateState::Done) {
        body = body.push(container(row![
            dq_button("下载", ButtonVariant::Secondary, ButtonSize::Sm, ButtonWidth::Hug, on_download),
            dq_button("放大预览", ButtonVariant::Ghost, ButtonSize::Sm, ButtonWidth::Hug, on_preview),
        ].spacing(spacing::SM)).align_x(Alignment::Center).width(Length::Fill));
    }

    if enhance_visible {
        body = body.push(dq_button(enhance_label, ButtonVariant::Secondary, ButtonSize::Sm, ButtonWidth::Fill, on_enhance));
    }

    section_card("当前预览", body.into())
}

fn recent_generations<'a, Message: Clone + 'a>(recent: &'a [RecentGeneration], on_refresh: Message) -> Element<'a, Message> {
    let body: Element<'a, Message> = if recent.is_empty() {
        container(text("暂无生成记录").size(typography::LABEL).color(color::TEXT_TERTIARY)).width(Length::Fill).padding(spacing::LG).align_x(Alignment::Center).into()
    } else {
        let mut list = column![].spacing(spacing::XS).width(Length::Fill);
        for item in recent.iter().take(8) {
            list = list.push(container(row![
                container(phosphor_icon(PhosphorIcon::Image, 20.0, color::TEXT_QUATERNARY))
                    .width(Length::Fixed(48.0)).height(Length::Fixed(48.0))
                    .align_x(Alignment::Center).align_y(Alignment::Center)
                    .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_INSET)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }),
                text(item.title.as_str()).size(typography::BODY).color(color::TEXT_PRIMARY),
            ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill))
                .padding(spacing::SM).width(Length::Fill)
                .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_SURFACE)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }));
        }
        scrollable(list).width(Length::Fill).height(Length::Fill).into()
    };

    let refresh_btn = iced::widget::button(phosphor_icon(PhosphorIcon::ArrowsClockwise, 14.0, color::TEXT_TERTIARY))
        .on_press(on_refresh)
        .style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() });

    surface_card(Some(text("最近生成").size(typography::TITLE).color(color::TEXT_PRIMARY).into()), Some(refresh_btn.into()), Some(body))
}
