use crate::create_page::{GenerateState, RecentGeneration, MemoryInfo};
use dq_components::{
    dq_button, dq_preview_canvas, dq_progress_bar, dq_progress_bar_muted,
    dq_section, refresh_icon, ButtonSize, ButtonVariant,
    ButtonWidth, PreviewState, SectionIcon,
    staging_area, StagedResult, StagingMessage,
    image_preview_with_meta,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{column, container, row, scrollable, text, Space};
use iced::{Alignment, Element, Length};

/// Right panel — adaptive layout:
/// 1. Resource monitor (fixed compact height)
/// 2. Current preview (fills width, keeps square aspect, max 340)
/// 3. Generation staging area (batch results grid)
/// 4. Recent generations (scrollable list, fills remaining height)
pub fn right_panel<'a, Message: Clone + 'a>(
    generate_state: GenerateState,
    width: u16,
    height: u16,
    model_id: String,
    seed: &'a str,
    turbo_model: bool,
    recent: &'a [RecentGeneration],
    staged_results: &'a [StagedResult],
    memory_info: &'a MemoryInfo,
    enhance_visible: bool,
    enhance_label: &'a str,
    preview_path: Option<std::path::PathBuf>,
    on_download: Option<Message>,
    on_preview: Option<Message>,
    on_refresh: Message,
    on_enhance: Option<Message>,
    on_staging: impl Fn(StagingMessage) -> Message + Clone + 'a,
) -> Element<'a, Message> {
    column![
        resource_monitor(memory_info),
        if turbo_model { Some(turbo_hint()) } else { None },
        current_preview(generate_state, width, height, model_id, seed, preview_path, on_download, on_preview, enhance_visible, enhance_label, on_enhance),
        if !staged_results.is_empty() {
            Some(staging_area(staged_results, on_staging))
        } else {
            None
        },
        recent_generations(recent, on_refresh),
    ]
    .spacing(spacing::MD)
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn turbo_hint<'a, Message: Clone + 'a>() -> Element<'a, Message> {
    container(
        text("Turbo 模型已启用：步数与 CFG 已自动优化，适合快速预览。")
            .size(typography::CAPTION)
            .color(color::TEXT_SECONDARY),
    )
    .padding(spacing::SM)
    .width(Length::Fill)
    .style(|_theme| iced::widget::container::Style {
        background: Some(iced::Background::Color(color::ACCENT_TINT)),
        border: iced::Border {
            color: color::ACCENT_MUTED,
            width: 1.0,
            radius: spacing::RADIUS_MD.into(),
        },
        ..Default::default()
    })
    .into()
}

fn resource_monitor<'a, Message: Clone + 'a>(
    info: &MemoryInfo,
) -> Element<'a, Message> {
    let mem_ratio = if info.total_gb > 0.0 {
        (info.used_gb / info.total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mlx_ratio = if info.total_gb > 0.0 {
        (info.mlx_active_gb / info.total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let body = column![
        resource_bar_live("内存", info.used_gb, info.total_gb, mem_ratio, true),
        resource_bar_live("模型占用", info.mlx_active_gb, 120.0, mlx_ratio, false),
    ]
    .spacing(spacing::SM)
    .width(Length::Fill)
    .into();

    dq_section(SectionIcon::Cube, "资源监控", None, Some(body))
}

fn resource_bar_live<'a, Message: Clone + 'a>(
    label: &'a str,
    used: f32,
    total: f32,
    ratio: f32,
    accent: bool,
) -> Element<'a, Message> {
    let bar = if accent {
        dq_progress_bar(ratio, 3.0)
    } else {
        dq_progress_bar_muted(ratio, 3.0)
    };

    column![
        row![
            text(label)
                .size(typography::LABEL)
                .color(color::TEXT_SECONDARY),
            Space::new().width(Length::Fill),
            text(format!("{:.1} / {:.1} GB", used, total))
                .size(typography::CAPTION)
                .color(color::TEXT_TERTIARY),
        ]
        .align_y(Alignment::Center)
        .width(Length::Fill),
        bar,
    ]
    .spacing(4.0)
    .width(Length::Fill)
    .into()
}

fn current_preview<'a, Message: Clone + 'a>(
    state: GenerateState,
    width: u16,
    height: u16,
    model_id: String,
    seed: &str,
    preview_path: Option<std::path::PathBuf>,
    on_download: Option<Message>,
    on_preview: Option<Message>,
    enhance_visible: bool,
    enhance_label: &'a str,
    on_enhance: Option<Message>,
) -> Element<'a, Message> {
    let meta = format!("{model_id} · {width}×{height} · seed {seed}");

    // Adaptive square preview: fixed height 340, no Fill fighting with scrollable.
    let canvas: Element<'a, Message> = match state {
        GenerateState::Idle => {
            if let Some(ref path) = preview_path {
                image_preview_with_meta(
                    Some(path),
                    "",
                    Length::Fill,
                    Length::Fixed(340.0),
                )
            } else {
                container(
                    dq_components::phosphor_icon(
                        dq_components::PhosphorIcon::Image,
                        48.0,
                        color::TEXT_QUATERNARY,
                    )
                )
                .width(Length::Fill)
                .height(Length::Fixed(340.0))
                .align_x(Alignment::Center)
                .align_y(Alignment::Center)
                .style(|_theme| iced::widget::container::Style {
                    background: Some(iced::Background::Color(color::BG_INSET)),
                    border: iced::Border {
                        color: color::BORDER_SUBTLE,
                        width: 1.0,
                        radius: spacing::RADIUS_MD.into(),
                    },
                    ..Default::default()
                })
                .into()
            }
        }
        GenerateState::Submitting => {
            dq_preview_canvas::<Message>(PreviewState::Generating { progress: 2 }, 340.0)
        }
        GenerateState::Generating { progress, .. } => {
            dq_preview_canvas::<Message>(PreviewState::Generating { progress }, 340.0)
        }
        GenerateState::Done => {
            if let Some(ref path) = preview_path {
                let meta_static: &'static str = Box::leak(meta.clone().into_boxed_str());
                image_preview_with_meta(
                    Some(path),
                    meta_static,
                    Length::Fill,
                    Length::Fixed(340.0),
                )
            } else {
                dq_preview_canvas::<Message>(PreviewState::Done, 340.0)
            }
        }
    };

    // Fixed height wrapper — no Fill fighting with recent_generations scrollable
    let square_wrapper: Element<'a, Message> = container(canvas)
        .width(Length::Fill)
        .height(Length::Fixed(340.0))
        .max_width(340.0)
        .align_x(Alignment::Center)
        .into();

    let mut body = column![
        square_wrapper,
        text(meta)
            .size(typography::CAPTION)
            .color(color::TEXT_QUATERNARY),
    ]
    .spacing(spacing::SM)
    .width(Length::Fill)
    .align_x(Alignment::Center);

    if matches!(state, GenerateState::Done) {
        body = body.push(
            container(
                row![
                    dq_button("下载", ButtonVariant::Secondary, ButtonSize::Sm, ButtonWidth::Hug, on_download),
                    dq_button("放大预览", ButtonVariant::Ghost, ButtonSize::Sm, ButtonWidth::Hug, on_preview),
                ]
                .spacing(spacing::SM),
            )
            .align_x(Alignment::Center)
            .width(Length::Fill),
        );
    }

    if enhance_visible {
        body = body.push(
            dq_button(enhance_label, ButtonVariant::Secondary, ButtonSize::Sm, ButtonWidth::Fill, on_enhance),
        );
    }

    dq_section(SectionIcon::Document, "当前预览", None, Some(body.into()))
}

fn recent_generations<'a, Message: Clone + 'a>(
    recent: &'a [RecentGeneration],
    on_refresh: Message,
) -> Element<'a, Message> {
    let list_body: Element<'a, Message> = if recent.is_empty() {
        container(
            text("暂无生成记录")
                .size(typography::LABEL)
                .color(color::TEXT_TERTIARY),
        )
        .width(Length::Fill)
        .padding(spacing::LG)
        .align_x(Alignment::Center)
        .into()
    } else {
        let mut list = column![].spacing(spacing::XS).width(Length::Fill);
        for item in recent.iter().take(8) {
            list = list.push(recent_list_item(item));
        }
        scrollable(list)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    };

    let refresh_btn: Element<'a, Message> = iced::widget::button(refresh_icon::<Message>())
        .on_press(on_refresh)
        .style(|_theme, _status| iced::widget::button::Style {
            background: None,
            ..Default::default()
        })
        .into();

    dq_section(SectionIcon::Clock, "最近生成", Some(refresh_btn), Some(list_body))
}

fn recent_list_item<'a, Message: Clone + 'a>(
    item: &'a RecentGeneration,
) -> Element<'a, Message> {
    row![
        container(
            dq_components::phosphor_icon(
                dq_components::PhosphorIcon::Image,
                20.0,
                color::TEXT_QUATERNARY,
            )
        )
        .width(Length::Fixed(48.0))
        .height(Length::Fixed(48.0))
        .align_x(Alignment::Center)
        .align_y(Alignment::Center)
        .style(|_theme| iced::widget::container::Style {
            background: Some(iced::Background::Color(color::BG_INSET)),
            border: iced::Border {
                color: color::BORDER_SUBTLE,
                width: 1.0,
                radius: spacing::RADIUS_SM.into(),
            },
            ..Default::default()
        }),
        text(item.title.as_str())
            .size(typography::BODY)
            .color(color::TEXT_PRIMARY),
    ]
    .spacing(spacing::SM)
    .align_y(Alignment::Center)
    .width(Length::Fill)
    .into()
}
