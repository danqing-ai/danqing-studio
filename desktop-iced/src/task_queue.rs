use dq_components::{dq_progress_bar, phosphor_icon, PhosphorIcon};
use dq_tokens::{color, spacing, typography};
use iced::widget::{button, column, container, row, scrollable, text, Space};
use iced::{Alignment, Color, Element, Length};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending,
    Queued { position: usize, eta_seconds: u32 },
    Running { step: u32, total: u32, phase: String },
    Completed,
    Failed { error: String },
    Cancelled,
}

impl TaskStatus {
    fn label(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "等待中",
            TaskStatus::Queued { .. } => "队列中",
            TaskStatus::Running { .. } => "运行中",
            TaskStatus::Completed => "已完成",
            TaskStatus::Failed { .. } => "失败",
            TaskStatus::Cancelled => "已取消",
        }
    }

    fn progress(&self) -> f32 {
        match self {
            TaskStatus::Pending => 0.0,
            TaskStatus::Queued { .. } => 0.05,
            TaskStatus::Running { step, total, .. } => {
                if *total > 0 {
                    (*step as f32 / *total as f32).clamp(0.0, 0.95)
                } else {
                    0.5
                }
            }
            TaskStatus::Completed => 1.0,
            TaskStatus::Failed { .. } => 1.0,
            TaskStatus::Cancelled => 1.0,
        }
    }

    fn status_color(&self) -> iced::Color {
        match self {
            TaskStatus::Pending | TaskStatus::Queued { .. } => color::ACCENT,
            TaskStatus::Running { .. } => color::WARNING,
            TaskStatus::Completed => color::SUCCESS,
            TaskStatus::Failed { .. } => color::DANGER,
            TaskStatus::Cancelled => color::TEXT_TERTIARY,
        }
    }

    fn is_active(&self) -> bool {
        matches!(self, TaskStatus::Running { .. } | TaskStatus::Queued { .. } | TaskStatus::Pending)
    }
}

#[derive(Debug, Clone)]
pub struct TaskItem {
    pub id: String,
    pub title: String,
    pub mode: String,
    pub model: String,
    pub status: TaskStatus,
    pub created_at: Instant,
    pub progress: f32,
}

impl TaskItem {
    pub fn duration_text(&self) -> String {
        let elapsed = self.created_at.elapsed();
        let secs = elapsed.as_secs();
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TaskQueue {
    pub tasks: Vec<TaskItem>,
    pub show_window: bool,
}

#[derive(Debug, Clone)]
pub enum TaskQueueMessage {
    ToggleWindow,
    CloseWindow,
    CancelTask(String),
    ClearCompleted,
    Refresh,
}

impl TaskQueue {
    pub fn new() -> Self {
        Self {
            tasks: vec![
                TaskItem {
                    id: "task-001".into(),
                    title: "日系肖像".into(),
                    mode: "文生图".into(),
                    model: "z-image-turbo".into(),
                    status: TaskStatus::Running {
                        step: 12,
                        total: 20,
                        phase: "denoising".into(),
                    },
                    created_at: Instant::now() - Duration::from_secs(45),
                    progress: 0.6,
                },
                TaskItem {
                    id: "task-002".into(),
                    title: "赛博朋克城市".into(),
                    mode: "文生图".into(),
                    model: "flux1-dev".into(),
                    status: TaskStatus::Queued {
                        position: 1,
                        eta_seconds: 120,
                    },
                    created_at: Instant::now() - Duration::from_secs(10),
                    progress: 0.05,
                },
                TaskItem {
                    id: "task-003".into(),
                    title: "精修放大".into(),
                    mode: "精修放大".into(),
                    model: "z-image-turbo".into(),
                    status: TaskStatus::Completed,
                    created_at: Instant::now() - Duration::from_secs(300),
                    progress: 1.0,
                },
                TaskItem {
                    id: "task-004".into(),
                    title: "局部修饰".into(),
                    mode: "局部修饰".into(),
                    model: "flux2-klein".into(),
                    status: TaskStatus::Failed {
                        error: "内存不足".into(),
                    },
                    created_at: Instant::now() - Duration::from_secs(600),
                    progress: 0.0,
                },
            ],
            show_window: false,
        }
    }

    pub fn update(&mut self, message: TaskQueueMessage) -> iced::Task<TaskQueueMessage> {
        match message {
            TaskQueueMessage::ToggleWindow => {
                self.show_window = !self.show_window;
                iced::Task::none()
            }
            TaskQueueMessage::CloseWindow => {
                self.show_window = false;
                iced::Task::none()
            }
            TaskQueueMessage::CancelTask(id) => {
                if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
                    task.status = TaskStatus::Cancelled;
                    task.progress = 1.0;
                }
                iced::Task::none()
            }
            TaskQueueMessage::ClearCompleted => {
                self.tasks.retain(|t| !matches!(t.status, TaskStatus::Completed | TaskStatus::Cancelled));
                iced::Task::none()
            }
            TaskQueueMessage::Refresh => {
                iced::Task::none()
            }
        }
    }

    pub fn running_count(&self) -> usize {
        self.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Running { .. })).count()
    }

    pub fn queued_count(&self) -> usize {
        self.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Queued { .. } | TaskStatus::Pending)).count()
    }

    pub fn completed_count(&self) -> usize {
        self.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Completed)).count()
    }

    /// Side panel view — inspired by Linear/Figma task lists.
    pub fn view(&self) -> Element<'_ , TaskQueueMessage> {
        let header = row![
            text("任务队列")
                .size(typography::BODY)
                .color(color::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            icon_button(PhosphorIcon::X, 12.0, Some(TaskQueueMessage::CloseWindow)),
        ]
        .spacing(spacing::SM)
        .align_y(Alignment::Center)
        .width(Length::Fill);

        // Filter tabs
        let tabs = row![
            filter_tab("运行中", self.running_count(), true),
            filter_tab("排队中", self.queued_count(), false),
            filter_tab("已完成", self.completed_count(), false),
        ]
        .spacing(spacing::XS)
        .align_y(Alignment::Center)
        .width(Length::Fill);

        let task_list: Element<'_ , TaskQueueMessage> = if self.tasks.is_empty() {
            container(
                column![
                    phosphor_icon(PhosphorIcon::Queue, 32.0, color::TEXT_TERTIARY),
                    text("暂无任务")
                        .size(typography::BODY)
                        .color(color::TEXT_SECONDARY),
                    text("提交生成任务后将显示在这里")
                        .size(typography::CAPTION)
                        .color(color::TEXT_TERTIARY),
                ]
                .spacing(spacing::SM)
                .align_x(Alignment::Center),
            )
            .width(Length::Fill)
            .padding(spacing::XL)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into()
        } else {
            let mut list = column![].spacing(spacing::XS).width(Length::Fill);
            for task in &self.tasks {
                list = list.push(task_card(task));
            }
            scrollable(list)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        };

        let footer = row![
            Space::new().width(Length::Fill),
            icon_text_button(PhosphorIcon::Trash, "清除已完成", Some(TaskQueueMessage::ClearCompleted)),
            icon_text_button(PhosphorIcon::ArrowsClockwise, "刷新", Some(TaskQueueMessage::Refresh)),
        ]
        .spacing(spacing::SM)
        .align_y(Alignment::Center)
        .width(Length::Fill);

        let panel = column![
            header,
            tabs,
            container(Space::new())
                .width(Length::Fill)
                .height(Length::Fixed(1.0))
                .style(|_theme| container::Style {
                    background: Some(iced::Background::Color(color::SEPARATOR)),
                    ..Default::default()
                }),
            task_list,
            footer,
        ]
        .spacing(spacing::MD)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(spacing::LG);

        // Side panel container — fixed width, dark background
        container(panel)
            .width(Length::Fixed(360.0))
            .height(Length::Fill)
            .style(|_theme| container::Style {
                background: Some(iced::Background::Color(color::BG_SURFACE)),
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

// ─── Helper Components ───────────────────────────────────────────────────────

fn icon_button<'a, Message: Clone + 'a>(
    icon: PhosphorIcon,
    size: f32,
    on_press: Option<Message>,
) -> Element<'a, Message> {
    button(
        container(phosphor_icon(icon, size, color::TEXT_SECONDARY))
            .width(Length::Fixed(24.0))
            .height(Length::Fixed(24.0))
            .align_x(Alignment::Center)
            .align_y(Alignment::Center),
    )
    .padding(0)
    .style(|_theme, _status| button::Style {
        background: None,
        ..Default::default()
    })
    .on_press_maybe(on_press)
    .into()
}

fn icon_text_button<'a, Message: Clone + 'a>(
    icon: PhosphorIcon,
    label: &'a str,
    on_press: Option<Message>,
) -> Element<'a, Message> {
    button(
        row![
            phosphor_icon(icon, 12.0, color::TEXT_TERTIARY),
            text(label)
                .size(typography::CAPTION)
                .color(color::TEXT_TERTIARY),
        ]
        .spacing(4.0)
        .align_y(Alignment::Center),
    )
    .padding([4.0, 8.0])
    .style(|_theme, status| {
        let base = button::Style {
            background: None,
            text_color: color::TEXT_SECONDARY,
            ..Default::default()
        };
        match status {
            button::Status::Hovered => button::Style {
                background: Some(iced::Background::Color(color::FILL_HOVER)),
                border: iced::Border {
                    color: color::BORDER_SUBTLE,
                    width: 1.0,
                    radius: spacing::RADIUS_SM.into(),
                },
                ..base
            },
            _ => base,
        }
    })
    .on_press_maybe(on_press)
    .into()
}

fn filter_tab<'a>(label: &'a str, count: usize, active: bool) -> Element<'a, TaskQueueMessage> {
    let count_text = if count > 0 { format!(" {}", count) } else { String::new() };
    container(
        text(format!("{}{}", label, count_text))
            .size(typography::CAPTION)
            .color(if active { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY }),
    )
    .padding([4.0, 10.0])
    .style(move |_theme| container::Style {
        background: if active {
            Some(iced::Background::Color(color::FILL_SELECTED))
        } else {
            None
        },
        border: iced::Border {
            color: if active { color::BORDER_SUBTLE } else { Color::TRANSPARENT },
            width: 1.0,
            radius: spacing::RADIUS_SM.into(),
        },
        ..Default::default()
    })
    .into()
}

fn status_dot<'a>(status: &'a TaskStatus) -> Element<'a, TaskQueueMessage> {
    container(Space::new())
        .width(Length::Fixed(6.0))
        .height(Length::Fixed(6.0))
        .style(move |_theme| container::Style {
            background: Some(iced::Background::Color(status.status_color())),
            border: iced::Border {
                color: Color::TRANSPARENT,
                width: 0.0,
                radius: 999.0.into(),
            },
            ..Default::default()
        })
        .into()
}

fn task_card<'a>(task: &'a TaskItem) -> Element<'a, TaskQueueMessage> {
    let status_text = match &task.status {
        TaskStatus::Queued { position, eta_seconds } => {
            format!("#{} · 预计{}s", position, eta_seconds)
        }
        TaskStatus::Running { step, total, phase } => {
            if *total > 0 {
                format!("{} · {}/{}", phase, step, total)
            } else {
                phase.clone()
            }
        }
        TaskStatus::Failed { error } => format!("错误: {}", error),
        _ => task.status.label().to_string(),
    };

    let cancel_btn = if task.status.is_active() {
        Some(icon_button(PhosphorIcon::X, 10.0, Some(TaskQueueMessage::CancelTask(task.id.clone()))))
    } else {
        None
    };

    let card_content = column![
        row![
            row![
                status_dot(&task.status),
                text(&task.title)
                    .size(typography::BODY)
                    .color(color::TEXT_PRIMARY),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center),
            Space::new().width(Length::Fill),
            cancel_btn,
        ]
        .spacing(spacing::SM)
        .align_y(Alignment::Center)
        .width(Length::Fill),
        row![
            text(format!("{} · {}", task.mode, task.model))
                .size(typography::CAPTION)
                .color(color::TEXT_TERTIARY),
            Space::new().width(Length::Fill),
            text(task.duration_text())
                .size(typography::CAPTION)
                .color(color::TEXT_QUATERNARY),
        ]
        .spacing(spacing::SM)
        .align_y(Alignment::Center)
        .width(Length::Fill),
        if task.status.is_active() {
            Some(dq_progress_bar(task.status.progress(), 3.0))
        } else {
            None
        },
    ]
    .spacing(spacing::XS)
    .width(Length::Fill);

    container(card_content)
        .width(Length::Fill)
        .padding(spacing::SM)
        .style(|_theme| container::Style {
            background: Some(iced::Background::Color(color::BG_INSET)),
            border: iced::Border {
                color: color::BORDER_SUBTLE,
                width: 1.0,
                radius: spacing::RADIUS_SM.into(),
            },
            ..Default::default()
        })
        .into()
}
