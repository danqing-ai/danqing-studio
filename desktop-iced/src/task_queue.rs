#![allow(dead_code)]
use dq_components::{
    dq_progress_bar, section_card, tag, TagType,
    phosphor_icon, PhosphorIcon,
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{button, column, container, row, scrollable, text, Space};
use iced::{Alignment, Element, Length};
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending,
    Queued { position: usize, eta_seconds: u32 },
    Running { step: u32, total: u32, phase: String },
    Completed, Failed { error: String }, Cancelled,
}

impl TaskStatus {
    fn label(&self) -> &'static str { match self { TaskStatus::Pending => "等待中", TaskStatus::Queued { .. } => "队列中", TaskStatus::Running { .. } => "运行中", TaskStatus::Completed => "已完成", TaskStatus::Failed { .. } => "失败", TaskStatus::Cancelled => "已取消" } }
    fn progress(&self) -> f32 { match self { TaskStatus::Pending => 0.0, TaskStatus::Queued { .. } => 0.05, TaskStatus::Running { step, total, .. } => if *total > 0 { (*step as f32 / *total as f32).clamp(0.0, 0.95) } else { 0.5 }, _ => 1.0 } }
    fn tag_type(&self) -> TagType { match self { TaskStatus::Pending | TaskStatus::Queued { .. } => TagType::Info, TaskStatus::Running { .. } => TagType::Warning, TaskStatus::Completed => TagType::Success, TaskStatus::Failed { .. } => TagType::Danger, TaskStatus::Cancelled => TagType::Default } }
    fn is_active(&self) -> bool { matches!(self, TaskStatus::Running { .. } | TaskStatus::Queued { .. } | TaskStatus::Pending) }
}

#[derive(Debug, Clone)]
pub struct TaskItem {
    pub id: String, pub title: String, pub mode: String,
    pub model: String, pub status: TaskStatus, pub created_at: Instant, pub progress: f32,
}

impl TaskItem {
    pub fn duration_text(&self) -> String {
        let secs = self.created_at.elapsed().as_secs();
        if secs < 60 { format!("{}s", secs) } else if secs < 3600 { format!("{}m {}s", secs / 60, secs % 60) } else { format!("{}h {}m", secs / 3600, (secs % 3600) / 60) }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TaskQueue {
    pub tasks: Vec<TaskItem>, pub show_window: bool,
}

#[derive(Debug, Clone)]
pub enum TaskQueueMessage {
    ToggleWindow, CloseWindow, CancelTask(String), ClearCompleted, Refresh, SetTasks(Vec<TaskItem>),
}

impl TaskQueue {
    pub fn new() -> Self { Self { tasks: vec![], show_window: false } }

    pub fn update(&mut self, message: TaskQueueMessage) -> iced::Task<TaskQueueMessage> {
        match message {
            TaskQueueMessage::ToggleWindow => { self.show_window = !self.show_window; iced::Task::none() }
            TaskQueueMessage::CloseWindow => { self.show_window = false; iced::Task::none() }
            TaskQueueMessage::CancelTask(id) => { if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) { task.status = TaskStatus::Cancelled; task.progress = 1.0; } iced::Task::none() }
            TaskQueueMessage::ClearCompleted => { self.tasks.retain(|t| !matches!(t.status, TaskStatus::Completed | TaskStatus::Cancelled)); iced::Task::none() }
            TaskQueueMessage::Refresh => iced::Task::none(),
            TaskQueueMessage::SetTasks(tasks) => { self.tasks = tasks; iced::Task::none() }
        }
    }

    pub fn running_count(&self) -> usize { self.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Running { .. })).count() }
    pub fn queued_count(&self) -> usize { self.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Queued { .. } | TaskStatus::Pending)).count() }
    pub fn completed_count(&self) -> usize { self.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Completed)).count() }

    pub fn view(&self) -> Element<'_, TaskQueueMessage> {
        let header = row![
            text("任务队列").size(typography::TITLE).color(color::TEXT_PRIMARY),
            Space::new().width(Length::Fill),
            button(phosphor_icon(PhosphorIcon::X, 12.0, color::TEXT_SECONDARY))
                .on_press(TaskQueueMessage::CloseWindow)
                .style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill);

        // Status count pills
        let status_row = row![
            container(text(format!("运行中 {}", self.running_count())).size(typography::LABEL).color(if self.running_count() > 0 { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY })).padding([4.0, 8.0]).style(|_theme: &iced::Theme| container::Style { background: if self.running_count() > 0 { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None }, border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }),
            container(text(format!("排队中 {}", self.queued_count())).size(typography::LABEL).color(if self.queued_count() > 0 { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY })).padding([4.0, 8.0]).style(|_theme: &iced::Theme| container::Style { background: if self.queued_count() > 0 { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None }, border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }),
            container(text(format!("已完成 {}", self.completed_count())).size(typography::LABEL).color(if self.completed_count() > 0 { color::TEXT_PRIMARY } else { color::TEXT_TERTIARY })).padding([4.0, 8.0]).style(|_theme: &iced::Theme| container::Style { background: if self.completed_count() > 0 { Some(iced::Background::Color(color::FILL_SELECTED)) } else { None }, border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_SM.into() }, ..Default::default() }),
        ].spacing(spacing::XS);

        let task_list: Element<'_, TaskQueueMessage> = if self.tasks.is_empty() {
            container(column![
                phosphor_icon(PhosphorIcon::Queue, 32.0, color::TEXT_TERTIARY),
                text("暂无任务").size(typography::BODY).color(color::TEXT_SECONDARY),
                text("提交生成任务后将显示在这里").size(typography::CAPTION).color(color::TEXT_TERTIARY),
            ].spacing(spacing::SM).align_x(Alignment::Center)).width(Length::Fill).padding(spacing::XL).center_x(Length::Fill).center_y(Length::Fill).into()
        } else {
            let mut list = column![].spacing(spacing::SM).width(Length::Fill);
            for task in &self.tasks {
                let status_text = match &task.status {
                    TaskStatus::Queued { position, eta_seconds } => format!("#{} · 预计{}s", position, eta_seconds),
                    TaskStatus::Running { step, total, phase } => if *total > 0 { format!("{} · {}/{}", phase, step, total) } else { phase.clone() },
                    TaskStatus::Failed { error } => format!("错误: {}", error),
                    _ => task.status.label().to_string(),
                };
                let cancel_btn: Option<Element<TaskQueueMessage>> = if task.status.is_active() {
                    Some(button(phosphor_icon(PhosphorIcon::X, 10.0, color::TEXT_TERTIARY))
                        .on_press(TaskQueueMessage::CancelTask(task.id.clone()))
                        .style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }).into())
                } else { None };

                let card = section_card(&task.title, column![
                    row![
                        tag(task.status.label(), task.status.tag_type()),
                        Space::new().width(Length::Fill),
                        if let Some(cb) = cancel_btn { cb } else { Space::new().width(0).into() },
                    ].spacing(spacing::SM).width(Length::Fill),
                    text(status_text).size(typography::CAPTION).color(color::TEXT_TERTIARY),
                    if task.status.is_active() { dq_progress_bar(task.status.progress(), 3.0) } else { Space::new().height(3).into() },
                ].spacing(spacing::XS).into());
                list = list.push(card);
            }
            scrollable(list).width(Length::Fill).height(Length::Fill).into()
        };

        let footer = row![
            Space::new().width(Length::Fill),
            button(text("清除已完成").size(typography::CAPTION).color(color::TEXT_TERTIARY))
                .on_press(TaskQueueMessage::ClearCompleted)
                .style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
            button(text("刷新").size(typography::CAPTION).color(color::TEXT_TERTIARY))
                .on_press(TaskQueueMessage::Refresh)
                .style(|_theme: &iced::Theme, _status| button::Style { background: None, ..Default::default() }),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill);

        let panel = column![header, status_row, container(Space::new()).width(Length::Fill).height(Length::Fixed(1.0)).style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::SEPARATOR)), ..Default::default() }), task_list, footer]
            .spacing(spacing::MD).width(Length::Fill).height(Length::Fill).padding(spacing::LG);

        container(panel)
            .width(Length::Fixed(360.0)).height(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style { background: Some(iced::Background::Color(color::BG_SURFACE)), border: iced::Border { color: color::BORDER_SUBTLE, width: 1.0, radius: spacing::RADIUS_MD.into() }, ..Default::default() })
            .into()
    }
}
