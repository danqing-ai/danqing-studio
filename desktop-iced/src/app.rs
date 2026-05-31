use crate::create_page::{self, CreatePage, ModelOption};
use crate::task_queue::{TaskQueue, TaskQueueMessage};
use dq_api::ApiClient;
use dq_components::StudioIcon;
use dq_tokens::{color, spacing, typography};

#[derive(Debug, Clone)]
pub enum Message {
    Nav(NavId),
    TaskQueue(TaskQueueMessage),
    GenerateShortcut,
    Create(create_page::Message),
    DownloadImage,
    PreviewImage,
    RefreshRecent,
    EditRecent(usize),
    UpscaleRecent(usize),
    // Backend API
    ApiHealthCheck,
    ApiHealthResult(Result<serde_json::Value, String>),
    ApiModelsLoaded(Result<serde_json::Value, String>),
    ApiGenerationSubmitted(Result<serde_json::Value, String>),
    ApiSubmitWithEndpoint { endpoint: String, request: serde_json::Value },
    ApiTaskPoll(String),
    ApiTaskResult(Result<serde_json::Value, String>),
    ApiTaskStreamEvent { task_id: String, event: String, data: serde_json::Value },
    ImageSaved(Result<String, String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavId {
    ImageCreate,
    VideoCreate,
    AudioCreate,
    Gallery,
    Models,
    Settings,
}

impl NavId {
    fn id_str(self) -> &'static str {
        match self {
            NavId::ImageCreate => "image",
            NavId::VideoCreate => "video",
            NavId::AudioCreate => "audio",
            NavId::Gallery => "gallery",
            NavId::Models => "models",
            NavId::Settings => "settings",
        }
    }

    fn label(self) -> &'static str {
        match self {
            NavId::ImageCreate => "图像创作",
            NavId::VideoCreate => "视频创作",
            NavId::AudioCreate => "音频创作",
            NavId::Gallery => "作品库",
            NavId::Models => "模型",
            NavId::Settings => "设置",
        }
    }
}

pub struct App {
    pub nav: NavId,
    pub create: CreatePage,
    pub task_queue: TaskQueue,
    pub api_client: Option<ApiClient>,
    pub api_connected: bool,
}

impl App {
    pub fn new() -> (Self, iced::Task<Message>) {
        let api_client = match ApiClient::from_env() {
            Ok(client) => {
                println!("API client initialized: {}", client.base_url());
                Some(client)
            }
            Err(e) => {
                eprintln!("Failed to initialize API client: {}", e);
                None
            }
        };

        let mut app = Self {
            nav: NavId::ImageCreate,
            create: CreatePage::default(),
            task_queue: TaskQueue::new(),
            api_client,
            api_connected: false,
        };

        // Attempt health check on startup
        let init_task = if app.api_client.is_some() {
            iced::Task::perform(
                async {},
                |_| Message::ApiHealthCheck,
            )
        } else {
            iced::Task::none()
        };

        (app, init_task)
    }

    pub fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::Nav(id) => {
                self.nav = id;
                iced::Task::none()
            }
            Message::TaskQueue(msg) => {
                self.task_queue.update(msg).map(Message::TaskQueue)
            }
            Message::GenerateShortcut => {
                if self.nav == NavId::ImageCreate {
                    self.create.update(create_page::Message::Generate).map(Message::Create)
                } else {
                    iced::Task::none()
                }
            }
            Message::Create(msg) => {
                let task = self.create.update(msg.clone()).map(Message::Create);
                // Intercept Generate to call backend API
                if let create_page::Message::Generate = msg {
                    if let Some(client) = self.api_client.clone() {
                        // Determine if we need to upload a source image first
                        let needs_upload = !matches!(self.create.mode, create_page::ImageMode::TextToImage)
                            && matches!(self.create.source_image, create_page::SourceImageState::Uploaded(_));

                        if needs_upload {
                            // Upload source image first, then submit generation
                            let client_clone = client.clone();
                            let source_path = match &self.create.source_image {
                                create_page::SourceImageState::Uploaded(p) => p.clone(),
                                _ => return task,
                            };
                            // Extract request fields upfront (avoids Clone)
                            let req_mode = self.create.mode;
                            let req_model = self.create.model.id().to_string();
                            let req_title = self.create.title.clone();
                            let req_prompt = self.create.prompt.text().to_string();
                            let req_negative = self.create.negative_prompt.text().to_string();
                            let req_width = self.create.width.value();
                            let req_height = self.create.height.value();
                            let req_steps = self.create.steps;
                            let req_cfg = self.create.cfg;
                            let req_seed = self.create.seed.clone();
                            let req_scheduler = self.create.scheduler.id().to_string();
                            let req_batch_count = self.create.batch_count.value();
                            let req_denoise = self.create.denoise.value() as f32;
                            let req_ref_strength = self.create.reference_strength.value() as f32;
                            let req_upscale_factor = self.create.upscale_factor.value();
                            let req_outpaint_dir = self.create.outpaint_direction;
                            let req_outpaint_px = self.create.outpaint_pixels;
                            let upload_task = iced::Task::perform(
                                async move {
                                    let path = std::path::Path::new(&source_path);
                                    match client_clone.upload_asset(path, "image/png").await {
                                        Ok(asset) => {
                                            let asset_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            Ok(asset_id.unwrap_or_default())
                                        }
                                        Err(e) => Err(e.to_string()),
                                    }
                                },
                                move |result| match result {
                                    Ok(asset_id) => {
                                        use serde_json::json;
                                        let (endpoint, request) = match req_mode {
                                            create_page::ImageMode::ReferenceImage => {
                                                ("/api/images/edits", json!({
                                                    "model": req_model,
                                                    "operation": "rewrite",
                                                    "source_asset_id": asset_id,
                                                    "title": req_title,
                                                    "prompt": req_prompt,
                                                    "negative_prompt": req_negative,
                                                    "source_fidelity": 1.0 - req_ref_strength,
                                                    "n": req_batch_count,
                                                    "steps": req_steps as i32,
                                                    "guidance": req_cfg,
                                                    "seed": req_seed.parse::<i64>().ok(),
                                                    "scheduler": req_scheduler,
                                                    "priority": "normal",
                                                }))
                                            }
                                            create_page::ImageMode::EditByDescription => {
                                                ("/api/images/edits", json!({
                                                    "model": req_model,
                                                    "operation": "rewrite",
                                                    "source_asset_id": asset_id,
                                                    "title": req_title,
                                                    "prompt": req_prompt,
                                                    "negative_prompt": req_negative,
                                                    "source_fidelity": req_denoise,
                                                    "n": req_batch_count,
                                                    "steps": req_steps as i32,
                                                    "guidance": req_cfg,
                                                    "seed": req_seed.parse::<i64>().ok(),
                                                    "scheduler": req_scheduler,
                                                    "priority": "normal",
                                                }))
                                            }
                                            create_page::ImageMode::Inpainting => {
                                                ("/api/images/edits", json!({
                                                    "model": req_model,
                                                    "operation": "retouch",
                                                    "source_asset_id": asset_id,
                                                    "title": req_title,
                                                    "prompt": req_prompt,
                                                    "negative_prompt": req_negative,
                                                    "source_fidelity": req_denoise,
                                                    "n": req_batch_count,
                                                    "steps": req_steps as i32,
                                                    "guidance": req_cfg,
                                                    "seed": req_seed.parse::<i64>().ok(),
                                                    "scheduler": req_scheduler,
                                                    "priority": "normal",
                                                }))
                                            }
                                            create_page::ImageMode::Outpainting => {
                                                ("/api/images/edits", json!({
                                                    "model": req_model,
                                                    "operation": "extend",
                                                    "source_asset_id": asset_id,
                                                    "title": req_title,
                                                    "prompt": req_prompt,
                                                    "negative_prompt": req_negative,
                                                    "extend": {
                                                        "directions": req_outpaint_dir.ids(),
                                                        "pixels": req_outpaint_px as u32,
                                                    },
                                                    "n": req_batch_count,
                                                    "steps": req_steps as i32,
                                                    "guidance": req_cfg,
                                                    "seed": req_seed.parse::<i64>().ok(),
                                                    "scheduler": req_scheduler,
                                                    "priority": "normal",
                                                }))
                                            }
                                            create_page::ImageMode::Upscale => {
                                                ("/api/images/upscales", json!({
                                                    "model": req_model,
                                                    "source_asset_id": asset_id,
                                                    "scale": req_upscale_factor,
                                                    "denoise": req_denoise,
                                                    "tile_size": 1024,
                                                    "priority": "normal",
                                                }))
                                            }
                                            create_page::ImageMode::TextToImage => {
                                                unreachable!("text2image should not need upload")
                                            }
                                        };
                                        Message::ApiSubmitWithEndpoint {
                                            endpoint: endpoint.to_string(),
                                            request,
                                        }
                                    }
                                    Err(e) => Message::ApiGenerationSubmitted(Err(e)),
                                },
                            );
                            return iced::Task::batch([task, upload_task]);
                        } else {
                            // Direct generation (text2image)
                            let (endpoint, request) = self.create.build_request(None);
                            let endpoint = endpoint.to_string();
                            let api_task = iced::Task::perform(
                                async move {
                                    match client.post::<serde_json::Value, _>(&endpoint, &request).await {
                                        Ok(v) => Message::ApiGenerationSubmitted(Ok(v)),
                                        Err(e) => Message::ApiGenerationSubmitted(Err(e.to_string())),
                                    }
                                },
                                |msg| msg,
                            );
                            return iced::Task::batch([task, api_task]);
                        }
                    }
                }
                task
            }
            Message::DownloadImage => {
                if let Some(src_path) = self.create.generated_image_path.clone() {
                    self.create.push_log("开始保存图片…".into());
                    return iced::Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || {
                                let dest = rfd::FileDialog::new()
                                    .add_filter("PNG", &["png"])
                                    .add_filter("JPEG", &["jpg", "jpeg"])
                                    .set_file_name("danqing_result.png")
                                    .save_file()?;
                                std::fs::copy(&src_path, &dest).ok()?;
                                Some(dest.to_string_lossy().to_string())
                            }).await.ok().flatten()
                        },
                        |result| {
                            if let Some(path) = result {
                                Message::ImageSaved(Ok(path))
                            } else {
                                Message::ImageSaved(Err("保存失败或用户取消".into()))
                            }
                        },
                    );
                } else {
                    self.create.push_log("没有可下载的图片".into());
                    iced::Task::none()
                }
            }
            Message::PreviewImage => {
                self.create.push_log("打开图片预览…".into());
                iced::Task::none()
            }
            Message::RefreshRecent => {
                self.create.push_log("刷新最近生成列表".into());
                iced::Task::none()
            }
            Message::EditRecent(idx) => {
                if idx < self.create.recent_generations.len() {
                    let title = self.create.recent_generations[idx].title.clone();
                    self.create.push_log(format!("打开改图：{title}"));
                }
                iced::Task::none()
            }
            Message::UpscaleRecent(idx) => {
                if idx < self.create.recent_generations.len() {
                    let title = self.create.recent_generations[idx].title.clone();
                    self.create.push_log(format!("打开放大：{title}"));
                }
                iced::Task::none()
            }
            // Backend API handlers
            Message::ApiHealthCheck => {
                if let Some(client) = self.api_client.clone() {
                    self.create.push_log("检查后端服务状态…".into());
                    iced::Task::perform(
                        async move {
                            match client.health().await {
                                Ok(v) => Message::ApiHealthResult(Ok(v)),
                                Err(e) => Message::ApiHealthResult(Err(e.to_string())),
                            }
                        },
                        |msg| msg,
                    )
                } else {
                    self.create.push_log("API 客户端未初始化".into());
                    iced::Task::none()
                }
            }
            Message::ApiHealthResult(result) => {
                match result {
                    Ok(_) => {
                        self.api_connected = true;
                        self.create.push_log("后端服务连接成功".into());
                        // After health check, load models
                        if let Some(client) = self.api_client.clone() {
                            return iced::Task::perform(
                                async move {
                                    match client.list_models().await {
                                        Ok(v) => Message::ApiModelsLoaded(Ok(serde_json::Value::Array(v))),
                                        Err(e) => Message::ApiModelsLoaded(Err(e.to_string())),
                                    }
                                },
                                |msg| msg,
                            );
                        }
                    }
                    Err(e) => {
                        self.api_connected = false;
                        self.create.push_log(format!("后端服务连接失败: {}", e));
                    }
                }
                iced::Task::none()
            }
            Message::ApiModelsLoaded(result) => {
                match result {
                    Ok(models) => {
                        let count = models.as_array().map_or(0, |a| a.len());
                        self.create.push_log(format!("已加载 {} 个模型", count));
                        // Parse dynamic models from backend
                        self.create.available_models.clear();
                        if let Some(arr) = models.as_array() {
                            for m in arr {
                                if let (Some(id), Some(name)) = (
                                    m.get("id").and_then(|v| v.as_str()),
                                    m.get("name").and_then(|v| v.as_str()),
                                ) {
                                    let steps = m.get("default_steps").and_then(|v| v.as_u64()).unwrap_or(20) as u8;
                                    let cfg = m.get("default_cfg").and_then(|v| v.as_f64()).unwrap_or(5.0) as f32;
                                    self.create.available_models.push(
                                        crate::create_page::ModelOption::Dynamic {
                                            id: id.to_string(),
                                            label: format!("{} · {}", name, id),
                                            steps,
                                            cfg,
                                        }
                                    );
                                }
                            }
                        }
                        // If backend returned models, select the first one
                        if let Some(first) = self.create.available_models.first().cloned() {
                            self.create.model = first.clone();
                            self.create.steps = first.default_steps();
                            self.create.cfg = first.default_cfg();
                            self.create.steps_input = format!("{:.0}", self.create.steps);
                            self.create.cfg_input = format!("{:.1}", self.create.cfg);
                        }
                    }
                    Err(e) => {
                        self.create.push_log(format!("加载模型列表失败: {}", e));
                    }
                }
                iced::Task::none()
            }
            Message::ApiSubmitWithEndpoint { endpoint, request } => {
                if let Some(client) = self.api_client.clone() {
                    self.create.push_log(format!("提交请求到 {}…", endpoint));
                    return iced::Task::perform(
                        async move {
                            match client.post::<serde_json::Value, _>(&endpoint, &request).await {
                                Ok(v) => Message::ApiGenerationSubmitted(Ok(v)),
                                Err(e) => Message::ApiGenerationSubmitted(Err(e.to_string())),
                            }
                        },
                        |msg| msg,
                    );
                }
                iced::Task::none()
            }
            Message::ApiGenerationSubmitted(result) => {
                match result {
                    Ok(response) => {
                        // Try nested task.id first, then legacy flat id
                        let task_id = response.get("task")
                            .and_then(|t| t.get("id"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| response.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()));

                        if let Some(tid) = task_id {
                            self.create.push_log(format!("任务已提交: {}", tid));
                            // Start SSE stream
                            let client = self.api_client.clone().unwrap();
                            let tid_clone = tid.clone();
                            return iced::Task::perform(
                                async move {
                                    let mut last_progress = 0u8;
                                    let mut last_status = String::new();
                                    let result = client.stream_task_events(&tid_clone, |event, data| {
                                        match event {
                                            "progress" => {
                                                if let Some(p) = data.get("progress").and_then(|v| v.as_f64()) {
                                                    last_progress = (p * 100.0) as u8;
                                                }
                                            }
                                            "status" => {
                                                if let Some(s) = data.get("status").and_then(|v| v.as_str()) {
                                                    last_status = s.to_string();
                                                }
                                            }
                                            "done" => {
                                                last_status = "completed".to_string();
                                            }
                                            "log" => {
                                                // Logs are ingested separately
                                            }
                                            _ => {}
                                        }
                                    }).await;
                                    (tid_clone, last_status, last_progress, result)
                                },
                                |(tid, status, progress, _result)| {
                                    if status == "completed" {
                                        Message::ApiTaskPoll(tid)
                                    } else {
                                        Message::ApiTaskResult(Ok(serde_json::json!({
                                            "task_id": tid,
                                            "status": status,
                                            "progress": progress,
                                        })))
                                    }
                                },
                            );
                        } else {
                            self.create.push_log("提交成功但无法获取任务 ID".into());
                            self.create.generate_state = crate::create_page::GenerateState::Idle;
                        }
                    }
                    Err(e) => {
                        self.create.push_log(format!("提交生成任务失败: {}", e));
                        self.create.generate_state = crate::create_page::GenerateState::Idle;
                    }
                }
                iced::Task::none()
            }
            Message::ApiTaskPoll(task_id) => {
                if let Some(client) = self.api_client.clone() {
                    let task_id_clone = task_id.clone();
                    return iced::Task::perform(
                        async move {
                            let path = format!("/api/tasks/{}", task_id_clone);
                            match client.get::<serde_json::Value>(&path).await {
                                Ok(v) => Message::ApiTaskResult(Ok(v)),
                                Err(e) => Message::ApiTaskResult(Err(e.to_string())),
                            }
                        },
                        |msg| msg,
                    );
                }
                iced::Task::none()
            }
            Message::ApiTaskResult(result) => {
                match result {
                    Ok(task_info) => {
                        let status = task_info.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
                        match status {
                            "completed" => {
                                self.create.generate_state = crate::create_page::GenerateState::Done;
                                self.create.push_recent();
                                self.create.push_log("生成完成".into());

                                // Extract primary_asset_id from result
                                let primary_asset_id = task_info.get("result")
                                    .and_then(|r| r.get("primary_asset_id"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());

                                if let Some(client) = self.api_client.clone() {
                                    if let Some(asset_id) = primary_asset_id {
                                        let download_url = client.asset_file_url(&asset_id);
                                        self.create.push_log(format!("下载结果: {}", download_url));
                                        return iced::Task::perform(
                                            async move {
                                                let temp_path = std::env::temp_dir().join(format!("danqing_result_{}.png", chrono::Local::now().timestamp_millis()));
                                                match client.download_file(&download_url, &temp_path).await {
                                                    Ok(()) => {
                                                        let path_str = temp_path.to_string_lossy().to_string();
                                                        Message::Create(create_page::Message::GenerateComplete { result_urls: vec![path_str] })
                                                    }
                                                    Err(e) => {
                                                        Message::Create(create_page::Message::GenerateFailed { error: format!("下载结果图片失败: {}", e) })
                                                    }
                                                }
                                            },
                                            |msg| msg,
                                        );
                                    }
                                }
                            }
                            "failed" => {
                                self.create.generate_state = crate::create_page::GenerateState::Idle;
                                let error = task_info.get("error")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(task_info.get("error_message")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("未知错误"));
                                self.create.push_log(format!("生成失败: {}", error));
                            }
                            _ => {
                                // Still running, poll again after delay
                                let progress = task_info.get("progress")
                                    .and_then(|v| v.as_f64())
                                    .map(|p| (p * 100.0) as u8)
                                    .unwrap_or(0);
                                let phase_str = task_info.get("phase")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("queued");
                                let step = task_info.get("step")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0) as u32;
                                let total = task_info.get("total")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(1) as u32;
                                self.create.generate_state = crate::create_page::GenerateState::Generating {
                                    progress,
                                    step,
                                    total,
                                    phase: crate::create_page::GeneratePhase::from_str(phase_str),
                                };
                                if let Some(task_id) = task_info.get("id")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .or_else(|| task_info.get("task_id").and_then(|v| v.as_str()).map(|s| s.to_string())) {
                                    return iced::Task::perform(
                                        async { tokio::time::sleep(std::time::Duration::from_secs(2)).await },
                                        move |_| Message::ApiTaskPoll(task_id),
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        self.create.push_log(format!("查询任务状态失败: {}", e));
                    }
                }
                iced::Task::none()
            }
            Message::ApiTaskStreamEvent { task_id, event, data } => {
                // SSE stream events are handled inline during streaming;
                // this variant exists for future async event routing.
                let _ = (task_id, event, data);
                iced::Task::none()
            }
            Message::ImageSaved(result) => {
                match result {
                    Ok(path) => self.create.push_log(format!("图片已保存: {}", path)),
                    Err(e) => self.create.push_log(format!("保存图片失败: {}", e)),
                }
                iced::Task::none()
            }
        }
    }

    pub fn subscription(&self) -> iced::Subscription<Message> {
        use iced::keyboard::{self, key::Named};
        use iced::time;
        use iced::window;

        let keyboard_sub = keyboard::listen().filter_map(|event| {
            if let keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(Named::Enter),
                modifiers,
                repeat,
                ..
            } = event
            {
                if !repeat && modifiers.command() {
                    return Some(Message::GenerateShortcut);
                }
            }
            None
        });

        let memory_poll = time::every(std::time::Duration::from_secs(5))
            .map(|_| Message::Create(create_page::Message::MemoryPoll));

        let file_drop = iced::event::listen().filter_map(|event| {
            if let iced::Event::Window(iced::window::Event::FileDropped(path)) = event {
                return Some(Message::Create(create_page::Message::SourceImageDropped(path.to_string_lossy().to_string())));
            }
            None
        });

        iced::Subscription::batch(vec![keyboard_sub, memory_poll, file_drop])
    }

    pub fn view(&self) -> iced::Element<'_, Message> {
        use dq_layout::{dq_sidebar, dq_studio_nav};
        use dq_layout::{NavItem, SidebarSection};
        use dq_theme::{page_container, subtle_scrollbar, vertical_divider};
        use iced::widget::{column, container, row, scrollable, text};
        use iced::{Alignment, Element, Length};
        use iced::widget::scrollable::{Direction, Scrollbar};

        let active_id = self.nav.id_str();

        // Left sidebar — icon-only with logo + task queue + settings at bottom
        let sidebar = dq_sidebar(
            vec![
                SidebarSection {
                    label: None,
                    items: vec![
                        NavItem {
                            id: "image".into(),
                            icon: StudioIcon::Image,
                            label: "图像创作".into(),
                            message: Message::Nav(NavId::ImageCreate),
                        },
                        NavItem {
                            id: "video".into(),
                            icon: StudioIcon::Video,
                            label: "视频创作".into(),
                            message: Message::Nav(NavId::VideoCreate),
                        },
                        NavItem {
                            id: "audio".into(),
                            icon: StudioIcon::Audio,
                            label: "音频创作".into(),
                            message: Message::Nav(NavId::AudioCreate),
                        },
                    ],
                },
                SidebarSection {
                    label: Some("资料".into()),
                    items: vec![
                        NavItem {
                            id: "gallery".into(),
                            icon: StudioIcon::Gallery,
                            label: "作品库".into(),
                            message: Message::Nav(NavId::Gallery),
                        },
                        NavItem {
                            id: "models".into(),
                            icon: StudioIcon::Models,
                            label: "模型库".into(),
                            message: Message::Nav(NavId::Models),
                        },
                    ],
                },
            ],
            active_id,
            Message::TaskQueue(TaskQueueMessage::ToggleWindow),
            Message::Nav(NavId::Settings),
        );

        // Main content area — ImageCreate has mode tabs at top, no page title
        let main_content: Element<Message> = match self.nav {
            NavId::ImageCreate => {
                let (tabs, left_panel) = self.create.workspace_view();

                // Right panel based on current mode
                let preview_path = match self.create.generate_state {
                    crate::create_page::GenerateState::Done => {
                        self.create.generated_image_path.as_ref().map(std::path::PathBuf::from)
                    }
                    _ => {
                        match &self.create.source_image {
                            crate::create_page::SourceImageState::Uploaded(path) => Some(std::path::PathBuf::from(path)),
                            _ => None,
                        }
                    }
                };

                let right_panel = crate::right_panel::right_panel(
                    self.create.generate_state,
                    self.create.width(),
                    self.create.height(),
                    self.create.model.id(),
                    &self.create.seed,
                    matches!(self.create.model, ModelOption::ZImageTurbo),
                    &self.create.recent_generations,
                    &self.create.staged_results,
                    &self.create.memory_info,
                    self.create.enhance_offer_visible,
                    "用 flux1-dev 精修增强",
                    preview_path,
                    Some(Message::DownloadImage),
                    Some(Message::PreviewImage),
                    Message::RefreshRecent,
                    Some(Message::Create(create_page::Message::StartEnhance)),
                    |msg| Message::Create(create_page::Message::StagingMsg(msg)),
                );

                let workspace = row![
                    container(left_panel.map(Message::Create))
                        .width(Length::FillPortion(60))
                        .padding([spacing::SM, spacing::MD]),
                    container(iced::widget::Space::new())
                        .width(Length::Fixed(1.0))
                        .height(Length::Fill)
                        .style(vertical_divider),
                    container(right_panel)
                        .width(Length::FillPortion(40))
                        .padding([spacing::SM, spacing::MD])
                        .height(Length::Fill),
                ]
                .width(Length::Fill)
                .height(Length::Fill);

                column![
                    // Fixed mode tabs at top
                    tabs.map(Message::Create),
                    // Scrollable content below
                    scrollable(workspace)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .direction(Direction::Vertical(Scrollbar::new().width(3).scroller_width(3)))
                        .style(subtle_scrollbar),
                ]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
            }
            _ => {
                // Other pages still have a title bar
                let page_title = text(self.nav.label())
                    .size(typography::HEADING)
                    .color(color::TEXT_PRIMARY);
                let top_bar = dq_studio_nav(page_title.into());
                column![
                    top_bar,
                    container(placeholder_page(self.nav)).width(Length::Fill).height(Length::Fill),
                ]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
            }
        };

        let body: Element<Message> = if self.task_queue.show_window {
            row![
                main_content,
                container(self.task_queue.view().map(Message::TaskQueue))
                    .height(Length::Fill),
            ]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
        } else {
            main_content
        };

        container(
            row![
                sidebar,
                container(iced::widget::Space::new())
                    .width(Length::Fixed(1.0))
                    .height(Length::Fill)
                    .style(vertical_divider),
                body,
            ]
            .width(Length::Fill)
            .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .style(page_container)
        .into()
    }
}

fn placeholder_page(nav: NavId) -> iced::Element<'static, Message> {
    use iced::widget::{container, text};
    let label = match nav {
        NavId::VideoCreate => "视频创作 — 即将推出",
        NavId::AudioCreate => "音频创作 — 即将推出",
        NavId::Gallery => "作品库 — 即将推出",
        NavId::Models => "模型管理 — 即将推出",
        NavId::Settings => "设置 — 即将推出",
        NavId::ImageCreate => unreachable!(),
    };
    container(text(label).color(color::TEXT_SECONDARY))
        .padding(32)
        .into()
}
