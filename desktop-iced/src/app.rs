// Allow dead code for future feature variants and API message types
#![allow(dead_code)]

use crate::create_page::{self, CreatePage, ModelOption};
use crate::gallery;
use crate::task_queue::{TaskQueue, TaskQueueMessage};
use crate::video_page::{self, VideoPage};

use crate::audio_page::{self, AudioCreatePage};
use crate::models_page::{self, ModelsPage};
use crate::settings_page::{self, SettingsPage};
use dq_api::ApiClient;
use dq_components::StudioIcon;
use dq_tokens::{spacing, typography};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeId {
    LinearDark,
    LinearLight,
    ChinaRedDark,
}

impl ThemeId {
    pub fn label(self) -> &'static str {
        match self {
            ThemeId::LinearDark => "Linear Dark",
            ThemeId::LinearLight => "Linear Light",
            ThemeId::ChinaRedDark => "中国红 · China Red",
        }
    }

    pub fn to_iced_theme(self) -> iced::Theme {
        match self {
            ThemeId::LinearDark => dq_theme::linear_theme(),
            ThemeId::LinearLight => dq_theme::linear_light_theme(),
            ThemeId::ChinaRedDark => dq_theme::china_red_dark_theme(),
        }
    }

    pub const ALL: &'static [ThemeId] = &[
        ThemeId::LinearDark,
        ThemeId::LinearLight,
        ThemeId::ChinaRedDark,
    ];
}

#[derive(Debug, Clone)]
pub enum Message {
    Nav(NavId),
    NavAndRefresh(NavId),
    ThemeChanged(ThemeId),
    TaskQueue(TaskQueueMessage),
    GenerateShortcut,
    Create(create_page::Message),
    Video(video_page::Message),
    Audio(audio_page::Message),
    Gallery(gallery::Message),
    Models(models_page::Message),
    Settings(settings_page::Message),
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
    ApiAudioSubmitted(Result<serde_json::Value, String>),
    ApiSubmitWithEndpoint { endpoint: String, request: serde_json::Value },
    ApiTaskPoll(String),
    ApiTaskResult(Result<serde_json::Value, String>),
    ApiTaskStreamEvent { task_id: String, event: String, data: serde_json::Value },
    ApiAssetsReady {
        source_asset_id: Option<String>,
        mask_asset_id: Option<String>,
        control_asset_id: Option<String>,
    },
    ApiTaskListPoll,
    ApiTaskListResult(Result<Vec<serde_json::Value>, String>),
    ImageSaved(Result<String, String>),
    KeyEvent(iced::keyboard::Key),
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
    pub video: VideoPage,
    pub audio: AudioCreatePage,
    pub gallery: gallery::GalleryPage,
    pub models: ModelsPage,
    pub settings: SettingsPage,
    pub task_queue: TaskQueue,
    pub api_client: Option<ApiClient>,
    pub api_connected: bool,
    pub current_task_id: Option<String>,
    pub current_theme: ThemeId,
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

        let app = Self {
            nav: NavId::ImageCreate,
            create: CreatePage::default(),
            video: VideoPage::default(),
            audio: AudioCreatePage::default(),
            gallery: gallery::GalleryPage::new(),
            models: ModelsPage::new(),
            settings: SettingsPage::new(),
            task_queue: TaskQueue::new(),
            api_client,
            api_connected: false,
            current_task_id: None,
            current_theme: ThemeId::LinearDark,
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
                match id {
                    NavId::Gallery => {
                        return self.gallery.update(gallery::Message::LoadAssets, &self.api_client).map(Message::Gallery);
                    }
                    NavId::Models => {
                        return self.models.update(models_page::Message::LoadModels, &self.api_client).map(Message::Models);
                    }
                    _ => {}
                }
                iced::Task::none()
            }
            Message::NavAndRefresh(id) => {
                self.nav = id;
                match id {
                    NavId::Gallery => {
                        return self.gallery.update(gallery::Message::LoadAssets, &self.api_client).map(Message::Gallery);
                    }
                    NavId::Models => {
                        return self.models.update(models_page::Message::LoadModels, &self.api_client).map(Message::Models);
                    }
                    _ => {}
                }
                iced::Task::none()
            }
            Message::ThemeChanged(theme) => {
                self.current_theme = theme;
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
            Message::Gallery(msg) => {
                if let gallery::Message::GoToCreate = msg {
                    return iced::Task::perform(async {}, |_| Message::Nav(NavId::ImageCreate));
                }
                self.gallery.update(msg, &self.api_client).map(Message::Gallery)
            }
            Message::Audio(msg) => {
                // Intercept Generate to call backend API
                if let audio_page::Message::Generate = msg {
                    if let Some(client) = self.api_client.clone() {
                        let (endpoint, request) = self.audio.build_request();
                        self.audio.push_log(format!("提交请求到 {}…", endpoint));
                        return iced::Task::perform(
                            async move {
                                match client.post::<serde_json::Value, _>(endpoint, &request).await {
                                    Ok(v) => Message::ApiAudioSubmitted(Ok(v)),
                                    Err(e) => Message::ApiAudioSubmitted(Err(e.to_string())),
                                }
                            },
                            |msg| msg,
                        );
                    }
                }
                self.audio.update(msg, &self.api_client).map(Message::Audio)
            }
            Message::Video(msg) => {
                if let video_page::Message::Generate = msg {
                    if let Some(client) = self.api_client.clone() {
                        let needs_start_image = matches!(self.video.mode, video_page::VideoMode::ImageToVideo)
                            && matches!(self.video.start_image, video_page::SourceVideoState::Uploaded(_));
                        let needs_tail_image = matches!(self.video.mode, video_page::VideoMode::ImageToVideo)
                            && matches!(self.video.tail_image, video_page::SourceVideoState::Uploaded(_));
                        let needs_source_video = matches!(self.video.mode, video_page::VideoMode::Upscale)
                            && matches!(self.video.source_video, video_page::SourceVideoState::Uploaded(_));

                        if needs_start_image || needs_tail_image || needs_source_video {
                            let client_clone = client.clone();
                            let start_path = match &self.video.start_image {
                                video_page::SourceVideoState::Uploaded(p) if needs_start_image => Some(p.clone()),
                                _ => None,
                            };
                            let tail_path = match &self.video.tail_image {
                                video_page::SourceVideoState::Uploaded(p) if needs_tail_image => Some(p.clone()),
                                _ => None,
                            };
                            let source_path = match &self.video.source_video {
                                video_page::SourceVideoState::Uploaded(p) if needs_source_video => Some(p.clone()),
                                _ => None,
                            };
                            let upload_task = iced::Task::perform(
                                async move {
                                    let mut start_id = None;
                                    let mut tail_id = None;
                                    let mut source_id = None;

                                    if let Some(ref path) = start_path {
                                        match client_clone.upload_asset(std::path::Path::new(path), "image/png").await {
                                            Ok(asset) => {
                                                start_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            }
                                            Err(e) => return Err(format!("上传起始图片失败: {}", e)),
                                        }
                                    }

                                    if let Some(ref path) = tail_path {
                                        match client_clone.upload_asset(std::path::Path::new(path), "image/png").await {
                                            Ok(asset) => {
                                                tail_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            }
                                            Err(e) => return Err(format!("上传末尾图片失败: {}", e)),
                                        }
                                    }

                                    if let Some(ref path) = source_path {
                                        let mime = "video/mp4";
                                        match client_clone.upload_asset(std::path::Path::new(path), mime).await {
                                            Ok(asset) => {
                                                source_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            }
                                            Err(e) => return Err(format!("上传源视频失败: {}", e)),
                                        }
                                    }

                                    Ok((start_id, tail_id, source_id))
                                },
                                |result| match result {
                                    Ok((start_id, tail_id, source_id)) => {
                                        Message::ApiAssetsReady {
                                            source_asset_id: source_id,
                                            mask_asset_id: start_id,
                                            control_asset_id: tail_id,
                                        }
                                    }
                                    Err(e) => Message::ApiGenerationSubmitted(Err(e)),
                                },
                            );
                            let task = self.video.update(msg).map(Message::Video);
                            return iced::Task::batch([task, upload_task]);
                        } else {
                            let (endpoint, request) = self.video.build_request(None, None, None);
                            let endpoint = endpoint.to_string();
                            let task = self.video.update(msg).map(Message::Video);
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
                self.video.update(msg).map(Message::Video)
            }
            Message::Models(msg) => {
                self.models.update(msg, &self.api_client).map(Message::Models)
            }
            Message::Settings(msg) => {
                // Intercept theme changes to apply globally
                if let settings_page::Message::ThemeSelected(theme) = msg {
                    return self.update(Message::ThemeChanged(theme));
                }
                self.settings.update(msg, &self.api_client).map(Message::Settings)
            }
            Message::Create(msg) => {
                let task = self.create.update(msg.clone()).map(Message::Create);
                // Intercept Generate to call backend API
                if let create_page::Message::Generate = msg {
                    if let Some(client) = self.api_client.clone() {
                        // Check what assets need uploading
                        let needs_source = !matches!(self.create.mode, create_page::ImageMode::TextToImage)
                            && matches!(self.create.source_image, create_page::SourceImageState::Uploaded(_));
                        let needs_mask = matches!(self.create.mode, create_page::ImageMode::Inpainting)
                            && matches!(self.create.mask_image, create_page::SourceImageState::Uploaded(_));
                        let needs_control = !matches!(self.create.controlnet, create_page::ControlNetOption::None)
                            && matches!(self.create.control_image, create_page::SourceImageState::Uploaded(_));

                        if needs_source || needs_mask || needs_control {
                            // Upload assets first, then submit generation
                            let client_clone = client.clone();
                            let source_path = match &self.create.source_image {
                                create_page::SourceImageState::Uploaded(p) if needs_source => Some(p.clone()),
                                _ => None,
                            };
                            let mask_path = match &self.create.mask_image {
                                create_page::SourceImageState::Uploaded(p) if needs_mask => Some(p.clone()),
                                _ => None,
                            };
                            let control_path = match &self.create.control_image {
                                create_page::SourceImageState::Uploaded(p) if needs_control => Some(p.clone()),
                                _ => None,
                            };
                            let upload_task = iced::Task::perform(
                                async move {
                                    let mut source_id = None;
                                    let mut mask_id = None;
                                    let mut control_id = None;

                                    if let Some(ref path) = source_path {
                                        match client_clone.upload_asset(std::path::Path::new(path), "image/png").await {
                                            Ok(asset) => {
                                                source_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            }
                                            Err(e) => return Err(format!("上传源图失败: {}", e)),
                                        }
                                    }

                                    if let Some(ref path) = mask_path {
                                        match client_clone.upload_asset(std::path::Path::new(path), "image/png").await {
                                            Ok(asset) => {
                                                mask_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            }
                                            Err(e) => return Err(format!("上传遮罩失败: {}", e)),
                                        }
                                    }

                                    if let Some(ref path) = control_path {
                                        match client_clone.upload_asset(std::path::Path::new(path), "image/png").await {
                                            Ok(asset) => {
                                                control_id = asset.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                            }
                                            Err(e) => return Err(format!("上传控制图失败: {}", e)),
                                        }
                                    }

                                    Ok((source_id, mask_id, control_id))
                                },
                                |result| match result {
                                    Ok((source_id, mask_id, control_id)) => Message::ApiAssetsReady {
                                        source_asset_id: source_id,
                                        mask_asset_id: mask_id,
                                        control_asset_id: control_id,
                                    },
                                    Err(e) => Message::ApiGenerationSubmitted(Err(e)),
                                },
                            );
                            return iced::Task::batch([task, upload_task]);
                        } else {
                            // Direct generation (text2image without assets)
                            let (endpoint, request) = self.create.build_request(None, None, None);
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
            Message::ApiAssetsReady { source_asset_id, mask_asset_id, control_asset_id } => {
                if let Some(client) = self.api_client.clone() {
                    if self.nav == NavId::VideoCreate {
                        // source_asset_id = source_video (for upscale) or start_image (for animate)
                        // mask_asset_id = start_image, control_asset_id = tail_image
                        let (endpoint, request) = self.video.build_request(
                            source_asset_id.clone(),
                            mask_asset_id.clone(),
                            control_asset_id.clone(),
                        );
                        let endpoint = endpoint.to_string();
                        return iced::Task::perform(
                            async move {
                                match client.post::<serde_json::Value, _>(&endpoint, &request).await {
                                    Ok(v) => Message::ApiGenerationSubmitted(Ok(v)),
                                    Err(e) => Message::ApiGenerationSubmitted(Err(e.to_string())),
                                }
                            },
                            |msg| msg,
                        );
                    } else {
                        let (endpoint, request) = self.create.build_request(source_asset_id, mask_asset_id, control_asset_id);
                        let endpoint = endpoint.to_string();
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
                }
                iced::Task::none()
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
                        // After health check, load models from registry
                        if let Some(client) = self.api_client.clone() {
                            return iced::Task::perform(
                                async move {
                                    // Load registry for full model config + models status for installation state
                                    let registry_future = client.get::<serde_json::Value>("/api/registry");
                                    let image_models_future = client.get::<serde_json::Value>("/api/models?media=image");
                                    let audio_models_future = client.get::<serde_json::Value>("/api/models?media=audio");
                                    let video_models_future = client.get::<serde_json::Value>("/api/models?media=video");
                                    match tokio::try_join!(registry_future, image_models_future, audio_models_future, video_models_future) {
                                        Ok((registry, image_models, audio_models, video_models)) => {
                                            let mut data = serde_json::Map::new();
                                            data.insert("registry".into(), registry);
                                            data.insert("models".into(), image_models);
                                            data.insert("audio_models".into(), audio_models);
                                            data.insert("video_models".into(), video_models);
                                            Message::ApiModelsLoaded(Ok(serde_json::Value::Object(data)))
                                        }
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
                    Ok(data) => {
                        let mut count = 0;
                        self.create.available_models.clear();

                        // Build lookup: model_id -> (installed, actions from /api/models)
                        let mut model_status: std::collections::HashMap<String, (bool, Vec<String>)> =
                            std::collections::HashMap::new();
                        if let Some(models_resp) = data.get("models").and_then(|v| v.get("models")).and_then(|v| v.as_object()) {
                            for (id, info) in models_resp {
                                let installed = info.get("installed").and_then(|v| v.as_bool()).unwrap_or(false);
                                let actions: Vec<String> = info.get("actions")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                                    .unwrap_or_default();
                                model_status.insert(id.clone(), (installed, actions));
                            }
                        }

                        // Parse registry for names, categories, parameters
                        let registry = data.get("registry").and_then(|v| v.as_object());
                        let models_reg = registry.and_then(|r| r.get("models")).and_then(|v| v.as_object());
                        let _index = registry.and_then(|r| r.get("_index")).and_then(|v| v.as_object());

                        if let Some(models_reg) = models_reg {
                            for (id, config) in models_reg {
                                // Skip non-image models (web UI: imageModelRow checks media === 'image')
                                let media = config.get("media").and_then(|v| v.as_str()).unwrap_or("");
                                if media != "image" {
                                    continue;
                                }

                                let category = config.get("category")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("base_models")
                                    .to_string();

                                // Web UI excludes only 'loras' from base model picker
                                if category == "loras" {
                                    continue;
                                }

                                // Get name from bilingual object or string
                                let name = config.get("name")
                                    .and_then(|v| v.as_object())
                                    .and_then(|o| o.get("zh").or_else(|| o.get("en")))
                                    .and_then(|v| v.as_str())
                                    .or_else(|| config.get("name").and_then(|v| v.as_str()))
                                    .unwrap_or(id)
                                    .to_string();

                                let family = config.get("family").and_then(|v| v.as_str()).unwrap_or("");

                                // Use actions from /api/models if available, else from registry
                                let (installed, actions) = model_status.get(id)
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        // Fallback: parse from registry config
                                        // Registry actions is an object: { "create": {}, "rewrite": {} }
                                        let acts: Vec<String> = config.get("actions")
                                            .and_then(|v| v.as_object())
                                            .map(|obj| obj.keys().cloned().collect())
                                            .unwrap_or_default();
                                        (false, acts)
                                    });

                                let commercial = config.get("commercial_use_allowed")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);

                                // Default steps/guidance from parameters
                                let steps = config.get("parameters")
                                    .and_then(|v| v.as_object())
                                    .and_then(|p| p.get("steps"))
                                    .and_then(|v| v.as_object())
                                    .and_then(|s| s.get("default"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(20) as u8;

                                let cfg = config.get("parameters")
                                    .and_then(|v| v.as_object())
                                    .and_then(|p| p.get("guidance"))
                                    .and_then(|v| v.as_object())
                                    .and_then(|s| s.get("default"))
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(5.0) as f32;

                                // Build display label: "Name · Family" (matching web UI style)
                                let label = if family.is_empty() {
                                    name.clone()
                                } else {
                                    format!("{} · {}", name, family)
                                };

                                self.create.available_models.push(
                                    crate::create_page::ModelOption::Dynamic {
                                        id: id.clone(),
                                        label,
                                        name,
                                        category,
                                        actions,
                                        installed,
                                        commercial_use_allowed: commercial,
                                        steps,
                                        cfg,
                                    }
                                );
                                count += 1;
                            }
                        }

                        self.create.push_log(format!("已加载 {} 个模型", count));

                        // Rebuild filtered list and select first valid model
                        self.create.rebuild_filtered_models();

                        // If filtered list is empty but we have models, log warning
                        if self.create.filtered_models.is_empty() && !self.create.available_models.is_empty() {
                            self.create.push_log("警告: 过滤后无可用模型，显示全部".into());
                        }

                        // Select first model that supports current mode
                        let required_action = crate::create_page::ModelOption::mode_required_action(self.create.mode);
                        let first_valid = self.create.available_models
                            .iter()
                            .find(|m| {
                                m.category() != "loras" &&
                                m.supports_action(required_action) &&
                                (!self.create.commercial_only || m.commercial_use_allowed())
                            })
                            .cloned();

                        if let Some(first) = first_valid {
                            self.create.model = first.clone();
                            self.create.steps = first.default_steps();
                            self.create.cfg = first.default_cfg();
                            self.create.steps_input = format!("{:.0}", self.create.steps);
                            self.create.cfg_input = format!("{:.1}", self.create.cfg);
                        }

                        // Load audio models for audio page
                        if let Some(audio_models_resp) = data.get("audio_models").and_then(|v| v.get("models")).and_then(|v| v.as_object()) {
                            let mut audio_count = 0;
                            self.audio.available_models.clear();
                            for (id, info) in audio_models_resp {
                                let name = info.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(id)
                                    .to_string();
                                let family = info.get("family")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let label = if family.is_empty() {
                                    name.clone()
                                } else {
                                    format!("{} · {}", name, family)
                                };
                                self.audio.available_models.push(
                                    crate::audio_page::AudioModelOption::Dynamic {
                                        id: id.clone(),
                                        label,
                                        name,
                                    }
                                );
                                audio_count += 1;
                            }
                            self.audio.push_log(format!("已加载 {} 个音频模型", audio_count));
                            if !self.audio.available_models.is_empty() {
                                self.audio.model = self.audio.available_models[0].clone();
                            }
                        }

                        // Load video models
                        if let Some(video_models_resp) = data.get("video_models").and_then(|v| v.get("models")).and_then(|v| v.as_object()) {
                            let mut video_count = 0;
                            self.video.available_models.clear();
                            for (id, info) in video_models_resp {
                                let name = info.get("name").and_then(|v| v.as_str()).unwrap_or(id).to_string();
                                let family = info.get("family").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let actions: Vec<String> = info.get("actions").and_then(|v| v.as_array()).map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
                                let installed = info.get("installed").and_then(|v| v.as_bool()).unwrap_or(false);
                                let label = if family.is_empty() { name.clone() } else { format!("{} · {}", name, family) };
                                self.video.available_models.push(crate::video_page::VideoModelOption::Dynamic { id: id.clone(), label, name, actions, installed });
                                video_count += 1;
                            }
                            self.video.push_log(format!("已加载 {} 个视频模型", video_count));
                            if !self.video.available_models.is_empty() {
                                self.video.model = self.video.available_models[0].clone();
                            }
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
                            self.current_task_id = Some(tid.clone());
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
            Message::ApiAudioSubmitted(result) => {
                self.audio.update(
                    crate::audio_page::Message::ApiAudioSubmitted(result),
                    &self.api_client,
                ).map(Message::Audio)
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
            Message::ApiTaskListPoll => {
                if let Some(client) = self.api_client.clone() {
                    return iced::Task::perform(
                        async move {
                            match client.get::<serde_json::Value>("/api/tasks?limit=50").await {
                                Ok(v) => {
                                    let tasks = v.get("tasks").and_then(|t| t.as_array()).cloned().unwrap_or_default();
                                    Message::ApiTaskListResult(Ok(tasks))
                                }
                                Err(e) => Message::ApiTaskListResult(Err(e.to_string())),
                            }
                        },
                        |msg| msg,
                    );
                }
                iced::Task::none()
            }
            Message::ApiTaskListResult(result) => {
                match result {
                    Ok(tasks) => {
                        let items: Vec<crate::task_queue::TaskItem> = tasks.into_iter().filter_map(|t| {
                            let id = t.get("id")?.as_str()?.to_string();
                            let status_str = t.get("status")?.as_str()?;
                            let kind = t.get("kind")?.as_str().unwrap_or("unknown");
                            let model = t.get("model_id")?.as_str().unwrap_or("");
                            let title = t.get("prompt")?.as_str().unwrap_or("");
                            let progress = t.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                            let created_at_str = t.get("created_at")?.as_str()?;
                            
                            // Parse created_at
                            let created_at = chrono::DateTime::parse_from_rfc3339(created_at_str)
                                .ok()
                                .map(|dt| std::time::Instant::now() - std::time::Duration::from_secs(
                                    chrono::Utc::now().timestamp().max(0) as u64 - dt.timestamp().max(0) as u64
                                ))
                                .unwrap_or_else(std::time::Instant::now);
                            
                            let status = match status_str {
                                "queued" => {
                                    let pos = t.get("queue_position").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                    let eta = t.get("estimated_wait_seconds").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                    crate::task_queue::TaskStatus::Queued { position: pos, eta_seconds: eta }
                                }
                                "running" => {
                                    let step = t.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                                    let total = t.get("total").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
                                    let phase = t.get("phase").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    crate::task_queue::TaskStatus::Running { step, total, phase }
                                }
                                "completed" => crate::task_queue::TaskStatus::Completed,
                                "failed" => {
                                    let error = t.get("error").and_then(|v| v.as_str()).unwrap_or("未知错误").to_string();
                                    crate::task_queue::TaskStatus::Failed { error }
                                }
                                "cancelled" => crate::task_queue::TaskStatus::Cancelled,
                                _ => crate::task_queue::TaskStatus::Pending,
                            };
                            
                            Some(crate::task_queue::TaskItem {
                                id,
                                title: if title.is_empty() { kind.to_string() } else { title.chars().take(40).collect() },
                                mode: kind.to_string(),
                                model: model.to_string(),
                                status,
                                created_at,
                                progress,
                            })
                        }).collect();
                        
                        return self.task_queue.update(crate::task_queue::TaskQueueMessage::SetTasks(items)).map(Message::TaskQueue);
                    }
                    Err(e) => {
                        // Silently ignore task list poll errors
                        let _ = e;
                    }
                }
                iced::Task::none()
            }
            Message::ImageSaved(result) => {
                match result {
                    Ok(path) => self.create.push_log(format!("图片已保存: {}", path)),
                    Err(e) => self.create.push_log(format!("保存图片失败: {}", e)),
                }
                iced::Task::none()
            }
            Message::KeyEvent(key) => {
                use iced::keyboard::key::Named;
                if self.nav == NavId::Gallery && self.gallery.show_lightbox {
                    if let iced::keyboard::Key::Named(named) = key {
                        return match named {
                            Named::ArrowLeft => self.gallery.update(gallery::Message::LightboxPrev, &self.api_client).map(Message::Gallery),
                            Named::ArrowRight => self.gallery.update(gallery::Message::LightboxNext, &self.api_client).map(Message::Gallery),
                            Named::Escape => self.gallery.update(gallery::Message::CloseLightbox, &self.api_client).map(Message::Gallery),
                            _ => iced::Task::none(),
                        };
                    }
                }
                iced::Task::none()
            }
        }
    }

    pub fn subscription(&self) -> iced::Subscription<Message> {
        use iced::keyboard::{self, key::Named};
        use iced::time;
        

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

        let task_poll = time::every(std::time::Duration::from_secs(3))
            .map(|_| Message::ApiTaskListPoll);

        let file_drop = iced::event::listen().filter_map(|event| {
            if let iced::Event::Window(iced::window::Event::FileDropped(path)) = event {
                return Some(Message::Create(create_page::Message::SourceImageDropped(path.to_string_lossy().to_string())));
            }
            None
        });

        let global_events = iced::event::listen().filter_map(|event| {
            // Forward keyboard events for dynamic dispatch in update()
            if let iced::Event::Keyboard(keyboard::Event::KeyPressed { key, .. }) = &event {
                return Some(Message::KeyEvent(key.clone()));
            }
            None
        });

        iced::Subscription::batch(vec![keyboard_sub, memory_poll, task_poll, file_drop, global_events])
    }

    pub fn view(&self) -> iced::Element<'_, Message> {
        use dq_layout::dq_sidebar;
        use dq_layout::{NavItem, SidebarSection};
        use dq_theme::{page_container, subtle_scrollbar, vertical_divider};
        use iced::widget::{column, container, row, scrollable};
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
            NavId::AudioCreate => {
                self.audio.view(&self.api_client).map(Message::Audio)
            }
            NavId::VideoCreate => {
                self.video.view().map(Message::Video)
            }
            NavId::Gallery => {
                self.gallery.view(&self.api_client).map(Message::Gallery)
            }
            NavId::Models => {
                self.models.view(&self.api_client).map(Message::Models)
            }
            NavId::Settings => {
                self.settings.view(&self.api_client, self.current_theme).map(Message::Settings)
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

        // Connection banner when API is unavailable
        let banner: Option<Element<Message>> = if self.api_client.is_some() && !self.api_connected {
            Some(container(
                row![
                    iced::widget::text("⚠").size(typography::BODY),
                    iced::widget::text("后端服务未连接 — 请确保后端 API 正在运行 (http://127.0.0.1:7800)")
                        .size(typography::CAPTION).color(dq_tokens::color::TEXT_PRIMARY),
                    iced::widget::Space::new().width(Length::Fill),
                    dq_components::dq_button("重新连接", dq_components::ButtonVariant::Secondary, dq_components::ButtonSize::Sm, dq_components::ButtonWidth::Hug, Some(Message::ApiHealthCheck)),
                ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill)
            )
            .padding([spacing::SM, spacing::MD]).width(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(dq_tokens::color::WARNING)),
                ..Default::default()
            })
            .into())
        } else { None };

        let main = column![
            if let Some(b) = banner { b } else { iced::widget::Space::new().height(0).into() },
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
        ]
        .width(Length::Fill)
        .height(Length::Fill);

        container(main)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(page_container)
            .into()
    }
}
