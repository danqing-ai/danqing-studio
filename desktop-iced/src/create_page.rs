#![allow(dead_code)]
use dq_components::{
    chevron_icon, dq_control_button, dq_header_button, dq_mode_tabs,
    dq_pick_list, dq_prompt_editor, dq_prompt_preset_row,
    dq_primary_action, dq_slider_with_input, dq_text_input, dq_text_input_multiline, image_placeholder_icon, log_panel, default_logs, LogLine, ModeTabOption,
    section_card, surface_card,
    phosphor_icon_button, PhosphorIcon,
    canvas_editor::{canvas_editor, tool_selector, brush_size_selector, zoom_controls,
        CanvasEditorState, CanvasEditorMessage, CanvasTool},
    before_after::{before_after_slider, before_after_labels,
        BeforeAfterState, BeforeAfterMessage},
    staging_area::{StagedResult, StagingMessage},
    resizable_split::{SplitViewState, SplitViewMessage},
};
use dq_tokens::{color, spacing, typography};
use iced::widget::{checkbox, column, container, row, text, text_editor, Space};
use iced::{Alignment, Element, Length};
use std::time::Duration;

const PROMPT_MAX: usize = 1000;
const NEGATIVE_MAX: usize = 500;

fn truncate_editor(content: &mut text_editor::Content, max_chars: usize) {
    let text = content.text();
    if text.chars().count() <= max_chars {
        return;
    }
    let truncated: String = text.chars().take(max_chars).collect();
    *content = text_editor::Content::with_text(&truncated);
}

fn editor_char_count(content: &text_editor::Content) -> usize {
    content.text().chars().count()
}

fn file_to_base64_data_uri(path: &str) -> Option<String> {
    let data = std::fs::read(path).ok()?;
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &data);
    let mime = match std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    Some(format!("data:{};base64,{}", mime, b64))
}

#[derive(Debug, Clone)]
pub enum Message {
    ModeChanged(ImageMode),
    ModelSelected(ModelOption),
    PresetSelected(PresetOption),
    TitleChanged(String),
    PromptEdited(text_editor::Action),
    NegativePromptEdited(text_editor::Action),
    ToggleNegativeOpen,
    ToggleAdvancedOpen,
    SeedChanged(String),
    RandomizeSeed,
    StepsChanged(f32),
    StepsInputChanged(String),
    CfgChanged(f32),
    CfgInputChanged(String),
    SchedulerSelected(SchedulerOption),
    WidthSelected(WidthOption),
    HeightSelected(HeightOption),
    LoraSelected(LoraOption),
    LoraWeightChanged(f32),
    LoraWeightInputChanged(String),
    BatchSizeSelected(BatchSizeOption),
    BatchCountSelected(BatchCountOption),
    ReferenceStrengthSelected(ReferenceStrengthOption),
    UpscaleFactorSelected(UpscaleFactorOption),
    DenoiseSelected(DenoiseOption),
    OutpaintDirectionSelected(OutpaintDirectionOption),
    OutpaintPixelsChanged(f32),
    OutpaintPixelsInputChanged(String),
    CommercialToggled(bool),
    RestoreDefaults,
    LoadPreset,
    ClearLogs,
    Generate,
    CancelGeneration,
    GenerateStep,
    GenerateProgress { progress: u8, step: u32, total: u32, phase: GeneratePhase },
    GenerateComplete { result_urls: Vec<String> },
    GenerateFailed { error: String },
    // Upload / mask
    UploadSourceImage,
    UploadMaskImage,
    UploadControlImage,
    MaskImageSelected(String),
    ControlImageSelected(String),
    ClearSourceImage,
    ClearMaskImage,
    ClearControlImage,
    ToggleMaskEditor,
    NoOp,
    // Canvas editor
    CanvasEditorMsg(CanvasEditorMessage),
    ToggleMaskDrawMode,
    // Drag & drop
    SourceImageDropped(String),
    // ControlNet
    ControlNetSelected(ControlNetOption),
    ControlNetStrengthChanged(f32),
    // Enhance
    StartEnhance,
    // Preview dialog
    PreviewImage,
    ClosePreview,
    // Memory poll
    MemoryPoll,
    // Before/After slider
    BeforeAfterMsg(BeforeAfterMessage),
    // Staging area
    StagingMsg(StagingMessage),
    // Split view resize
    SplitViewMsg(SplitViewMessage),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageMode {
    #[default]
    TextToImage,
    ReferenceImage,
    EditByDescription,
    Inpainting,
    Outpainting,
    Upscale,
}

impl ImageMode {
    const ALL: &'static [ImageMode] = &[
        ImageMode::TextToImage,
        ImageMode::ReferenceImage,
        ImageMode::EditByDescription,
        ImageMode::Inpainting,
        ImageMode::Outpainting,
        ImageMode::Upscale,
    ];

    fn label(self) -> &'static str {
        match self {
            ImageMode::TextToImage => "文生图",
            ImageMode::ReferenceImage => "参考原图生成",
            ImageMode::EditByDescription => "按描述改图",
            ImageMode::Inpainting => "局部修饰",
            ImageMode::Outpainting => "扩展画布",
            ImageMode::Upscale => "精修放大",
        }
    }

    #[allow(dead_code)]
    fn description(self) -> &'static str {
        match self {
            ImageMode::TextToImage => {
                "根据文字描述生成图像，无需原图；可在高级参数中调整分辨率、步数与 LoRA 等。"
            }
            ImageMode::ReferenceImage => {
                "上传参考图并结合提示词生成新画面；可在高级参数中调整强度与分辨率。"
            }
            ImageMode::EditByDescription => {
                "用自然语言描述修改目标，基于原图进行定向改图。"
            }
            ImageMode::Inpainting => "在局部选区内按提示词重绘，保留其余区域。",
            ImageMode::Outpainting => "向指定方向扩展画布并生成延伸内容。",
            ImageMode::Upscale => "对已有图像进行放大与细节增强。",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerateState {
    #[default]
    Idle,
    Submitting,
    Generating { progress: u8, step: u32, total: u32, phase: GeneratePhase },
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneratePhase {
    #[default]
    Queued,
    Denoising,
    Decoding,
    PostProcessing,
}

impl GeneratePhase {
    pub fn from_str(s: &str) -> Self {
        match s {
            "denoising" => GeneratePhase::Denoising,
            "decoding" => GeneratePhase::Decoding,
            "post_processing" => GeneratePhase::PostProcessing,
            _ => GeneratePhase::Queued,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecentGeneration {
    pub title: String,
    pub subtitle: String,
    pub prompt_snippet: String,
}

#[derive(Debug, Clone)]
pub enum ModelOption {
    Flux2Klein,
    Flux1Dev,
    ZImageTurbo,
    Dynamic {
        id: String,
        label: String,
        name: String,
        category: String,
        actions: Vec<String>,
        installed: bool,
        commercial_use_allowed: bool,
        steps: u8,
        cfg: f32,
    },
}

impl PartialEq for ModelOption {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ModelOption::Flux2Klein, ModelOption::Flux2Klein) => true,
            (ModelOption::Flux1Dev, ModelOption::Flux1Dev) => true,
            (ModelOption::ZImageTurbo, ModelOption::ZImageTurbo) => true,
            (ModelOption::Dynamic { id: a, .. }, ModelOption::Dynamic { id: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl Eq for ModelOption {}

impl Default for ModelOption {
    fn default() -> Self {
        ModelOption::ZImageTurbo
    }
}

impl ModelOption {
    const STATIC_ALL: &'static [ModelOption] = &[
        ModelOption::ZImageTurbo,
        ModelOption::Flux2Klein,
        ModelOption::Flux1Dev,
    ];

    pub fn id(&self) -> String {
        match self {
            ModelOption::Flux2Klein => "flux2-klein".into(),
            ModelOption::Flux1Dev => "flux1-dev".into(),
            ModelOption::ZImageTurbo => "z-image-turbo".into(),
            ModelOption::Dynamic { id, .. } => id.clone(),
        }
    }

    pub fn label(&self) -> String {
        match self {
            ModelOption::Flux2Klein => "flux2-klein · FP16 完整版".into(),
            ModelOption::Flux1Dev => "flux1-dev · FP16".into(),
            ModelOption::ZImageTurbo => "Z-Image-Turbo · FP16 完整版".into(),
            ModelOption::Dynamic { label, .. } => label.clone(),
        }
    }

    pub fn name(&self) -> String {
        match self {
            ModelOption::Flux2Klein => "flux2-klein".into(),
            ModelOption::Flux1Dev => "flux1-dev".into(),
            ModelOption::ZImageTurbo => "z-image-turbo".into(),
            ModelOption::Dynamic { name, .. } => name.clone(),
        }
    }

    pub fn category(&self) -> String {
        match self {
            ModelOption::Flux2Klein => "base_models".into(),
            ModelOption::Flux1Dev => "base_models".into(),
            ModelOption::ZImageTurbo => "base_models".into(),
            ModelOption::Dynamic { category, .. } => category.clone(),
        }
    }

    pub fn actions(&self) -> Vec<String> {
        match self {
            // API-level action names (matching backend api_action_frozenset)
            ModelOption::Flux2Klein => vec!["generate".into()],
            ModelOption::Flux1Dev => vec!["generate".into(), "edit".into(), "upscale".into()],
            ModelOption::ZImageTurbo => vec!["generate".into(), "edit".into()],
            ModelOption::Dynamic { actions, .. } => actions.clone(),
        }
    }

    pub fn installed(&self) -> bool {
        match self {
            ModelOption::Flux2Klein => true,
            ModelOption::Flux1Dev => true,
            ModelOption::ZImageTurbo => true,
            ModelOption::Dynamic { installed, .. } => *installed,
        }
    }

    pub fn commercial_use_allowed(&self) -> bool {
        match self {
            ModelOption::Flux2Klein => true,
            ModelOption::Flux1Dev => true,
            ModelOption::ZImageTurbo => true,
            ModelOption::Dynamic { commercial_use_allowed, .. } => *commercial_use_allowed,
        }
    }

    /// Check if this model supports the given action
    pub fn supports_action(&self, action: &str) -> bool {
        self.actions().iter().any(|a| a == action)
    }

    #[allow(dead_code)]
    fn description(&self) -> String {
        match self {
            ModelOption::ZImageTurbo => {
                "道义实验室 Z-Image-Turbo，6B 参数 DiT，9 步快速高质量出图".into()
            }
            ModelOption::Flux2Klein => "FLUX.2 Klein 轻量模型，适合快速迭代与预览。".into(),
            ModelOption::Flux1Dev => "FLUX.1 Dev 高质量文生图，步数与 CFG 可调。".into(),
            ModelOption::Dynamic { .. } => "自定义模型".into(),
        }
    }

    #[allow(dead_code)]
    fn steps_hint(&self) -> &'static str {
        match self {
            ModelOption::ZImageTurbo => "Turbo 模型建议 8–12 步",
            _ => "真实人像建议 15–20 步",
        }
    }

    #[allow(dead_code)]
    fn cfg_hint(&self) -> &'static str {
        match self {
            ModelOption::ZImageTurbo => "Turbo 模型固定为 0",
            _ => "CFG 越高越贴近提示词，建议 4–7",
        }
    }

    pub fn default_steps(&self) -> f32 {
        match self {
            ModelOption::ZImageTurbo => 10.0,
            ModelOption::Dynamic { steps, .. } => *steps as f32,
            _ => 20.0,
        }
    }

    pub fn default_cfg(&self) -> f32 {
        match self {
            ModelOption::ZImageTurbo => 0.0,
            ModelOption::Dynamic { cfg, .. } => *cfg,
            _ => 5.0,
        }
    }

    /// Returns the API-level action required for the current image mode.
    /// Matches backend api_action_frozenset mapping:
    ///   create -> generate, rewrite/retouch/extend -> edit, upscale -> upscale
    pub fn mode_required_action(mode: ImageMode) -> &'static str {
        match mode {
            ImageMode::TextToImage => "generate",
            ImageMode::ReferenceImage => "edit",
            ImageMode::EditByDescription => "edit",
            ImageMode::Inpainting => "edit",
            ImageMode::Outpainting => "edit",
            ImageMode::Upscale => "upscale",
        }
    }
}

impl std::fmt::Display for ModelOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetOption {
    Default,
    Photo,
    Illustration,
}

impl PresetOption {
    const ALL: &'static [PresetOption] = &[
        PresetOption::Default,
        PresetOption::Photo,
        PresetOption::Illustration,
    ];

    fn label(&self) -> &'static str {
        match self {
            PresetOption::Default => "默认",
            PresetOption::Photo => "写实摄影",
            PresetOption::Illustration => "插画风格",
        }
    }
}

impl std::fmt::Display for PresetOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchSizeOption {
    #[default]
    One,
    Two,
    Four,
}

impl BatchSizeOption {
    const ALL: &'static [BatchSizeOption] = &[BatchSizeOption::One, BatchSizeOption::Two, BatchSizeOption::Four];

    fn label(&self) -> &'static str {
        match self {
            BatchSizeOption::One => "1 张",
            BatchSizeOption::Two => "2 张",
            BatchSizeOption::Four => "4 张",
        }
    }

    pub fn value(&self) -> u8 {
        match self {
            BatchSizeOption::One => 1,
            BatchSizeOption::Two => 2,
            BatchSizeOption::Four => 4,
        }
    }
}

impl std::fmt::Display for BatchSizeOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchCountOption {
    #[default]
    One,
    Two,
    Three,
    Four,
}

impl BatchCountOption {
    const ALL: &'static [BatchCountOption] = &[BatchCountOption::One, BatchCountOption::Two, BatchCountOption::Three, BatchCountOption::Four];

    fn label(&self) -> &'static str {
        match self {
            BatchCountOption::One => "1 组",
            BatchCountOption::Two => "2 组",
            BatchCountOption::Three => "3 组",
            BatchCountOption::Four => "4 组",
        }
    }

    pub fn value(&self) -> u8 {
        match self {
            BatchCountOption::One => 1,
            BatchCountOption::Two => 2,
            BatchCountOption::Three => 3,
            BatchCountOption::Four => 4,
        }
    }
}

impl std::fmt::Display for BatchCountOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceStrengthOption {
    #[default]
    Low,
    Medium,
    High,
    VeryHigh,
}

impl ReferenceStrengthOption {
    const ALL: &'static [ReferenceStrengthOption] = &[
        ReferenceStrengthOption::Low,
        ReferenceStrengthOption::Medium,
        ReferenceStrengthOption::High,
        ReferenceStrengthOption::VeryHigh,
    ];

    fn label(&self) -> &'static str {
        match self {
            ReferenceStrengthOption::Low => "低 (0.3)",
            ReferenceStrengthOption::Medium => "中 (0.5)",
            ReferenceStrengthOption::High => "高 (0.7)",
            ReferenceStrengthOption::VeryHigh => "极高 (0.9)",
        }
    }

    pub fn value(&self) -> f32 {
        match self {
            ReferenceStrengthOption::Low => 0.3,
            ReferenceStrengthOption::Medium => 0.5,
            ReferenceStrengthOption::High => 0.7,
            ReferenceStrengthOption::VeryHigh => 0.9,
        }
    }
}

impl std::fmt::Display for ReferenceStrengthOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleFactorOption {
    #[default]
    X2,
    X4,
}

impl UpscaleFactorOption {
    const ALL: &'static [UpscaleFactorOption] = &[UpscaleFactorOption::X2, UpscaleFactorOption::X4];

    fn label(&self) -> &'static str {
        match self {
            UpscaleFactorOption::X2 => "2×",
            UpscaleFactorOption::X4 => "4×",
        }
    }

    pub fn value(&self) -> u8 {
        match self {
            UpscaleFactorOption::X2 => 2,
            UpscaleFactorOption::X4 => 4,
        }
    }
}

impl std::fmt::Display for UpscaleFactorOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenoiseOption {
    #[default]
    Low,
    Medium,
    High,
}

impl DenoiseOption {
    const ALL: &'static [DenoiseOption] = &[DenoiseOption::Low, DenoiseOption::Medium, DenoiseOption::High];

    fn label(&self) -> &'static str {
        match self {
            DenoiseOption::Low => "轻度 (0.2)",
            DenoiseOption::Medium => "中度 (0.4)",
            DenoiseOption::High => "重度 (0.6)",
        }
    }

    pub fn value(&self) -> f32 {
        match self {
            DenoiseOption::Low => 0.2,
            DenoiseOption::Medium => 0.4,
            DenoiseOption::High => 0.6,
        }
    }
}

impl std::fmt::Display for DenoiseOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutpaintDirectionOption {
    #[default]
    All,
    Left,
    Right,
    Up,
    Down,
}

impl OutpaintDirectionOption {
    const ALL: &'static [OutpaintDirectionOption] = &[
        OutpaintDirectionOption::All,
        OutpaintDirectionOption::Left,
        OutpaintDirectionOption::Right,
        OutpaintDirectionOption::Up,
        OutpaintDirectionOption::Down,
    ];

    fn label(&self) -> &'static str {
        match self {
            OutpaintDirectionOption::All => "四周",
            OutpaintDirectionOption::Left => "向左",
            OutpaintDirectionOption::Right => "向右",
            OutpaintDirectionOption::Up => "向上",
            OutpaintDirectionOption::Down => "向下",
        }
    }

    pub fn id(&self) -> &'static str {
        match self {
            OutpaintDirectionOption::All => "all",
            OutpaintDirectionOption::Left => "left",
            OutpaintDirectionOption::Right => "right",
            OutpaintDirectionOption::Up => "up",
            OutpaintDirectionOption::Down => "down",
        }
    }

    pub fn ids(&self) -> Vec<String> {
        match self {
            OutpaintDirectionOption::All => vec!["top".into(), "bottom".into(), "left".into(), "right".into()],
            OutpaintDirectionOption::Left => vec!["left".into()],
            OutpaintDirectionOption::Right => vec!["right".into()],
            OutpaintDirectionOption::Up => vec!["top".into()],
            OutpaintDirectionOption::Down => vec!["bottom".into()],
        }
    }
}

impl std::fmt::Display for OutpaintDirectionOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerOption {
    #[default]
    Linear,
    Euler,
    Dpmpp,
}

impl SchedulerOption {
    const ALL: &'static [SchedulerOption] = &[
        SchedulerOption::Linear,
        SchedulerOption::Euler,
        SchedulerOption::Dpmpp,
    ];

    pub fn id(self) -> &'static str {
        match self {
            SchedulerOption::Linear => "linear",
            SchedulerOption::Euler => "euler",
            SchedulerOption::Dpmpp => "dpmpp_2m",
        }
    }

    fn label(self) -> &'static str {
        self.id()
    }
}

impl std::fmt::Display for SchedulerOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WidthOption {
    W768,
    #[default]
    W1024,
    W1344,
}

impl WidthOption {
    const ALL: &'static [WidthOption] = &[WidthOption::W768, WidthOption::W1024, WidthOption::W1344];

    pub fn value(self) -> u16 {
        match self {
            WidthOption::W768 => 768,
            WidthOption::W1024 => 1024,
            WidthOption::W1344 => 1344,
        }
    }
}

impl std::fmt::Display for WidthOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeightOption {
    #[default]
    H768,
    H1024,
    H1280,
}

impl HeightOption {
    const ALL: &'static [HeightOption] = &[HeightOption::H768, HeightOption::H1024, HeightOption::H1280];

    pub fn value(self) -> u16 {
        match self {
            HeightOption::H768 => 768,
            HeightOption::H1024 => 1024,
            HeightOption::H1280 => 1280,
        }
    }
}

impl std::fmt::Display for HeightOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoraOption {
    #[default]
    None,
    Portrait,
    Anime,
}

impl LoraOption {
    const ALL: &'static [LoraOption] = &[LoraOption::None, LoraOption::Portrait, LoraOption::Anime];

    pub fn id(&self) -> &'static str {
        match self {
            LoraOption::None => "none",
            LoraOption::Portrait => "portrait",
            LoraOption::Anime => "anime",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            LoraOption::None => "不使用 LoRA",
            LoraOption::Portrait => "人像增强 LoRA",
            LoraOption::Anime => "动漫风格 LoRA",
        }
    }
}

impl std::fmt::Display for LoraOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlNetOption {
    #[default]
    None,
    Canny,
    Depth,
    OpenPose,
}

impl ControlNetOption {
    const ALL: &'static [ControlNetOption] = &[
        ControlNetOption::None,
        ControlNetOption::Canny,
        ControlNetOption::Depth,
        ControlNetOption::OpenPose,
    ];

    pub fn id(&self) -> &'static str {
        match self {
            ControlNetOption::None => "none",
            ControlNetOption::Canny => "canny",
            ControlNetOption::Depth => "depth",
            ControlNetOption::OpenPose => "openpose",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            ControlNetOption::None => "不使用 ControlNet",
            ControlNetOption::Canny => "Canny 边缘",
            ControlNetOption::Depth => "深度图",
            ControlNetOption::OpenPose => "OpenPose 姿态",
        }
    }
}

impl std::fmt::Display for ControlNetOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SourceImageState {
    #[default]
    Empty,
    Uploaded(String),
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub used_gb: f32,
    pub total_gb: f32,
    pub mlx_active_gb: f32,
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            used_gb: 24.0,
            total_gb: 128.0,
            mlx_active_gb: 22.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreatePage {
    pub mode: ImageMode,
    pub mode_tabs: Vec<ModeTabOption<ImageMode>>,
    pub model: ModelOption,
    pub preset: PresetOption,
    pub width: WidthOption,
    pub height: HeightOption,
    pub scheduler: SchedulerOption,
    pub lora: LoraOption,
    pub batch_size: BatchSizeOption,
    pub batch_count: BatchCountOption,
    pub title: String,
    pub prompt: text_editor::Content,
    pub negative_prompt: text_editor::Content,
    pub negative_open: bool,
    pub advanced_open: bool,
    pub seed: String,
    pub steps: f32,
    pub steps_input: String,
    pub cfg: f32,
    pub cfg_input: String,
    pub commercial_only: bool,
    pub params_dirty: bool,
    pub generate_state: GenerateState,
    pub validation_error: Option<String>,
    pub logs: Vec<LogLine>,
    pub recent_generations: Vec<RecentGeneration>,
    // 模式专属参数
    pub reference_strength: ReferenceStrengthOption,
    pub upscale_factor: UpscaleFactorOption,
    pub denoise: DenoiseOption,
    pub outpaint_direction: OutpaintDirectionOption,
    pub outpaint_pixels: f32,
    pub outpaint_pixels_input: String,
    // 上传状态
    pub source_image: SourceImageState,
    pub mask_image: SourceImageState,
    pub control_image: SourceImageState,
    pub mask_editor_open: bool,
    pub mask_draw_mode: bool,
    // 画布编辑器
    pub canvas_editor_state: CanvasEditorState,
    // Preview dialog
    pub preview_open: bool,
    // LoRA weight
    pub lora_weight: f32,
    pub lora_weight_input: String,
    // ControlNet
    pub controlnet: ControlNetOption,
    pub controlnet_strength: f32,
    pub controlnet_strength_input: String,
    // 内存监控
    pub memory_info: MemoryInfo,
    // 增强offer
    pub enhance_offer_visible: bool,
    // Before/After 对比
    pub before_after_state: BeforeAfterState,
    // 暂存区
    pub staged_results: Vec<StagedResult>,
    // 可调整分割视图
    pub split_view_state: SplitViewState,
    // 生成结果图片路径
    pub generated_image_path: Option<String>,
    // 后端动态模型列表
    pub available_models: Vec<ModelOption>,
    // 当前过滤后的模型列表（用于显示）
    pub filtered_models: Vec<ModelOption>,
}

impl CreatePage {
    pub fn width(&self) -> u16 {
        self.width.value()
    }

    pub fn height(&self) -> u16 {
        self.height.value()
    }

    /// Compute total `n` from batch_size × batch_count (clamped to backend max 8).
    fn total_n(&self) -> i32 {
        let total = self.batch_size.value() as i32 * self.batch_count.value() as i32;
        total.clamp(1, 8)
    }

    /// Build adapters array for LoRA.
    fn build_adapters(&self) -> Vec<serde_json::Value> {
        match self.lora {
            LoraOption::None => vec![],
            LoraOption::Portrait => vec![serde_json::json!({"id": "portrait", "weight": self.lora_weight })],
            LoraOption::Anime => vec![serde_json::json!({"id": "anime", "weight": self.lora_weight })],
        }
    }

    /// Build structural_guide for ControlNet if active.
    fn build_structural_guide(&self, control_asset_id: Option<String>) -> Option<serde_json::Value> {
        if matches!(self.controlnet, ControlNetOption::None) {
            return None;
        }
        let control_id = control_asset_id?;
        Some(serde_json::json!({
            "asset_id": control_id,
            "type": self.controlnet.id(),
            "weight": self.controlnet_strength,
        }))
    }

    /// Build a JSON request body aligned with backend contracts.
    /// Accepts uploaded asset IDs for source/mask/control images.
    pub fn build_request(
        &self,
        source_asset_id: Option<String>,
        mask_asset_id: Option<String>,
        control_asset_id: Option<String>,
    ) -> (&'static str, serde_json::Value) {
        use serde_json::json;

        let n = self.total_n();
        let adapters = self.build_adapters();
        let structural_guide = self.build_structural_guide(control_asset_id);
        let seed = self.seed.parse::<i64>().ok();

        match self.mode {
            ImageMode::TextToImage => {
                let mut payload = json!({
                    "model": self.model.id(),
                    "title": self.title,
                    "prompt": self.prompt.text(),
                    "negative_prompt": self.negative_prompt.text(),
                    "size": format!("{}x{}", self.width.value(), self.height.value()),
                    "n": n,
                    "steps": self.steps as i32,
                    "guidance": self.cfg,
                    "seed": seed,
                    "scheduler": self.scheduler.id(),
                    "adapters": adapters,
                    "priority": "normal",
                    "metadata": {},
                });
                if let Some(sg) = structural_guide {
                    payload["structural_guide"] = sg;
                }
                ("/api/images/generations", payload)
            }
            ImageMode::ReferenceImage => {
                let mut payload = json!({
                    "model": self.model.id(),
                    "operation": "rewrite",
                    "source_asset_id": source_asset_id,
                    "title": self.title,
                    "prompt": self.prompt.text(),
                    "negative_prompt": self.negative_prompt.text(),
                    "source_fidelity": 1.0 - self.reference_strength.value(),
                    "rewrite_mode": "reference",
                    "n": n,
                    "steps": self.steps as i32,
                    "guidance": self.cfg,
                    "seed": seed,
                    "scheduler": self.scheduler.id(),
                    "adapters": adapters,
                    "priority": "normal",
                    "metadata": {},
                });
                if let Some(sg) = structural_guide {
                    payload["structural_guide"] = sg;
                }
                ("/api/images/edits", payload)
            }
            ImageMode::EditByDescription => {
                let mut payload = json!({
                    "model": self.model.id(),
                    "operation": "rewrite",
                    "source_asset_id": source_asset_id,
                    "title": self.title,
                    "prompt": self.prompt.text(),
                    "negative_prompt": self.negative_prompt.text(),
                    "source_fidelity": self.denoise.value(),
                    "rewrite_mode": "instruct",
                    "n": n,
                    "steps": self.steps as i32,
                    "guidance": self.cfg,
                    "seed": seed,
                    "scheduler": self.scheduler.id(),
                    "adapters": adapters,
                    "priority": "normal",
                    "metadata": {},
                });
                if let Some(sg) = structural_guide {
                    payload["structural_guide"] = sg;
                }
                ("/api/images/edits", payload)
            }
            ImageMode::Inpainting => {
                let mut payload = json!({
                    "model": self.model.id(),
                    "operation": "retouch",
                    "source_asset_id": source_asset_id,
                    "mask_asset_id": mask_asset_id,
                    "title": self.title,
                    "prompt": self.prompt.text(),
                    "negative_prompt": self.negative_prompt.text(),
                    "source_fidelity": self.denoise.value(),
                    "n": n,
                    "steps": self.steps as i32,
                    "guidance": self.cfg,
                    "seed": seed,
                    "scheduler": self.scheduler.id(),
                    "adapters": adapters,
                    "priority": "normal",
                    "metadata": {},
                });
                if let Some(sg) = structural_guide {
                    payload["structural_guide"] = sg;
                }
                ("/api/images/edits", payload)
            }
            ImageMode::Outpainting => {
                let mut payload = json!({
                    "model": self.model.id(),
                    "operation": "extend",
                    "source_asset_id": source_asset_id,
                    "title": self.title,
                    "prompt": self.prompt.text(),
                    "negative_prompt": self.negative_prompt.text(),
                    "extend": {
                        "directions": self.outpaint_direction.ids(),
                        "pixels": self.outpaint_pixels as u32,
                    },
                    "n": n,
                    "steps": self.steps as i32,
                    "guidance": self.cfg,
                    "seed": seed,
                    "scheduler": self.scheduler.id(),
                    "adapters": adapters,
                    "priority": "normal",
                    "metadata": {},
                });
                if let Some(sg) = structural_guide {
                    payload["structural_guide"] = sg;
                }
                ("/api/images/edits", payload)
            }
            ImageMode::Upscale => {
                let payload = json!({
                    "model": self.model.id(),
                    "source_asset_id": source_asset_id,
                    "scale": self.upscale_factor.value(),
                    "denoise": self.denoise.value(),
                    "tile_size": 1024,
                    "priority": "normal",
                    "metadata": {},
                });
                ("/api/images/upscales", payload)
            }
        }
    }

    /// Legacy helper — builds a generic JSON value (used for preview / logging).
    pub fn build_generation_request(&self) -> serde_json::Value {
        let (_, payload) = self.build_request(None, None, None);
        payload
    }

    pub fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::ModeChanged(m) => {
                self.mode = m;
                self.rebuild_filtered_models();
                // When mode changes, ensure current model supports the new mode's action
                let required_action = ModelOption::mode_required_action(m);
                if !self.model.supports_action(required_action) {
                    // Find first model that supports this action
                    let first_valid = self.filtered_models.first().cloned();
                    if let Some(model) = first_valid {
                        self.model = model.clone();
                        self.steps = model.default_steps();
                        self.cfg = model.default_cfg();
                        self.steps_input = format!("{:.0}", self.steps);
                        self.cfg_input = format!("{:.1}", self.cfg);
                    }
                }
                iced::Task::none()
            }
            Message::ModelSelected(m) => {
                self.model = m;
                self.steps = self.model.default_steps();
                self.cfg = self.model.default_cfg();
                self.params_dirty = false;
                self.push_log(format!("已切换模型：{}", self.model.label()));
                iced::Task::none()
            }
            Message::PresetSelected(p) => {
                self.preset = p;
                self.push_log(format!("已加载预设：{}", self.preset.label()));
                iced::Task::none()
            }
            Message::TitleChanged(s) => {
                self.title = s;
                iced::Task::none()
            }
            Message::PromptEdited(action) => {
                self.prompt.perform(action);
                truncate_editor(&mut self.prompt, PROMPT_MAX);
                self.validation_error = None;
                iced::Task::none()
            }
            Message::NegativePromptEdited(action) => {
                self.negative_prompt.perform(action);
                truncate_editor(&mut self.negative_prompt, NEGATIVE_MAX);
                iced::Task::none()
            }
            Message::ToggleNegativeOpen => {
                self.negative_open = !self.negative_open;
                iced::Task::none()
            }
            Message::ToggleAdvancedOpen => {
                self.advanced_open = !self.advanced_open;
                iced::Task::none()
            }
            Message::SeedChanged(s) => {
                self.seed = s;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::RandomizeSeed => {
                let random_seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                self.seed = format!("{}", random_seed);
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::StepsChanged(v) => {
                self.steps = v.clamp(1.0, 50.0);
                self.steps_input = format!("{:.0}", self.steps);
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::StepsInputChanged(s) => {
                self.steps_input = s;
                if let Ok(v) = self.steps_input.parse::<f32>() {
                    self.steps = v.clamp(1.0, 50.0);
                    self.params_dirty = true;
                }
                iced::Task::none()
            }
            Message::CfgChanged(v) => {
                self.cfg = v.clamp(0.0, 20.0);
                self.cfg_input = format!("{:.1}", self.cfg);
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::CfgInputChanged(s) => {
                self.cfg_input = s;
                if let Ok(v) = self.cfg_input.parse::<f32>() {
                    self.cfg = v.clamp(0.0, 20.0);
                    self.params_dirty = true;
                }
                iced::Task::none()
            }
            Message::SchedulerSelected(s) => {
                self.scheduler = s;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::WidthSelected(w) => {
                self.width = w;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::HeightSelected(h) => {
                self.height = h;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::LoraSelected(l) => {
                self.lora = l;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::LoraWeightChanged(v) => {
                self.lora_weight = v.clamp(0.0, 2.0);
                self.lora_weight_input = format!("{:.2}", self.lora_weight);
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::LoraWeightInputChanged(s) => {
                let v: Option<f32> = s.parse().ok();
                self.lora_weight_input = s;
                if let Some(v) = v { self.lora_weight = v.clamp(0.0, 2.0); }
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::BatchSizeSelected(b) => {
                self.batch_size = b;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::BatchCountSelected(b) => {
                self.batch_count = b;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::ReferenceStrengthSelected(r) => {
                self.reference_strength = r;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::UpscaleFactorSelected(u) => {
                self.upscale_factor = u;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::DenoiseSelected(d) => {
                self.denoise = d;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::OutpaintDirectionSelected(d) => {
                self.outpaint_direction = d;
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::OutpaintPixelsChanged(v) => {
                self.outpaint_pixels = v.clamp(64.0, 512.0);
                self.outpaint_pixels_input = format!("{:.0}", self.outpaint_pixels);
                self.params_dirty = true;
                iced::Task::none()
            }
            Message::OutpaintPixelsInputChanged(s) => {
                self.outpaint_pixels_input = s;
                if let Ok(v) = self.outpaint_pixels_input.parse::<f32>() {
                    self.outpaint_pixels = v.clamp(64.0, 512.0);
                    self.params_dirty = true;
                }
                iced::Task::none()
            }
            Message::CommercialToggled(v) => {
                self.commercial_only = v;
                self.rebuild_filtered_models();
                // When commercial filter changes, ensure current model is still valid
                let required_action = ModelOption::mode_required_action(self.mode);
                if !self.model.supports_action(required_action)
                    || (self.commercial_only && !self.model.commercial_use_allowed())
                {
                    let first_valid = self.filtered_models.first().cloned();
                    if let Some(model) = first_valid {
                        self.model = model.clone();
                        self.steps = model.default_steps();
                        self.cfg = model.default_cfg();
                        self.steps_input = format!("{:.0}", self.steps);
                        self.cfg_input = format!("{:.1}", self.cfg);
                    }
                }
                iced::Task::none()
            }
            Message::RestoreDefaults => {
                self.apply_model_defaults();
                self.params_dirty = false;
                self.push_log("已恢复默认配置".into());
                iced::Task::none()
            }
            Message::LoadPreset => {
                self.push_log(format!("已加载预设：{}", self.preset.label()));
                iced::Task::none()
            }
            Message::ClearLogs => {
                self.logs.clear();
                iced::Task::none()
            }
            Message::Generate => {
                let needs_prompt = !matches!(
                    self.mode,
                    ImageMode::Upscale | ImageMode::ReferenceImage
                );
                if needs_prompt && self.prompt.text().trim().is_empty() {
                    self.validation_error = Some("请输入提示词后再生成".into());
                    return iced::Task::none();
                }
                self.validation_error = None;
                self.generate_state = GenerateState::Submitting;
                self.enhance_offer_visible = false;
                let mode_label = self.mode.label();
                self.push_log(format!("{mode_label} 任务已提交到队列"));
                // Return a task that signals App to call the backend API
                // App will handle the actual HTTP request
                iced::Task::perform(
                    async { tokio::time::sleep(Duration::from_millis(100)).await },
                    |_| Message::GenerateStep,
                )
            }
            Message::CancelGeneration => {
                self.generate_state = GenerateState::Idle;
                self.push_log("已取消生成".into());
                iced::Task::none()
            }
            Message::GenerateStep => {
                // This is now a placeholder — actual API call happens in App
                // App will call backend and then send GenerateProgress / GenerateComplete messages
                match self.generate_state {
                    GenerateState::Submitting => {
                        self.generate_state = GenerateState::Generating {
                            progress: 5,
                            step: 1,
                            total: self.steps as u32,
                            phase: GeneratePhase::Queued,
                        };
                        self.push_log("等待后端处理…".into());
                        iced::Task::none()
                    }
                    _ => iced::Task::none(),
                }
            }
            Message::GenerateProgress { progress, step, total, phase } => {
                self.generate_state = GenerateState::Generating {
                    progress,
                    step,
                    total,
                    phase,
                };
                iced::Task::none()
            }
            Message::GenerateComplete { result_urls } => {
                self.generate_state = GenerateState::Done;
                self.push_recent();
                self.push_log("生成完成".into());
                if let Some(first_url) = result_urls.first() {
                    self.generated_image_path = Some(first_url.clone());
                }
                if matches!(self.model, ModelOption::ZImageTurbo) {
                    self.enhance_offer_visible = true;
                }
                iced::Task::none()
            }
            Message::GenerateFailed { error } => {
                self.generate_state = GenerateState::Idle;
                self.push_log(format!("生成失败: {}", error));
                iced::Task::none()
            }
            // Upload / mask
            Message::UploadSourceImage => {
                iced::Task::perform(
                    async {
                        tokio::task::spawn_blocking(|| {
                            rfd::FileDialog::new()
                                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                                .pick_file()
                        }).await.ok().flatten()
                    },
                    |path| {
                        if let Some(path) = path {
                            Message::SourceImageDropped(path.to_string_lossy().to_string())
                        } else {
                            Message::NoOp
                        }
                    },
                )
            }
            Message::UploadMaskImage => {
                iced::Task::perform(
                    async {
                        tokio::task::spawn_blocking(|| {
                            rfd::FileDialog::new()
                                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                                .pick_file()
                        }).await.ok().flatten()
                    },
                    |path| {
                        if let Some(path) = path {
                            Message::MaskImageSelected(path.to_string_lossy().to_string())
                        } else {
                            Message::NoOp
                        }
                    },
                )
            }
            Message::UploadControlImage => {
                iced::Task::perform(
                    async {
                        tokio::task::spawn_blocking(|| {
                            rfd::FileDialog::new()
                                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                                .pick_file()
                        }).await.ok().flatten()
                    },
                    |path| {
                        if let Some(path) = path {
                            Message::ControlImageSelected(path.to_string_lossy().to_string())
                        } else {
                            Message::NoOp
                        }
                    },
                )
            }
            Message::MaskImageSelected(path) => {
                self.mask_image = SourceImageState::Uploaded(path.clone());
                self.push_log(format!("已选择遮罩图片: {}", path));
                iced::Task::none()
            }
            Message::ControlImageSelected(path) => {
                self.control_image = SourceImageState::Uploaded(path.clone());
                self.push_log(format!("已选择 ControlNet 控制图: {}", path));
                iced::Task::none()
            }
            Message::ClearSourceImage => {
                self.source_image = SourceImageState::Empty;
                self.canvas_editor_state.base_image_path = None;
                iced::Task::none()
            }
            Message::ClearMaskImage => {
                self.mask_image = SourceImageState::Empty;
                iced::Task::none()
            }
            Message::ClearControlImage => {
                self.control_image = SourceImageState::Empty;
                iced::Task::none()
            }
            Message::NoOp => iced::Task::none(),
            Message::ToggleMaskEditor => {
                self.mask_editor_open = !self.mask_editor_open;
                iced::Task::none()
            }
            Message::ToggleMaskDrawMode => {
                self.mask_draw_mode = !self.mask_draw_mode;
                if self.mask_draw_mode {
                    self.canvas_editor_state.tool = CanvasTool::Brush;
                    self.push_log("进入遮罩绘制模式".into());
                } else {
                    self.canvas_editor_state.tool = CanvasTool::Pan;
                }
                iced::Task::none()
            }
            Message::CanvasEditorMsg(msg) => {
                self.canvas_editor_state.update(msg);
                iced::Task::none()
            }
            Message::SourceImageDropped(path) => {
                self.source_image = SourceImageState::Uploaded(path.clone());
                self.canvas_editor_state.base_image_path = Some(path.clone().into());
                self.push_log(format!("已拖入图片: {}", path));
                iced::Task::none()
            }
            Message::BeforeAfterMsg(msg) => {
                // We'll need bounds for the update - for now use a default
                let bounds = iced::Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: 340.0,
                    height: 300.0,
                };
                self.before_after_state.update(msg, bounds);
                iced::Task::none()
            }
            Message::StagingMsg(msg) => {
                match msg {
                    StagingMessage::SelectResult(id) => {
                        for result in &mut self.staged_results {
                            result.selected = result.id == id;
                        }
                    }
                    StagingMessage::DownloadResult(id) => {
                        self.push_log(format!("下载暂存区图片 #{}", id));
                    }
                    StagingMessage::DeleteResult(id) => {
                        self.staged_results.retain(|r| r.id != id);
                    }
                    StagingMessage::ClearAll => {
                        self.staged_results.clear();
                    }
                }
                iced::Task::none()
            }
            Message::SplitViewMsg(msg) => {
                let bounds = iced::Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: 1360.0,
                    height: 880.0,
                };
                self.split_view_state.update(msg, bounds);
                iced::Task::none()
            }
            // ControlNet
            Message::ControlNetSelected(c) => {
                self.controlnet = c;
                if matches!(self.controlnet, ControlNetOption::None) {
                    self.control_image = SourceImageState::Empty;
                }
                iced::Task::none()
            }
            Message::ControlNetStrengthChanged(v) => {
                self.controlnet_strength = v.clamp(0.0, 2.0);
                self.controlnet_strength_input = format!("{:.1}", self.controlnet_strength);
                iced::Task::none()
            }
            // Enhance
            Message::StartEnhance => {
                self.push_log("启动精修增强…".into());
                self.enhance_offer_visible = false;
                self.model = ModelOption::Flux1Dev;
                self.push_log("已切换至 flux1-dev 进行精修".into());
                iced::Task::none()
            }
            Message::ClosePreview => {
                self.preview_open = false;
                iced::Task::none()
            }
            Message::PreviewImage => {
                self.preview_open = true;
                iced::Task::none()
            }
            // Memory poll
            Message::MemoryPoll => {
                use sysinfo::{System, RefreshKind};
                let mut sys = System::new_with_specifics(RefreshKind::everything());
                sys.refresh_memory();
                let total = sys.total_memory() as f32 / 1024.0 / 1024.0 / 1024.0;
                let used = sys.used_memory() as f32 / 1024.0 / 1024.0 / 1024.0;
                self.memory_info.total_gb = total;
                self.memory_info.used_gb = used;
                // mlx_active is approximate — use a fraction of used memory
                self.memory_info.mlx_active_gb = (used * 0.85).clamp(0.0, total);
                iced::Task::none()
            }
        }
    }

    fn apply_model_defaults(&mut self) {
        self.steps = self.model.default_steps();
        self.steps_input = format!("{:.0}", self.steps);
        self.cfg = self.model.default_cfg();
        self.cfg_input = format!("{:.1}", self.cfg);
        self.seed = "随机".into();
        self.scheduler = SchedulerOption::Linear;
        self.width = WidthOption::W768;
        self.height = HeightOption::H1280;
        self.lora = LoraOption::None;
        self.batch_size = BatchSizeOption::One;
        self.batch_count = BatchCountOption::One;
        self.reference_strength = ReferenceStrengthOption::Medium;
        self.upscale_factor = UpscaleFactorOption::X2;
        self.denoise = DenoiseOption::Medium;
        self.outpaint_direction = OutpaintDirectionOption::All;
        self.outpaint_pixels = 256.0;
        self.outpaint_pixels_input = "256".into();
    }

    /// Rebuild filtered_models based on current mode and commercial filter.
    /// Matches web UI logic: imageModelRow (media=image, category!=loras) + action filter + commercial.
    pub fn rebuild_filtered_models(&mut self) {
        let required_action = ModelOption::mode_required_action(self.mode);
        self.filtered_models = if self.available_models.is_empty() {
            // Fallback to static models when backend not connected
            ModelOption::STATIC_ALL
                .iter()
                .filter(|m| m.supports_action(required_action))
                .cloned()
                .collect()
        } else {
            self.available_models
                .iter()
                .filter(|m| {
                    // Web UI imageModelRow: exclude only loras
                    let category = m.category();
                    if category == "loras" {
                        return false;
                    }
                    // Must support current mode's action
                    if !m.supports_action(required_action) {
                        return false;
                    }
                    // Respect commercial filter
                    if self.commercial_only && !m.commercial_use_allowed() {
                        return false;
                    }
                    true
                })
                .cloned()
                .collect()
        };
    }

    pub fn push_log(&mut self, message: String) {
        let now = chrono::Local::now();
        let time = now.format("%H:%M:%S").to_string();
        self.logs.push(LogLine {
            time,
            message,
        });
        if self.logs.len() > 20 {
            self.logs.remove(0);
        }
    }

    pub fn push_recent(&mut self) {
        let title = if self.title.trim().is_empty() {
            match self.mode {
                ImageMode::TextToImage => self.prompt.text().chars().take(24).collect::<String>(),
                _ => format!("{} · {}", self.mode.label(), self.model.id()),
            }
        } else {
            self.title.clone()
        };
        let subtitle = format!(
            "{} · {}×{} · {} · seed {}",
            self.model.id(),
            self.width(),
            self.height(),
            self.mode.label(),
            self.seed
        );
        let prompt_snippet = if self.prompt.text().trim().is_empty() {
            match self.mode {
                ImageMode::ReferenceImage => format!("参考强度 {:.1}", self.reference_strength.value()),
                ImageMode::Upscale => format!("{} 放大 · 降噪 {:?}", self.upscale_factor.label(), self.denoise.value()),
                ImageMode::Outpainting => format!("{:?} 扩展 {:.0}px", self.outpaint_direction.label(), self.outpaint_pixels),
                ImageMode::Inpainting => "局部重绘".into(),
                ImageMode::EditByDescription => "按描述改图".into(),
                _ => self.prompt.text().chars().take(48).collect::<String>(),
            }
        } else {
            self.prompt.text().chars().take(48).collect::<String>()
        };
        self.recent_generations.insert(
            0,
            RecentGeneration {
                title,
                subtitle,
                prompt_snippet,
            },
        );
        if self.recent_generations.len() > 8 {
            self.recent_generations.truncate(8);
        }
    }

    /// Full workspace: mode tabs + two-column layout (params | right panel).
    /// Returns (tabs, left_panel) — right panel is assembled by app.rs.
    /// left_panel includes the preview dialog overlay when open.
    pub fn workspace_view(&self) -> (Element<'_, Message>, Element<'_, Message>) {
        let tabs = self.mode_tabs();

        let left_panel = match self.mode {
            ImageMode::TextToImage => self.text_to_image_params(),
            ImageMode::ReferenceImage => self.reference_image_params(),
            ImageMode::EditByDescription => self.edit_image_params(),
            ImageMode::Inpainting => self.inpaint_params(),
            ImageMode::Outpainting => self.outpaint_params(),
            ImageMode::Upscale => self.upscale_params(),
        };

        // Wrap left_panel with preview dialog overlay if open
        let left_panel = if self.preview_open {
            let dialog = self.preview_dialog();
            container(column![left_panel, dialog].spacing(0).width(Length::Fill).height(Length::Fill))
                .width(Length::Fill).height(Length::Fill).into()
        } else {
            left_panel
        };

        (tabs, left_panel)
    }

    fn preview_dialog(&self) -> Element<'_, Message> {
        let path = self.generated_image_path.as_ref()
            .or_else(|| match &self.source_image { SourceImageState::Uploaded(p) => Some(p), _ => None });

        let content: Element<Message> = if let Some(path_str) = path {
            let p = std::path::Path::new(path_str);
            let img_el: Element<Message> = container(iced::widget::image(iced::widget::image::Handle::from_path(p)))
                .width(Length::Fill).height(Length::Fill).align_x(Alignment::Center).align_y(Alignment::Center).into();
            column![
                row![
                    text("预览").size(typography::TITLE).color(color::TEXT_PRIMARY),
                    Space::new().width(Length::Fill),
                    iced::widget::button(dq_components::phosphor_icon(PhosphorIcon::X, 14.0, color::TEXT_SECONDARY))
                        .on_press(Message::ClosePreview)
                        .style(|_theme: &iced::Theme, _status| iced::widget::button::Style { background: None, ..Default::default() }),
                ]
                .spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill),
                img_el,
            ]
            .spacing(spacing::MD).width(Length::Fill).height(Length::Fill).into()
        } else {
            text("暂无预览").size(typography::BODY).color(color::TEXT_SECONDARY).into()
        };

        container(content)
            .width(Length::Fill).height(Length::Fill).padding(spacing::LG)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(color::BG_OVERLAY)),
                ..Default::default()
            })
            .into()
    }

    fn mode_tabs(&self) -> Element<'_, Message> {
        use iced::widget::container;
        use dq_theme::mode_tabs_container;

        container(dq_mode_tabs(&self.mode_tabs, &self.mode, Message::ModeChanged))
            .padding([spacing::SM, spacing::MD])
            .width(Length::Fill)
            .style(mode_tabs_container)
            .into()
    }

    // ===== Left Panel Params (per mode) =====

    fn text_to_image_params(&self) -> Element<'_, Message> {
        column![
            self.model_section(),
            self.title_section(),
            self.prompt_section(),
            self.controlnet_section(),
            self.advanced_section(),
            self.generate_and_log_section(),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill)
        .into()
    }

    fn reference_image_params(&self) -> Element<'_, Message> {
        use iced::widget::row;

        column![
            self.model_section(),
            self.unified_image_hub("参考图片"),
            row![
                container(text("参考强度").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                dq_pick_list(
                    ReferenceStrengthOption::ALL,
                    Some(&self.reference_strength),
                    Message::ReferenceStrengthSelected,
                    "选择参考强度",
                ),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill),
            self.title_section(),
            self.prompt_section(),
            self.controlnet_section(),
            self.advanced_section(),
            self.generate_and_log_section(),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill)
        .into()
    }

    fn edit_image_params(&self) -> Element<'_, Message> {
        column![
            self.model_section(),
            self.title_section(),
            self.unified_image_hub("源图片"),
            self.prompt_section(),
            self.controlnet_section(),
            self.advanced_section(),
            self.generate_and_log_section(),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill)
        .into()
    }

    fn inpaint_params(&self) -> Element<'_, Message> {
        let mask_section: Element<'_, Message> = if self.mask_draw_mode {
            column![
                row![
                    tool_selector(
                        self.canvas_editor_state.tool,
                        |tool| Message::CanvasEditorMsg(CanvasEditorMessage::SetTool(tool)),
                    ),
                    Space::new().width(Length::Fill),
                    dq_control_button(
                        "退出绘制",
                        Some(Message::ToggleMaskDrawMode),
                    ),
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                container(
                    canvas_editor(
                        &self.canvas_editor_state,
                        |msg| Message::CanvasEditorMsg(msg),
                    )
                )
                .width(Length::Fill)
                .height(Length::Fixed(300.0))
                .style(|_theme| iced::widget::container::Style {
                    background: Some(iced::Background::Color(color::BG_INSET)),
                    border: iced::Border {
                        color: color::BORDER_SUBTLE,
                        width: 1.0,
                        radius: spacing::RADIUS_MD.into(),
                    },
                    ..Default::default()
                }),
                row![
                    brush_size_selector(
                        self.canvas_editor_state.brush_size,
                        |size| Message::CanvasEditorMsg(CanvasEditorMessage::SetBrushSize(size)),
                    ),
                    Space::new().width(Length::Fill),
                    zoom_controls(
                        Message::CanvasEditorMsg(CanvasEditorMessage::ZoomIn),
                        Message::CanvasEditorMsg(CanvasEditorMessage::ZoomOut),
                        Message::CanvasEditorMsg(CanvasEditorMessage::ResetView),
                        Message::CanvasEditorMsg(CanvasEditorMessage::ClearMask),
                    ),
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
            ]
            .spacing(spacing::SM)
            .width(Length::Fill)
            .into()
        } else {
            row![
                container(text("遮罩").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                column![
                    self.upload_area(
                        "点击上传遮罩图片",
                        matches!(self.mask_image, SourceImageState::Uploaded(_)),
                        Message::UploadMaskImage,
                        Message::ClearMaskImage,
                    ),
                    dq_control_button(
                        "或绘制遮罩",
                        Some(Message::ToggleMaskDrawMode),
                    ),
                ]
                .spacing(spacing::XS)
                .width(Length::Fill),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill)
            .into()
        };

        column![
            self.model_section(),
            self.title_section(),
            self.unified_image_hub("原图"),
            mask_section,
            self.prompt_section(),
            self.controlnet_section(),
            self.advanced_section(),
            self.generate_and_log_section(),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill)
        .into()
    }

    fn outpaint_params(&self) -> Element<'_, Message> {
        use iced::widget::row;

        column![
            self.model_section(),
            self.title_section(),
            self.unified_image_hub("原图"),
            row![
                container(text("扩展方向").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                dq_pick_list(
                    OutpaintDirectionOption::ALL,
                    Some(&self.outpaint_direction),
                    Message::OutpaintDirectionSelected,
                    "选择方向",
                ),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill),
            row![
                container(text("扩展像素").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                dq_slider_with_input(
                    64.0..=512.0,
                    64.0,
                    self.outpaint_pixels,
                    &self.outpaint_pixels_input,
                    Message::OutpaintPixelsChanged,
                    Message::OutpaintPixelsInputChanged,
                ),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill),
            self.prompt_section(),
            self.controlnet_section(),
            self.advanced_section(),
            self.generate_and_log_section(),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill)
        .into()
    }

    fn upscale_params(&self) -> Element<'_, Message> {
        use iced::widget::row;

        let mut params = column![
            self.model_section(),
            self.title_section(),
            self.unified_image_hub("源图片"),
            row![
                container(text("放大倍数").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                dq_pick_list(
                    UpscaleFactorOption::ALL,
                    Some(&self.upscale_factor),
                    Message::UpscaleFactorSelected,
                    "选择倍数",
                ),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill),
            row![
                container(text("降噪强度").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                dq_pick_list(
                    DenoiseOption::ALL,
                    Some(&self.denoise),
                    Message::DenoiseSelected,
                    "选择强度",
                ),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill);

        // Show before/after comparison when result is ready
        if matches!(self.generate_state, GenerateState::Done) {
            params = params.push(
                column![
                    before_after_labels("原图", "放大后"),
                    before_after_slider(
                        &self.before_after_state,
                        "原图",
                        "放大后",
                        |msg| Message::BeforeAfterMsg(msg),
                    ),
                ]
                .spacing(spacing::XS)
                .width(Length::Fill)
            );
        }

        params = params.push(self.prompt_section());
        params = params.push(self.controlnet_section());
        params = params.push(self.advanced_section());
        params = params.push(self.generate_and_log_section());

        params.into()
    }

    fn model_not_ready_alert(&self) -> Option<Element<'_, Message>> {
        if !self.model.installed() {
            Some(dq_components::alert(
                "该模型尚未下载，请先在模型库中安装",
                dq_components::AlertType::Warning,
                None,
            ))
        } else {
            None
        }
    }

    fn model_section(&self) -> Element<'_, Message> {
        let required_action = ModelOption::mode_required_action(self.mode);
        let display_models: &[ModelOption] = if !self.filtered_models.is_empty() {
            &self.filtered_models
        } else if !self.available_models.is_empty() {
            &self.available_models
        } else {
            ModelOption::STATIC_ALL
        };

        let selected = if display_models.iter().any(|m| m == &self.model) {
            Some(&self.model)
        } else {
            display_models.iter().find(|m| m.supports_action(required_action))
                .or_else(|| display_models.first())
        };

        let body = row![
            container(dq_pick_list(display_models, selected, Message::ModelSelected, "选择模型与版本")).width(Length::Fill),
            container(checkbox(self.commercial_only).label("仅可商用").text_size(typography::LABEL).spacing(6.0).on_toggle(Message::CommercialToggled))
                .height(Length::Fixed(spacing::CONTROL_HEIGHT)).align_y(Alignment::Center),
        ].spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill).into();

        let mut col = column![section_card("模型", body)].spacing(spacing::SM).width(Length::Fill);
        if !self.model.installed() {
            col = col.push(dq_components::alert("该模型尚未下载，请先在模型库中安装", dq_components::AlertType::Warning, None));
        }
        col.into()
    }

    fn title_section(&self) -> Element<'_, Message> {
        // Single row, no label — panel title already says "标题"
        let body = dq_text_input(
            "给这次创作起个名字（可选）",
            &self.title,
            Message::TitleChanged,
        );

        section_card("标题", body)
    }

    fn prompt_section(&self) -> Element<'_, Message> {
        let preset_row = dq_prompt_preset_row(
            dq_pick_list(
                PresetOption::ALL,
                Some(&self.preset),
                Message::PresetSelected,
                "预设",
            ),
            dq_control_button("加载", Some(Message::LoadPreset)),
        );

        let mut body = column![
            preset_row,
            dq_prompt_editor(
                column![].into(),
                "描述你想要生成的图像…",
                &self.prompt,
                Message::PromptEdited,
                format!("{}/{}", editor_char_count(&self.prompt), PROMPT_MAX),
            ),
        ]
        .spacing(spacing::SM)
        .width(Length::Fill);

        // Negative prompt (collapsible, inside prompt panel)
        let negative_trailing = Some(
            iced::widget::button(chevron_icon::<Message>(!self.negative_open))
                .on_press(Message::ToggleNegativeOpen)
                .style(|_theme, _status| iced::widget::button::Style {
                    background: None,
                    ..Default::default()
                })
                .into()
        );

        if self.negative_open {
            body = body.push(
                dq_text_input_multiline(
                    "描述你不需要的内容…",
                    &self.negative_prompt,
                    Message::NegativePromptEdited,
                    format!(
                        "{}/{}",
                        editor_char_count(&self.negative_prompt),
                        NEGATIVE_MAX
                    ),
                )
            );
        }

        surface_card(Some(text("提示词").size(typography::TITLE).color(color::TEXT_PRIMARY).into()), negative_trailing, Some(body.into()))
    }

    fn advanced_section(&self) -> Element<'_, Message> {
        let trailing = Some(
            iced::widget::button(chevron_icon::<Message>(!self.advanced_open))
                .on_press(Message::ToggleAdvancedOpen)
                .style(|_theme, _status| iced::widget::button::Style {
                    background: None,
                    ..Default::default()
                })
                .into()
        );

        let body = if self.advanced_open {
            let steps_control = dq_slider_with_input(
                1.0..=50.0,
                1.0,
                self.steps,
                &self.steps_input,
                Message::StepsChanged,
                Message::StepsInputChanged,
            );

            let cfg_control = dq_slider_with_input(
                0.0..=20.0,
                0.5,
                self.cfg,
                &self.cfg_input,
                Message::CfgChanged,
                Message::CfgInputChanged,
            );

            let resolution_row = row![
                container(dq_pick_list(
                    WidthOption::ALL,
                    Some(&self.width),
                    Message::WidthSelected,
                    "宽",
                ))
                .width(Length::FillPortion(1)),
                container(
                    text("×")
                        .size(typography::BODY)
                        .color(color::TEXT_TERTIARY),
                )
                .align_y(Alignment::Center),
                container(dq_pick_list(
                    HeightOption::ALL,
                    Some(&self.height),
                    Message::HeightSelected,
                    "高",
                ))
                .width(Length::FillPortion(1)),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill);

            let seed_row = row![
                container(dq_text_input("随机", &self.seed, Message::SeedChanged)).width(Length::Fill),
                phosphor_icon_button(PhosphorIcon::ArrowsClockwise, 14.0, Some(Message::RandomizeSeed)),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill);

            let batch_row = row![
                container(dq_pick_list(
                    BatchSizeOption::ALL,
                    Some(&self.batch_size),
                    Message::BatchSizeSelected,
                    "单批数量",
                ))
                .width(Length::FillPortion(1)),
                container(
                    text("×")
                        .size(typography::BODY)
                        .color(color::TEXT_TERTIARY),
                )
                .align_y(Alignment::Center),
                container(dq_pick_list(
                    BatchCountOption::ALL,
                    Some(&self.batch_count),
                    Message::BatchCountSelected,
                    "批次数",
                ))
                .width(Length::FillPortion(1)),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill);

            let advanced_body = column![
                row![
                    container(text("步数").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    steps_control,
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                row![
                    container(text("CFG").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    cfg_control,
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                row![
                    container(text("调度器").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    dq_pick_list(
                        SchedulerOption::ALL,
                        Some(&self.scheduler),
                        Message::SchedulerSelected,
                        "选择调度器",
                    ),
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                row![
                    container(text("分辨率").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    resolution_row,
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                row![
                    container(text("种子").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    seed_row,
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                row![
                    container(text("数量").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    batch_row,
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                row![
                    container(text("LoRA").size(typography::LABEL))
                        .width(Length::Fixed(80.0))
                        .align_y(Alignment::Center),
                    dq_pick_list(
                        LoraOption::ALL,
                        Some(&self.lora),
                        Message::LoraSelected,
                        "选择 LoRA",
                    ),
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
                if !matches!(self.lora, LoraOption::None) {
                    let r: Element<Message> = row![
                        container(text("权重").size(typography::LABEL))
                            .width(Length::Fixed(80.0))
                            .align_y(Alignment::Center),
                        dq_components::dq_slider_with_input(
                            0.0..=2.0, 0.05, self.lora_weight, &self.lora_weight_input,
                            |v| Message::LoraWeightChanged(v),
                            |s| Message::LoraWeightInputChanged(s),
                        ),
                    ]
                    .spacing(spacing::SM).align_y(Alignment::Center).width(Length::Fill).into();
                    r
                } else { Space::new().height(0).into() },
                container(dq_header_button("恢复默认配置", Some(Message::RestoreDefaults)))
                    .width(Length::Fill)
                    .align_x(Alignment::Start),
            ]
            .spacing(spacing::SM)
            .width(Length::Fill)
            .into();

            Some(advanced_body)
        } else {
            None
        };

        surface_card(Some(text("高级参数").size(typography::TITLE).color(color::TEXT_PRIMARY).into()), trailing, body)
    }

    pub fn generate_and_log_section(&self) -> Element<'_, Message> {
        use dq_components::dq_progress_bar;

        let mut section = column![].spacing(spacing::SM).width(Length::Fill);

        // Primary CTA
        let (label, show_cancel) = match self.generate_state {
            GenerateState::Submitting => ("提交中…", false),
            GenerateState::Generating { .. } => ("生成中…", true),
            GenerateState::Done => ("再次生成", false),
            GenerateState::Idle => ("生成", false),
        };
        let on_press = if matches!(self.generate_state, GenerateState::Generating { .. } | GenerateState::Submitting) {
            None
        } else {
            Some(Message::Generate)
        };
        section = section.push(row![
            dq_primary_action(label, on_press),
            if show_cancel {
                dq_components::dq_button("取消", dq_components::ButtonVariant::Ghost, dq_components::ButtonSize::Md, dq_components::ButtonWidth::Hug, Some(Message::CancelGeneration))
            } else { Space::new().width(0).into() },
        ].spacing(spacing::SM).width(Length::Fill));

        // Progress / status
        if let Some(err) = &self.validation_error {
            section = section.push(
                text(err.as_str())
                    .size(typography::CAPTION)
                    .color(color::DANGER),
            );
        }

        match self.generate_state {
            GenerateState::Submitting => {
                section = section.push(
                    row![
                        iced::widget::text("⟳").size(typography::BODY).color(color::WARNING),
                        text("正在提交任务…")
                            .size(typography::CAPTION)
                            .color(color::TEXT_SECONDARY),
                    ]
                    .spacing(spacing::SM)
                    .align_y(Alignment::Center),
                );
            }
            GenerateState::Generating { progress, step, total, phase } => {
                let phase_label = match phase {
                    GeneratePhase::Queued => "排队中",
                    GeneratePhase::Denoising => "去噪中",
                    GeneratePhase::Decoding => "解码中",
                    GeneratePhase::PostProcessing => "后处理",
                };
                let status = if total > 0 {
                    format!("Step {}/{} · {}", step, total, phase_label)
                } else {
                    phase_label.into()
                };
                section = section.push(dq_progress_bar(progress as f32 / 100.0, 4.0));
                section = section.push(
                    row![
                        text(status)
                            .size(typography::CAPTION)
                            .color(color::TEXT_SECONDARY),
                        Space::new().width(Length::Fill),
                        text(format!("{}%", progress))
                            .size(typography::CAPTION)
                            .color(color::TEXT_TERTIARY),
                    ]
                    .align_y(Alignment::Center)
                    .width(Length::Fill),
                );
            }
            GenerateState::Done => {
                section = section.push(
                    row![
                        text("✓ 已完成 · 已加入最近生成")
                            .size(typography::CAPTION)
                            .color(color::SUCCESS),
                    ]
                    .align_y(Alignment::Center)
                    .width(Length::Fill),
                );
            }
            GenerateState::Idle => {}
        }

        // Shortcut hint
        section = section.push(
            row![
                text(self.queue_hint_idle())
                    .size(typography::CAPTION)
                    .color(color::TEXT_TERTIARY),
                Space::new().width(Length::Fill),
                text("⌘ + Enter 提交")
                    .size(typography::MINI)
                    .color(color::TEXT_QUATERNARY),
            ]
            .align_y(Alignment::Center)
            .width(Length::Fill),
        );

        // Logs
        section = section.push(log_panel(&self.logs, Some(Message::ClearLogs)));

        section.into()
    }

    fn queue_hint_idle(&self) -> String {
        match self.generate_state {
            GenerateState::Idle => "队列空闲 · 预计 ~30 秒".into(),
            _ => String::new(),
        }
    }

    // Unified image hub — shared across image-based modes
    fn unified_image_hub<'a>(&self, label: &'a str) -> Element<'a, Message> {
        use std::path::PathBuf;
        
        match &self.source_image {
            SourceImageState::Uploaded(path) => {
                let path_buf = PathBuf::from(path);
                container(
                    column![
                        dq_components::image_viewer(
                            Some(&path_buf),
                            label,
                        ),
                        row![
                            text(label)
                                .size(typography::BODY)
                                .color(color::TEXT_PRIMARY),
                            Space::new().width(Length::Fill),
                            dq_control_button("更换", Some(Message::UploadSourceImage)),
                            dq_control_button("清除", Some(Message::ClearSourceImage)),
                        ]
                        .spacing(spacing::SM)
                        .align_y(Alignment::Center)
                        .width(Length::Fill),
                    ]
                    .spacing(spacing::SM)
                    .width(Length::Fill),
                )
                .padding(spacing::SM)
                .width(Length::Fill)
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
            SourceImageState::Empty => {
                let upload_label = format!("点击上传{}", label);
                self.upload_area(
                    upload_label.leak(),
                    false,
                    Message::UploadSourceImage,
                    Message::ClearSourceImage,
                )
            }
        }
    }

    // ControlNet top-level section
    fn controlnet_section(&self) -> Element<'_, Message> {
        let body: Element<'_, Message> = column![
            row![
                container(text("控制类型").size(typography::LABEL))
                    .width(Length::Fixed(80.0))
                    .align_y(Alignment::Center),
                dq_pick_list(
                    ControlNetOption::ALL,
                    Some(&self.controlnet),
                    Message::ControlNetSelected,
                    "选择控制类型",
                ),
            ]
            .spacing(spacing::SM)
            .align_y(Alignment::Center)
            .width(Length::Fill),
            if !matches!(self.controlnet, ControlNetOption::None) {
                let active_body: Element<'_, Message> = column![
                    // Control image upload + preview
                    self.control_image_preview(),
                    row![
                        container(text("强度").size(typography::LABEL))
                            .width(Length::Fixed(80.0))
                            .align_y(Alignment::Center),
                        dq_slider_with_input(
                            0.0..=2.0,
                            0.1,
                            self.controlnet_strength,
                            &self.controlnet_strength_input,
                            Message::ControlNetStrengthChanged,
                            Message::CfgInputChanged,
                        ),
                    ]
                    .spacing(spacing::SM)
                    .align_y(Alignment::Center)
                    .width(Length::Fill),
                ]
                .spacing(spacing::SM)
                .width(Length::Fill)
                .into();
                Some(active_body)
            } else {
                None
            },
        ]
        .spacing(spacing::SM)
        .width(Length::Fill)
        .into();

        section_card("ControlNet", body)
    }

    fn control_image_preview(&self) -> Element<'_, Message> {
        use std::path::PathBuf;

        match &self.control_image {
            SourceImageState::Uploaded(path) => {
                let path_buf = PathBuf::from(path);
                container(
                    column![
                        dq_components::image_viewer(
                            Some(&path_buf),
                            "ControlNet 控制图",
                        ),
                        row![
                            text("ControlNet 控制图")
                                .size(typography::BODY)
                                .color(color::TEXT_PRIMARY),
                            Space::new().width(Length::Fill),
                            dq_control_button("更换", Some(Message::UploadControlImage)),
                            dq_control_button("清除", Some(Message::ClearControlImage)),
                        ]
                        .spacing(spacing::SM)
                        .align_y(Alignment::Center)
                        .width(Length::Fill),
                    ]
                    .spacing(spacing::SM)
                    .width(Length::Fill),
                )
                .padding(spacing::SM)
                .width(Length::Fill)
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
            SourceImageState::Empty => {
                self.upload_area(
                    "点击上传 ControlNet 控制图",
                    false,
                    Message::UploadControlImage,
                    Message::ClearControlImage,
                )
            }
        }
    }

    // Upload area helper
    fn upload_area<'a>(&self, label: &'a str, uploaded: bool, _on_upload: Message, on_clear: Message) -> Element<'a, Message> {
        use dq_theme::inset_panel;
        let content: Element<'_, Message> = if uploaded {
            column![
                row![
                    text("已上传")
                        .size(typography::BODY)
                        .color(color::SUCCESS),
                    Space::new().width(Length::Fill),
                    iced::widget::button(
                        text("清除")
                            .size(typography::CAPTION)
                            .color(color::DANGER),
                    )
                    .on_press(on_clear)
                    .style(|_theme: &iced::Theme, _status| iced::widget::button::Style {
                        background: None,
                        ..Default::default()
                    }),
                ]
                .spacing(spacing::SM)
                .align_y(Alignment::Center)
                .width(Length::Fill),
            ]
            .spacing(spacing::SM)
            .width(Length::Fill)
            .into()
        } else {
            column![
                image_placeholder_icon(),
                text(label)
                    .size(typography::CAPTION)
                    .color(color::TEXT_TERTIARY),
            ]
            .spacing(spacing::SM)
            .align_x(Alignment::Center)
            .into()
        };

        container(content)
            .width(Length::Fill)
            .padding(spacing::LG)
            .center_x(Length::Fill)
            .style(inset_panel)
            .into()
    }

}

impl Default for CreatePage {
    fn default() -> Self {
        let model = ModelOption::ZImageTurbo;
        let steps = model.default_steps();
        let cfg = model.default_cfg();
        Self {
            mode: ImageMode::TextToImage,
            mode_tabs: ImageMode::ALL
                .iter()
                .map(|mode| ModeTabOption {
                    label: mode.label().into(),
                    value: *mode,
                })
                .collect(),
            model,
            preset: PresetOption::Photo,
            width: WidthOption::W768,
            height: HeightOption::H1280,
            scheduler: SchedulerOption::Linear,
            lora: LoraOption::None,
            batch_size: BatchSizeOption::One,
            batch_count: BatchCountOption::One,
            title: String::new(),
            prompt: text_editor::Content::with_text(
                "黄昏时分的宁静山景，山谷薄雾，电影感光影",
            ),
            negative_prompt: text_editor::Content::with_text("模糊，低质量，过饱和"),
            negative_open: false,
            advanced_open: false,
            seed: "随机".into(),
            steps,
            steps_input: format!("{:.0}", steps),
            cfg,
            cfg_input: format!("{:.1}", cfg),
            commercial_only: false,
            params_dirty: false,
            generate_state: GenerateState::Idle,
            validation_error: None,
            logs: default_logs(),
            recent_generations: vec![
                RecentGeneration {
                    title: "日系肖像".into(),
                    subtitle: "z-image-turbo · 768×1280 · seed 1847293".into(),
                    prompt_snippet: "1girl, light brown hair, soft lighting, portrait".into(),
                },
                RecentGeneration {
                    title: "赛博朋克城市".into(),
                    subtitle: "flux1-dev · 1024×1024 · seed 892341".into(),
                    prompt_snippet: "cyberpunk city, neon lights, rain, night".into(),
                },
                RecentGeneration {
                    title: "精修放大".into(),
                    subtitle: "z-image-turbo · 1536×2048 · 2× 放大".into(),
                    prompt_snippet: "2× 放大 · 降噪 0.4".into(),
                },
                RecentGeneration {
                    title: "参考原图生成".into(),
                    subtitle: "flux2-klein · 768×1280 · 参考强度 0.7".into(),
                    prompt_snippet: "参考强度 0.7".into(),
                },
            ],
            reference_strength: ReferenceStrengthOption::Medium,
            upscale_factor: UpscaleFactorOption::X2,
            denoise: DenoiseOption::Medium,
            outpaint_direction: OutpaintDirectionOption::All,
            outpaint_pixels: 256.0,
            outpaint_pixels_input: "256".into(),
            // Upload state
            source_image: SourceImageState::Empty,
            mask_image: SourceImageState::Empty,
            control_image: SourceImageState::Empty,
            mask_editor_open: false,
            mask_draw_mode: false,
            canvas_editor_state: CanvasEditorState::new(),
            preview_open: false,
            lora_weight: 0.8,
            lora_weight_input: "0.8".into(),
            // ControlNet
            controlnet: ControlNetOption::None,
            controlnet_strength: 0.8,
            controlnet_strength_input: "0.8".into(),
            // Memory
            memory_info: MemoryInfo::default(),
            // Enhance
            enhance_offer_visible: false,
            // Before/After
            before_after_state: BeforeAfterState::new(),
            // Staging area
            staged_results: vec![
                StagedResult { id: 1, title: "生成 1".into(), selected: false },
                StagedResult { id: 2, title: "生成 2".into(), selected: true },
                StagedResult { id: 3, title: "生成 3".into(), selected: false },
            ],
            // Split view
            split_view_state: SplitViewState::new(),
            // Generated result
            generated_image_path: None,
            // Dynamic models from backend
            available_models: Vec::new(),
            filtered_models: Vec::new(),
        }
    }
}
