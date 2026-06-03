//! API interface definitions for Danqing Studio — aligned with backend contracts.
//! See backend/core/contracts.py for the source of truth.
//!
//! These types are kept as reference documentation matching the backend contracts.
//! Runtime API calls use dynamic serde_json::Value instead.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ------------------------------------------------------------------
// Shared types
// ------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterRef {
    pub id: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralGuide {
    pub asset_id: String,
    #[serde(rename = "type")]
    pub guide_type: String, // "canny" | "depth" | "pose" | "redux"
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleGuide {
    pub asset_id: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendSpec {
    pub directions: Vec<String>, // ["top", "bottom", "left", "right"]
    pub pixels: u32,
}

// ------------------------------------------------------------------
// Image generation / edit / upscale requests
// ------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub title: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub negative_prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_n")]
    pub n: i32,
    pub steps: Option<i32>,
    pub guidance: Option<f32>,
    pub seed: Option<i64>,
    pub scheduler: Option<String>,
    #[serde(default)]
    pub adapters: Vec<AdapterRef>,
    pub structural_guide: Option<StructuralGuide>,
    pub style_guide: Option<StyleGuide>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

fn default_size() -> String { "1024x1024".into() }
fn default_n() -> i32 { 1 }
fn default_priority() -> String { "normal".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEditRequest {
    pub model: String,
    pub operation: String, // "rewrite" | "retouch" | "extend"
    pub source_asset_id: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub title: String,
    pub prompt: String,
    #[serde(default = "default_source_fidelity")]
    pub source_fidelity: f32,
    pub mask_asset_id: Option<String>,
    pub extend: Option<ExtendSpec>,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub negative_prompt: String,
    #[serde(default = "default_n")]
    pub n: i32,
    pub steps: Option<i32>,
    pub seed: Option<i64>,
    pub guidance: Option<f32>,
    pub scheduler: Option<String>,
    #[serde(default)]
    pub adapters: Vec<AdapterRef>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub rewrite_mode: Option<String>, // "reference" | "instruct"
}

fn default_source_fidelity() -> f32 { 0.6 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUpscaleRequest {
    pub model: String,
    pub source_asset_id: String,
    #[serde(default = "default_scale")]
    pub scale: u8,
    #[serde(default = "default_denoise")]
    pub denoise: f32,
    #[serde(default = "default_tile_size")]
    pub tile_size: i32,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

fn default_scale() -> u8 { 2 }
fn default_denoise() -> f32 { 0.0 }
fn default_tile_size() -> i32 { 1024 }

// ------------------------------------------------------------------
// Task / response types
// ------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSubmitResponse {
    pub task: Option<TaskInfo>,
    pub id: Option<String>, // legacy flat id
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    pub id: String,
    pub status: String,
    pub queue_position: Option<u32>,
    pub estimated_wait_seconds: Option<u32>,
    pub progress: Option<f32>,
    pub step: Option<u32>,
    pub total: Option<u32>,
    pub phase: Option<String>,
    pub error: Option<String>,
    pub result: Option<TaskResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub primary_asset_id: Option<String>,
    pub asset_ids: Option<Vec<String>>,
    pub output_paths: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub id: String,
    pub kind: String,
    pub path: String,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub duration_seconds: Option<f32>,
    pub thumbnail_url: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetsListResponse {
    pub items: Vec<Asset>,
    pub total: Option<i64>,
}

// ------------------------------------------------------------------
// Model / registry types
// ------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub family: String,
    pub media: String,
    pub actions: Option<Vec<String>>,
    pub parameters: Option<serde_json::Value>,
    pub backends: Option<Vec<String>>,
    pub recommended: Option<bool>,
    pub commercial_use_allowed: Option<bool>,
    pub versions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryData {
    pub schema_version: i32,
    pub engines: serde_json::Value,
    pub categories: serde_json::Value,
    pub models: serde_json::Value,
    pub _index: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub status: String,
    pub backends: Option<Vec<String>>,
    pub mlx: Option<serde_json::Value>,
    pub cuda: Option<serde_json::Value>,
}

// ------------------------------------------------------------------
// SSE stream event
// ------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub event: String,
    pub data: serde_json::Value,
}

// ------------------------------------------------------------------
// Static helpers
// ------------------------------------------------------------------

impl TaskSubmitResponse {
    pub fn task_id(&self) -> Option<String> {
        if let Some(task) = &self.task {
            if !task.id.is_empty() {
                return Some(task.id.clone());
            }
        }
        self.id.clone()
    }
}
