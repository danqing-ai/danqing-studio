export interface GalleryItem {
  path: string;
  name: string;
  width: number;
  height: number;
  duration_seconds: number | null;
  created_at: string;
  title: string;
  prompt: string;
  model: string;
  thumbnail: string;
  mime_type?: string;
  metadata: Record<string, unknown>;
}

export interface GalleryGroup {
  id: string;
  title: string;
  kind: string;
  asset_count: number;
  preview_assets: GalleryItem[];
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface AssetRow {
  id: string;
  path?: string;
  mime_type?: string;
  metadata?: Record<string, unknown>;
  thumbnail_url?: string;
  duration_seconds?: number | null;
  width?: number;
  height?: number;
  created_at?: string;
  kind?: string;
  parent_asset_id?: string | null;
  relation_type?: string | null;
}

export interface Task {
  id: string;
  kind: string;
  status: string;
  progress?: number;
  step?: number;
  total?: number;
  priority?: number;
  params?: {
    model?: string;
    title?: string;
    prompt?: string;
    [key: string]: unknown;
  };
  estimated_wait_seconds?: number;
  progressMessage?: string;
}

export interface QueueState {
  running: Task[];
  queued: Task[];
}

export interface ModelConfig {
  name: string | { zh?: string; en?: string };
  description?: string | { zh?: string; en?: string };
  name_en?: string;
  description_en?: string;
  successor?: string;
  distilled_from?: string;
  distilled_variant?: string;
  [key: string]: unknown;
}

export interface VersionConfig {
  name?: string | { zh?: string; en?: string };
  [key: string]: unknown;
}

export interface RegistryData {
  engines?: Record<string, unknown>;
  models?: Record<string, ModelConfig>;
  categories?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SystemInfo {
  env_ready: boolean;
  platform: string;
  architecture: string;
  memory_gb: number;
  memory_used_gb?: number;
  memory_available_gb?: number;
  mlx_active_gb?: number;
  mlx_peak_gb?: number;
  mlx_memory_limit: number;
  /** Host can run FLUX ControlNet / Fill (MLX today). From GET /api/settings/system. */
  controlnet_runtime_available?: boolean;
  python_version?: string;
  dependencies?: Record<string, string>;
  [key: string]: unknown;
}

export interface SettingsData {
  language?: string;
  [key: string]: unknown;
}

export type PageKey =
  | 'image_create'
  | 'video_create'
  | 'avatar_create'
  | 'long_video_create'
  | 'audio_create'
  | 'lora_train'
  | 'models'
  | 'prompts'
  | 'assistant'
  | 'settings';

export interface LineageNode {
  id: string;
  kind: string;
  file_path: string;
  thumbnail_path: string | null;
  width: number | null;
  height: number | null;
  created_at: string;
  metadata: Record<string, unknown>;
  relation_type: string | null;
  parent: LineageNode | null;
  children: LineageNode[];
}

export type CanvasLayerRole = 'asset' | 'reference' | 'control';

export interface CanvasItemState {
  x: number;
  y: number;
  scale: number;
  visible: boolean;
  zIndex: number;
  /** User or AI note attached to this canvas node (MindCraft-style describe). */
  note?: string;
  /** Session-local display name (does not rename gallery asset). */
  label?: string;
  /** Layer semantics (Invoke-style); default asset nodes. */
  layerRole?: CanvasLayerRole;
}

export interface CanvasOverlayLayer {
  path: string;
  x: number;
  y: number;
  scale: number;
  opacity: number;
  visible: boolean;
}

export type CanvasOverlayKind =
  | 'reference'
  | 'control'
  | 'start_frame'
  | 'tail_frame'
  | 'video_source'
  | 'cover_source';

/** Ephemeral retouch mask preview (not persisted in canvas session). */
export interface CanvasMaskPreviewState {
  dataUrl: string;
  width: number;
  height: number;
}

export type CanvasExtendDirection = 'top' | 'bottom' | 'left' | 'right';

/** Ephemeral extend-outpaint zone preview on canvas (not persisted). */
export interface CanvasExtendPreviewState {
  directions: CanvasExtendDirection[];
  pixels: number;
}

export interface CanvasOverlaysState {
  reference?: CanvasOverlayLayer | null;
  control?: CanvasOverlayLayer | null;
  start_frame?: CanvasOverlayLayer | null;
  tail_frame?: CanvasOverlayLayer | null;
  video_source?: CanvasOverlayLayer | null;
  cover_source?: CanvasOverlayLayer | null;
}

export interface CanvasViewportState {
  zoom: number;
  panX: number;
  panY: number;
}

export interface CanvasStagingState {
  x: number;
  y: number;
  width: number;
  height: number;
  visible: boolean;
}

export interface CanvasEdge {
  from: string;
  to: string;
  relation: string;
}

export interface CanvasComposerSnapshot {
  prompt?: string;
  title?: string;
  model?: string;
  version?: string;
  negative_prompt?: string;
  seed?: string;
  mode?: string;
  /** Image: reference / control asset paths (`asset:…`). */
  reference_path?: string;
  control_path?: string;
  controlnet?: string;
  controlnet_strength?: string;
  /** Video: animate / upscale source paths. */
  start_image_path?: string;
  tail_image_path?: string;
  source_video_path?: string;
  /** Video: Bernini R2V / RV2V reference image paths. */
  reference_image_paths?: string[];
  /** Audio: cover source asset path. */
  cover_source_path?: string;
  /** Image extend drawer: JSON array of direction strings. */
  extend_directions?: string;
  /** Image extend drawer: pixel padding per side. */
  extend_pixels?: string;
  /** Image editor drawer mode when last persisted. */
  editor_mode?: string;
  /** Asset path under edit in drawer (`asset:…`). */
  edit_asset_path?: string;
  retouch_model_version?: string;
  extend_model_version?: string;
  upscale_model_version?: string;
  upscale_scale?: string;
  upscale_denoise?: string;
  /** Fill retouch/extend: steps independent from text-to-image composer. */
  fill_edit_steps?: string;
  fill_edit_guidance?: string;
}

export type LongVideoChainMode = 'keyframe_only' | 'last_frame' | 'first_last' | 'reference_r2v';

export type LongVideoSegmentRole =
  | 'establishing'
  | 'pre_anchor'
  | 'face_anchor'
  | 'post_anchor'
  | 'keyframe'
  | 'tail_continuation';

export type LongVideoFlfMode = 'none' | 'first_last' | 'continuation';

export interface LongVideoCharacterLook {
  id: string;
  label: string;
  body: string;
  /** Manual vision backfill for keyframe outfit injection (does not replace body). */
  vision_description?: string;
  /** Cast portrait / reference image (gallery asset). */
  reference_asset_id?: string;
  lora_id?: string;
  portrait_prompt?: string;
}

export type LongVideoEditorTab = 'settings' | 'script' | 'cast' | 'scenes' | 'storyboard';

export type LongVideoInspectorTab = 'frame' | 'clip';

export interface LongVideoCharacter {
  id: string;
  name: string;
  looks: LongVideoCharacterLook[];
  default_look_id: string;
}

export interface LongVideoSceneLook {
  id: string;
  label: string;
  body: string;
  /** Manual vision backfill for keyframe environment injection (does not replace body). */
  vision_description?: string;
  /** Scene / set-piece reference image (gallery asset). */
  reference_asset_id?: string;
  /** Cached T2I prompt for scene reference generation only. */
  environment_prompt?: string;
}

export interface LongVideoScene {
  id: string;
  name: string;
  looks: LongVideoSceneLook[];
  default_look_id: string;
  spatial_layout_json?: Record<string, unknown>;
  grounding_panorama_asset_id?: string;
  grounding_depth_asset_id?: string;
}

export interface LongVideoShotSceneLook {
  scene_id: string;
  look_id: string;
}

export interface LongVideoShotCastLook {
  character_id: string;
  look_id: string;
}

export type LongVideoStartFrameMode = 'keyframe' | 'prev_segment_tail' | 'anchor_link';

export type CharacterVisibility = 'invisible' | 'silhouette' | 'partial' | 'full_face';

export type FirstFrameStrategy =
  | 'direct_reuse_portrait'
  | 'scene_composite'
  | 'reuse_prev_tail'
  | 'img2img_light'
  | 't2i_from_grounding'
  | 'causal_generate';

export interface LongVideoShotState {
  id: string;
  order: number;
  visual_prompt: string;
  motion_prompt: string;
  /** Full clip description (video-first parse). */
  video_prompt?: string;
  /** t=0 still derived from video_prompt. */
  start_visual_prompt?: string;
  end_visual_prompt?: string;
  anchor_visual_prompt?: string;
  segment_role: LongVideoSegmentRole;
  start_frame_mode: LongVideoStartFrameMode;
  segment_group_id?: string;
  segment_group_index?: number;
  face_anchor_shot_id?: string;
  flf_mode?: LongVideoFlfMode;
  end_frame_sync_anchor?: boolean;
  end_frame_asset_id?: string;
  /** Scene-only prompt from Expand (without cast reference blocks). */
  scene_prompt?: string;
  /** Per-character outfit selection for this keyframe. */
  cast_looks?: LongVideoShotCastLook[];
  /** Per-shot scene entity / variant binding. */
  scene_look?: LongVideoShotSceneLook;
  keyframe_asset_id?: string;
  /** Optional img2img reference for keyframe generation. */
  reference_asset_id?: string;
  segment_asset_id?: string;
  status?: 'draft' | 'keyframe_ready' | 'segment_ready' | 'failed';
  error?: string;
  seed?: number;
  /** Per-edge I2V chain mode; falls back to project ``chain_mode`` when unset. */
  chain_mode?: LongVideoChainMode;
  /** Per-shot segment length (seconds); defaults to 5 when generating. */
  duration_sec?: number;
  first_frame_visibility?: CharacterVisibility;
  end_visibility?: CharacterVisibility;
  characters_on_screen?: string[];
  clip_start_state?: string;
  clip_end_state?: string;
  /** Inspector / first-frame strategy only — not merged into T2I scene prompt. */
  first_frame_requirement?: string;
  camera_zone_id?: string;
  first_frame_strategy?: FirstFrameStrategy;
  /** Parsed beat location (distinct from full scene_prompt when present). */
  location?: string;
  /** Source beat index in chapter scene_beats. */
  narrative_beat_index?: number;
  shot_size?: string;
}

export type LongVideoSelection =
  | { kind: 'segment'; index: number }
  | { kind: 'clip'; index: number }
  | { kind: 'beat_group'; groupId: string }
  | null;

export interface LongVideoBeatGroup {
  groupId: string;
  beatIndex: number;
  title: string;
  shotIndices: number[];
}

export interface LongVideoChapterScene {
  order: number;
  title?: string;
  beat: string;
}

export interface LongVideoChapterAnalysis {
  synopsis: string;
  mood?: string;
  scene_beats: LongVideoChapterScene[];
  character_anchor?: string;
  style_anchor?: string;
  characters?: LongVideoCharacter[];
  scenes?: LongVideoScene[];
  /** Input-driven parse quality notices (non-blocking). */
  quality_warnings?: string[];
  quality_issues?: Array<{
    code: string;
    message: string;
    severity?: 'warning' | 'critical';
    shot_index?: number | null;
    beat_index?: number | null;
  }>;
  /** Latest script parse run id (prun_*), for project activity lookup. */
  parse_run_id?: string;
  last_parse_at?: string;
  /** Cached parse phase trail from last analyze response. */
  parse_phases?: Array<{ phase: string; message?: string }>;
  /** Latest per-shot T2I assembly provenance (keyed by shot id). */
  shot_t2i_provenance?: Record<string, KeyframeT2iProvenance>;
  /** Rolling parse run snapshots (newest last, max 5). */
  parse_history?: LongVideoParseHistoryEntry[];
}

export type KeyframeT2iProvenanceSkipReason =
  | 'face_anchor'
  | 'close_up'
  | 'token_coverage_sufficient'
  | 'narrative_already_covered'
  | 'empty_narrative';

export type KeyframeFfrProvenanceSkipReason = 'inspector_only' | 'empty_ffr';

export interface KeyframeT2iProvenance {
  narrative_merged: boolean;
  narrative_skip_reason?: KeyframeT2iProvenanceSkipReason;
  narrative_token_coverage?: number;
  /** How shot.location was folded into sceneNarrative (token/overlap rules). */
  location_merge?: 'none' | 'prepended' | 'scene_only';
  /** Always false — FFR is not merged into T2I scene line. */
  first_frame_requirement_merged: boolean;
  ffr_skip_reason?: KeyframeFfrProvenanceSkipReason;
  /** @deprecated FFR no longer merges into T2I; kept for parse-run JSON compat. */
  ffr_clauses_total?: number;
  /** @deprecated FFR no longer merges into T2I; kept for parse-run JSON compat. */
  ffr_clauses_merged?: number;
  scene_parts: Array<{
    source: 'beat_narrative' | 'first_frame_requirement' | 'visual_prompt' | 'location';
    text_preview: string;
  }>;
  composed_scene_line: string;
}

export interface LongVideoParseHistoryEntry {
  parse_run_id: string;
  at: string;
  shot_count: number;
  provenance_by_shot_id: Record<string, KeyframeT2iProvenance>;
}

export interface LongVideoProjectState {
  version: 1 | 2;
  strategy: 'segmented_i2v' | 'latent_extend';
  /** Main editor tab: script source → cast roster → scene roster → storyboard timeline. */
  editor_tab?: LongVideoEditorTab;
  title?: string;
  script_text?: string;
  /** @deprecated use script_text */
  brief?: string;
  /** @deprecated use script_text */
  chapter_text?: string;
  chapter_title?: string;
  chapter_analysis?: LongVideoChapterAnalysis;
  target_duration_sec: number;
  /** Plan-round [Anchor] cached from last storyboard expand; reference only. */
  character_anchor?: string;
  /** Structured cast roster from storyboard expand (multi-look per character). */
  characters?: LongVideoCharacter[];
  /** Structured scene / location roster (multi-variant per place). */
  scenes?: LongVideoScene[];
  style_anchor?: string;
  character_lora_id?: string;
  /** Optional model override for cast portrait generation (defaults to keyframe_model). */
  portrait_model?: string;
  /** LLM for script/chapter parse (defaults to app default_model_llm when empty). */
  script_parse_llm_model?: string;
  keyframe_model: string;
  segment_video_model: string;
  segment_duration_sec: number;
  overlap_frames: number;
  /** Default chain mode for new / unset segment edges. */
  chain_mode: LongVideoChainMode;
  /** Keyframe + segment output size (must match segment model presets, e.g. 1280x704). */
  output_size?: string;
  shots: LongVideoShotState[];
  final_asset_id?: string;
  /** Server-side project row id (long_video_projects table). */
  project_id?: string;
  selection?: LongVideoSelection;
}

export interface LongVideoProjectSummary {
  id: string;
  title: string;
  shot_count: number;
  keyframe_count: number;
  segment_count: number;
  has_final: boolean;
  created_at: string;
  updated_at: string;
}

export interface LongVideoProjectDetail {
  id: string;
  title: string;
  state: Omit<LongVideoProjectState, 'project_id' | 'selection'>;
  created_at: string;
  updated_at: string;
}

export interface CanvasSessionState {
  items: Record<string, CanvasItemState>;
  viewport: CanvasViewportState;
  staging: CanvasStagingState;
  active_asset_path: string;
  overlays?: CanvasOverlaysState;
  /** Cached lineage edges among session nodes (DB is authoritative). */
  edges?: CanvasEdge[];
  composer_snapshot?: CanvasComposerSnapshot | null;
}

export interface CanvasSessionSummary {
  id: string;
  media: string;
  title: string;
  item_count: number;
  created_at: string;
  updated_at: string;
}

export interface CanvasSessionDetail {
  id: string;
  media: string;
  title: string;
  state: CanvasSessionState;
  created_at: string;
  updated_at: string;
}