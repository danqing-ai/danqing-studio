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
  python_version?: string;
  dependencies?: Record<string, string>;
  [key: string]: unknown;
}

export interface SettingsData {
  language?: string;
  [key: string]: unknown;
}

export type PageKey = 'image_create' | 'video_create' | 'audio_create' | 'gallery' | 'models' | 'settings';

export const VALID_PAGES: PageKey[] = [
  'image_create',
  'video_create',
  'audio_create',
  'gallery',
  'models',
  'settings',
];