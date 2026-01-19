/**
 * TypeScript types for the LLM Learning Lab Dashboard
 */

// =============================================================================
// Common Types
// =============================================================================

export type Track = 'pretraining' | 'attention' | 'probing';

// =============================================================================
// Pretraining Types
// =============================================================================

export type TrainingState = 'idle' | 'running' | 'paused' | 'completed' | 'error';

export interface TrainingConfig {
  config_name: string;
  corpus: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  save_checkpoints: boolean;
}

export interface TrainingStatus {
  state: TrainingState;
  current_step: number;
  current_epoch: number;
  total_steps: number;
  train_loss: number | null;
  val_loss: number | null;
  tokens_seen: number;
  elapsed_time: number;
  config: TrainingConfig | null;
}

export interface TrainingMetrics {
  type: 'metrics' | 'generation' | 'complete' | 'error' | 'status' | 'heartbeat';
  step?: number;
  epoch?: number;
  train_loss?: number;
  val_loss?: number;
  learning_rate?: number;
  tokens_seen?: number;
  tokens_per_sec?: number;
  elapsed_time?: number;
  text?: string;
  message?: string;
  state?: TrainingState;
}

export interface CheckpointInfo {
  id: string;
  path: string;
  step: number;
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  timestamp: string | null;
}

export interface GenerateRequest {
  checkpoint_id?: string;
  prompt: string;
  max_tokens: number;
  temperature: number;
  top_k?: number;
  top_p?: number;
}

export interface GenerateResponse {
  text: string;
  tokens_generated: number;
  prompt: string;
}

// Model configs available for pretraining
export const MODEL_CONFIGS = ['nano', 'small', 'medium'] as const;
export type ModelConfig = typeof MODEL_CONFIGS[number];

// Corpora available for pretraining
// Built-in: verdict, tiny (small, for testing)
// Downloadable: tinystories, wikitext2, shakespeare
// Run: python -m experiments.pretraining.download_corpora --all
export const CORPORA = ['verdict', 'tiny', 'tinystories', 'wikitext2', 'shakespeare'] as const;
export type Corpus = typeof CORPORA[number];

// =============================================================================
// Attention Types
// =============================================================================

export interface AttentionExtractRequest {
  text: string;
  layers?: number[];
  model_name?: string;
}

export interface AttentionHeadData {
  head_idx: number;
  weights: number[][];
}

export interface AttentionLayerData {
  layer_idx: number;
  num_heads: number;
  heads: AttentionHeadData[];
  average_weights: number[][];
}

export interface AttentionExtractResponse {
  tokens: string[];
  layers: AttentionLayerData[];
  model_info: Record<string, unknown>;
  seq_len: number;
  num_layers_captured: number;
}

export interface ModelInfo {
  name: string;
  is_loaded: boolean;
  description?: string;
}

export interface ModelStatus {
  model_loaded: boolean;
  model_name: string | null;
  loading: boolean;
  gpu_memory_allocated?: number;
  gpu_memory_reserved?: number;
}

// =============================================================================
// Probing Types
// =============================================================================

export interface ActivationExtractRequest {
  text: string;
  layers?: number[];
  positions: string[];
  model_name?: string;
}

export interface PositionActivation {
  position: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  norm: number;
}

export interface LayerActivationData {
  layer_idx: number;
  positions: PositionActivation[];
  attention_contrib_norm?: number;
  ffn_contrib_norm?: number;
}

export interface ActivationExtractResponse {
  tokens: string[];
  layers: LayerActivationData[];
  model_info: Record<string, unknown>;
  seq_len: number;
  hidden_size: number;
}

export interface LayerDiffRequest {
  text: string;
  layer: number;
  model_name?: string;
}

export interface LayerDiffResponse {
  tokens: string[];
  layer_idx: number;
  attention_contribution: Record<string, number>;
  ffn_contribution: Record<string, number>;
  per_token_attention_norm: number[];
  per_token_ffn_norm: number[];
}
