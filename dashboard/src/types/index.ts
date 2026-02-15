/**
 * TypeScript types for the LLM Learning Lab Dashboard
 */

// =============================================================================
// Common Types
// =============================================================================

export type Track = 'pretraining' | 'fine-tuning' | 'attention' | 'probing';

// =============================================================================
// Pretraining Types
// =============================================================================

export type TrainingState = 'idle' | 'loading' | 'running' | 'paused' | 'completed' | 'error';
export type PretrainingAttentionImpl = 'manual' | 'sdpa';
export type PretrainingPrecision = 'fp32' | 'bf16' | 'fp16';
export type PretrainingOptimizer = 'adamw' | 'adamw_8bit' | 'paged_adamw_8bit';

export interface TrainingConfig {
  config_name: string;
  corpus: string;
  val_corpus?: string;  // Separate validation corpus (e.g., 'pg19_validation')
  epochs: number;
  batch_size: number;
  grad_accum_steps: number;
  learning_rate: number;
  warmup_steps: number;
  save_checkpoints: boolean;
  context_length?: number;  // Optional override for model's default
  resume_from?: string;  // checkpoint_id to resume from
  attention_impl: PretrainingAttentionImpl;
  precision: PretrainingPrecision;
  optimizer: PretrainingOptimizer;
  gradient_checkpointing: boolean;
  tie_embeddings: boolean;
}

export interface TrainingStatus {
  state: TrainingState;
  run_id?: string | null;
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
  type: 'metrics' | 'validation' | 'generation' | 'checkpoint' | 'complete' | 'error' | 'warning' | 'status' | 'heartbeat' | 'loading_progress';
  run_id?: string;
  step?: number;
  epoch?: number;
  train_loss?: number;
  val_loss?: number;
  learning_rate?: number;
  tokens_seen?: number;
  tokens_per_sec?: number;
  elapsed_time?: number;
  text?: string;
  message?: string;  // Loading/error messages
  state?: TrainingState;
  total_steps?: number;
  resumed_from_step?: number | null;
  final_step?: number;
  final_train_loss?: number;
  checkpoint_path?: string;
  path?: string;
  manual?: boolean;
  percentage?: number;
  // Loading progress fields
  phase?: string;
  bytes_read?: number;
  total_bytes?: number;
  tokens?: number;
  percent?: number;
}

export interface CheckpointInfo {
  id: string;
  path: string;
  step: number;
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  timestamp: string | null;
  corpus?: string;
  batch_size?: number;
  context_length?: number;
  optimizer?: string;
}

export interface RunMetric {
  step: number;
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  learning_rate: number | null;
  tokens_seen: number | null;
  tokens_per_sec: number | null;
  elapsed_time: number | null;
}

export interface PretrainingRunSummary {
  run_id: string;
  state: TrainingState;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  current_step: number;
  total_steps: number;
  tokens_seen: number;
  final_train_loss: number | null;
  final_val_loss: number | null;
  config: TrainingConfig | null;
}

export interface PretrainingRunDetail extends PretrainingRunSummary {
  metrics: RunMetric[];
  generations: {
    step: number;
    epoch: number;
    text: string;
    timestamp?: string | null;
  }[];
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
export const MODEL_CONFIGS = ['nano', 'small', 'medium', 'large', 'xlarge'] as const;
export type ModelConfig = typeof MODEL_CONFIGS[number];

// Dataset definitions with optional train/val splits
// For datasets with official splits, val_corpus is set automatically
export interface DatasetConfig {
  name: string;
  corpus: string;
  val_corpus?: string;  // If set, uses official validation split
  description: string;
  size: string;
}

export const DATASETS: DatasetConfig[] = [
  // Built-in small datasets (auto-split)
  { name: 'Verdict', corpus: 'verdict', description: 'Built-in test corpus', size: '~8 KB' },
  { name: 'Tiny', corpus: 'tiny', description: 'Minimal test corpus', size: '~350 B' },

  // Downloadable datasets (auto-split)
  { name: 'Shakespeare', corpus: 'shakespeare', description: 'Complete works of Shakespeare', size: '~1 MB' },
  { name: 'WikiText-2', corpus: 'wikitext2', description: 'Wikipedia articles', size: '~13 MB' },
  { name: 'Wikipedia GA Intros', corpus: 'wikipedia_ga_intros', description: 'Introductions from 50K+ Good Article Wikipedia pages', size: '~65 MB' },
  { name: 'TinyStories', corpus: 'tinystories', description: '2.1M synthetic short stories', size: '~1.8 GB' },

  // PG-19 with official splits
  { name: 'PG-19 (small)', corpus: 'pg19_train_small', val_corpus: 'pg19_validation_small',
    description: 'Project Gutenberg subset (100 books)', size: '~40 MB' },
  { name: 'PG-19 (full)', corpus: 'pg19_train', val_corpus: 'pg19_validation',
    description: 'Project Gutenberg pre-1919 books', size: '~11 GB' },
  { name: 'PG-19 (small, normalized)', corpus: 'pg19_train_small_normalized', val_corpus: 'pg19_validation_small_normalized',
    description: 'PG-19 subset with prose line-wrap normalization', size: '~40 MB' },
  { name: 'PG-19 (full, normalized)', corpus: 'pg19_train_normalized', val_corpus: 'pg19_validation_normalized',
    description: 'PG-19 with prose line-wrap normalization', size: '~11 GB' },
  { name: 'PG-19 (small, docs+EOT)', corpus: 'pg19_train_small_docs', val_corpus: 'pg19_validation_small_docs',
    description: 'PG-19 JSONL docs with explicit document boundaries', size: '~40 MB' },
  { name: 'PG-19 (full, docs+EOT)', corpus: 'pg19_train_docs', val_corpus: 'pg19_validation_docs',
    description: 'PG-19 JSONL docs with explicit document boundaries', size: '~11 GB' },
  { name: 'PG-19 (small, docs+EOT normalized)', corpus: 'pg19_train_small_docs_normalized', val_corpus: 'pg19_validation_small_docs_normalized',
    description: 'PG-19 JSONL docs with normalization + explicit boundaries', size: '~40 MB' },
  { name: 'PG-19 (full, docs+EOT normalized)', corpus: 'pg19_train_docs_normalized', val_corpus: 'pg19_validation_docs_normalized',
    description: 'PG-19 JSONL docs with normalization + explicit boundaries', size: '~11 GB' },
];

// Legacy: individual corpus names for backward compatibility
export const CORPORA = [
  'verdict', 'tiny', 'tinystories', 'wikitext2', 'wikipedia_ga_intros', 'shakespeare',
  'pg19_train', 'pg19_validation', 'pg19_test',
  'pg19_train_small', 'pg19_validation_small', 'pg19_test_small',
  'pg19_train_normalized', 'pg19_validation_normalized', 'pg19_test_normalized',
  'pg19_train_small_normalized', 'pg19_validation_small_normalized', 'pg19_test_small_normalized',
  'pg19_train_docs', 'pg19_validation_docs', 'pg19_test_docs',
  'pg19_train_small_docs', 'pg19_validation_small_docs', 'pg19_test_small_docs',
  'pg19_train_docs_normalized', 'pg19_validation_docs_normalized', 'pg19_test_docs_normalized',
  'pg19_train_small_docs_normalized', 'pg19_validation_small_docs_normalized', 'pg19_test_small_docs_normalized',
] as const;
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

// =============================================================================
// Fine-Tuning Types
// =============================================================================

export type FineTuningState = 'idle' | 'loading' | 'running' | 'paused' | 'completed' | 'error';
export type FineTuningOptimizer = 'adamw_8bit' | 'paged_adamw_8bit' | 'adamw_torch';

export interface FineTuningModelOption {
  name: string;
  hf_id: string;
  family: string;
  params_billion: number;
  recommended_max_seq: number;
  recommended_lora_r: number;
  tags: string[];
}

export interface FineTuningVRAMEstimate {
  model_mb: number;
  lora_mb: number;
  optimizer_mb: number;
  activations_mb: number;
  overhead_mb: number;
  total_mb: number;
  total_gb: number;
  warning: string | null;
}

export interface FineTuningConfig {
  model_name: string;
  max_seq_length: number;
  n_examples: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  learning_rate: number;
  num_epochs: number;
  batch_size: number;
  gradient_accumulation: number;
  warmup_ratio: number;
  logging_steps: number;
  eval_steps: number;
  optim: FineTuningOptimizer;
  fast_mode: boolean;
  use_gradient_checkpointing: boolean;
  save_adapter: boolean;
  resume_from?: string;
}

export interface FineTuningStatus {
  state: FineTuningState;
  current_step: number;
  total_steps: number;
  current_epoch: number;
  train_loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  elapsed_time: number;
  config: FineTuningConfig | null;
  trainable_params: number | null;
  total_params: number | null;
}

export interface FineTuningMetrics {
  type: 'metrics' | 'validation' | 'checkpoint' | 'complete' | 'error' | 'status' | 'heartbeat';
  step?: number;
  epoch?: number;
  train_loss?: number;
  eval_loss?: number;
  learning_rate?: number;
  elapsed_time?: number;
  message?: string;
  state?: FineTuningState;
  total_steps?: number;
  trainable_params?: number;
  total_params?: number;
}

export interface AdapterCheckpointInfo {
  id: string;
  path: string;
  step: number;
  train_loss: number | null;
  eval_loss: number | null;
  timestamp: string | null;
}

export const FINE_TUNING_MODELS: FineTuningModelOption[] = [
  {
    name: 'Qwen 2.5 7B Instruct',
    hf_id: 'unsloth/Qwen2.5-7B-Instruct',
    family: 'qwen',
    params_billion: 7.0,
    recommended_max_seq: 2048,
    recommended_lora_r: 32,
    tags: ['balanced', 'default'],
  },
  {
    name: 'Qwen 2.5 14B Instruct',
    hf_id: 'unsloth/Qwen2.5-14B-Instruct',
    family: 'qwen',
    params_billion: 14.0,
    recommended_max_seq: 1024,
    recommended_lora_r: 16,
    tags: ['high-capacity', '16gb-stretch'],
  },
];
export const LORA_RANKS = [8, 16, 32, 64] as const;
