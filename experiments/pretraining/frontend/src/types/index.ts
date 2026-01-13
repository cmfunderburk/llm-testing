/**
 * TypeScript types for the Pretraining Lab frontend
 */

// Training state
export type TrainingState = 'idle' | 'running' | 'paused' | 'completed' | 'error';

// Training configuration
export interface TrainingConfig {
  config_name: string;
  corpus: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  save_checkpoints: boolean;
}

// Training status from API
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

// Metrics from WebSocket
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

// Checkpoint info
export interface CheckpointInfo {
  id: string;
  path: string;
  step: number;
  epoch: number;
  train_loss: number | null;
  val_loss: number | null;
  timestamp: string | null;
}

// Generation request/response
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

// Model configs available
export const MODEL_CONFIGS = ['nano', 'small', 'medium'] as const;
export type ModelConfig = typeof MODEL_CONFIGS[number];

// Corpus options
export const CORPORA = ['verdict', 'tiny'] as const;
export type Corpus = typeof CORPORA[number];
