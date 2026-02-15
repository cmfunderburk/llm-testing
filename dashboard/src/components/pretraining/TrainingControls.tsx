/**
 * Training Controls Component
 */

import { useState, useEffect, useRef } from 'react';
import { useTraining } from '../../context/TrainingContext';
import type { TrainingConfig, CheckpointInfo, DatasetConfig, PretrainingOptimizer } from '../../types';
import { MODEL_CONFIGS, DATASETS } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Model config display names with parameter counts
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  nano: 'nano (~10M)',
  small: 'small (~50M)',
  medium: 'medium (~124M)',
  large: 'large (~204M)',
  xlarge: 'xlarge (~355M)',
};

// Default context lengths for each model size
const DEFAULT_CONTEXT_LENGTHS: Record<string, number> = {
  nano: 256,
  small: 512,
  medium: 1024,
  large: 1024,
  xlarge: 1024,
};

interface VRAMEstimate {
  total_gb: number;
  total_mb: number;
  model_mb: number;
  optimizer_mb: number;
  gradients_mb: number;
  activations_mb: number;
  warning: string | null;
}

const OPTIMIZERS: PretrainingOptimizer[] = ['adamw', 'adamw_8bit', 'paged_adamw_8bit'];

export function TrainingControls() {
  const { status, isLoading, startTraining, pauseTraining, resumeTraining, stopTraining } = useTraining();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showAdvancedGuide, setShowAdvancedGuide] = useState(false);
  const guideCloseButtonRef = useRef<HTMLButtonElement | null>(null);

  const [selectedDataset, setSelectedDataset] = useState<DatasetConfig>(DATASETS[0]);
  const [config, setConfig] = useState<TrainingConfig>({
    config_name: 'nano',
    corpus: DATASETS[0].corpus,
    val_corpus: DATASETS[0].val_corpus,
    epochs: 10,
    batch_size: 4,
    grad_accum_steps: 1,
    learning_rate: 3e-4,
    warmup_steps: 100,
    save_checkpoints: false,
    context_length: 256,
    resume_from: undefined,
    attention_impl: 'manual',
    precision: 'fp32',
    optimizer: 'adamw',
    gradient_checkpointing: false,
    tie_embeddings: false,
  });

  // Update corpus/val_corpus when dataset changes
  const handleDatasetChange = (datasetName: string) => {
    const dataset = DATASETS.find(d => d.name === datasetName);
    if (dataset) {
      setSelectedDataset(dataset);
      setConfig(prev => ({
        ...prev,
        corpus: dataset.corpus,
        val_corpus: dataset.val_corpus,
      }));
    }
  };

  const [vramEstimate, setVramEstimate] = useState<VRAMEstimate | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);
  const [isSavingNow, setIsSavingNow] = useState(false);
  const [configWarnings, setConfigWarnings] = useState<string[]>([]);
  const effectiveBatchSize = config.batch_size * config.grad_accum_steps;

  // Fetch checkpoints for resume dropdown
  const fetchCheckpoints = async () => {
    setIsLoadingCheckpoints(true);
    try {
      const configs = ['nano', 'small', 'medium', 'large', 'xlarge'];
      const allCheckpoints: CheckpointInfo[] = [];

      for (const cfg of configs) {
        const res = await fetch(`${API_URL}/api/pretraining/checkpoints?config_name=${cfg}`);
        if (res.ok) {
          const data = await res.json();
          allCheckpoints.push(...data);
        }
      }

      setCheckpoints(allCheckpoints);
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
    } finally {
      setIsLoadingCheckpoints(false);
    }
  };

  useEffect(() => {
    fetchCheckpoints();
  }, []);

  // Sync local config from ongoing training run's config (e.g., after page reload)
  useEffect(() => {
    // Only sync if training is active and has a config
    const isActiveRun = status.state === 'running' || status.state === 'loading' || status.state === 'paused';
    if (!isActiveRun || !status.config) return;

    // Update local config to match the running training
    setConfig(status.config);

    // Also sync the selectedDataset to match corpus
    const matchingDataset = DATASETS.find(d => d.corpus === status.config!.corpus);
    if (matchingDataset) {
      setSelectedDataset(matchingDataset);
    }
  }, [status.state, status.config]);

  // Auto-update context_length when model size changes (only if not resuming and not in active run)
  useEffect(() => {
    // Don't auto-update during an active training run - respect the synced config
    const isActiveRun = status.state === 'running' || status.state === 'loading' || status.state === 'paused';
    if (!config.resume_from && !isActiveRun) {
      const defaultContextLength = DEFAULT_CONTEXT_LENGTHS[config.config_name] || 256;
      setConfig(prev => ({ ...prev, context_length: defaultContextLength }));
    }
  }, [config.config_name, config.resume_from, status.state]);

  // Fetch VRAM estimate when config changes
  useEffect(() => {
    const fetchEstimate = async () => {
      try {
        const contextLength = config.context_length || DEFAULT_CONTEXT_LENGTHS[config.config_name] || 256;
        const query = new URLSearchParams({
          config_name: config.config_name,
          batch_size: String(config.batch_size),
          context_length: String(contextLength),
          precision: config.precision,
          optimizer: config.optimizer,
          attention_impl: config.attention_impl,
          gradient_checkpointing: String(config.gradient_checkpointing),
          tie_embeddings: String(config.tie_embeddings),
        });

        const res = await fetch(`${API_URL}/api/pretraining/estimate-vram?${query.toString()}`);
        if (res.ok) {
          const data = await res.json();
          setVramEstimate(data);
        }
      } catch (err) {
        console.error('Failed to fetch VRAM estimate:', err);
      }
    };
    fetchEstimate();
  }, [
    config.config_name,
    config.batch_size,
    config.context_length,
    config.precision,
    config.optimizer,
    config.attention_impl,
    config.gradient_checkpointing,
    config.tie_embeddings,
  ]);

  // Handle checkpoint selection for resume
  const handleResumeFromChange = (checkpointId: string) => {
    if (!checkpointId) {
      setConfig(prev => ({ ...prev, resume_from: undefined }));
      setConfigWarnings([]);
      return;
    }

    const checkpoint = checkpoints.find(c => c.id === checkpointId);
    if (!checkpoint) return;

    // Auto-populate config from checkpoint
    const newConfig = {
      ...config,
      resume_from: checkpointId,
    };

    // Extract config_name from checkpoint id (first part before _)
    const parts = checkpointId.split('_');
    if (parts.length > 0 && MODEL_CONFIGS.includes(parts[0] as typeof MODEL_CONFIGS[number])) {
      newConfig.config_name = parts[0];
    }

    // Apply checkpoint's training params if available
    if (checkpoint.corpus) newConfig.corpus = checkpoint.corpus;
    if (checkpoint.batch_size) newConfig.batch_size = checkpoint.batch_size;
    if (checkpoint.context_length) newConfig.context_length = checkpoint.context_length;
    if (checkpoint.optimizer && OPTIMIZERS.includes(checkpoint.optimizer as PretrainingOptimizer)) {
      newConfig.optimizer = checkpoint.optimizer as PretrainingOptimizer;
    }

    setConfig(newConfig);

    // Note: warnings will be shown if user changes these values after selection
    setConfigWarnings([]);
  };

  // Check for config differences from checkpoint
  useEffect(() => {
    if (!config.resume_from) {
      setConfigWarnings([]);
      return;
    }

    const checkpoint = checkpoints.find(c => c.id === config.resume_from);
    if (!checkpoint) return;

    const warnings: string[] = [];
    if (checkpoint.corpus && checkpoint.corpus !== config.corpus) {
      warnings.push(`corpus: ${checkpoint.corpus} -> ${config.corpus}`);
    }
    if (checkpoint.batch_size && checkpoint.batch_size !== config.batch_size) {
      warnings.push(`batch_size: ${checkpoint.batch_size} -> ${config.batch_size}`);
    }
    if (checkpoint.context_length && checkpoint.context_length !== config.context_length) {
      warnings.push(`context_length: ${checkpoint.context_length} -> ${config.context_length}`);
    }
    if (checkpoint.optimizer && checkpoint.optimizer !== config.optimizer) {
      warnings.push(`optimizer: ${checkpoint.optimizer} -> ${config.optimizer}`);
    }

    setConfigWarnings(warnings);
  }, [config, checkpoints]);

  const handleStart = () => {
    startTraining(config);
  };

  const handleSaveNow = async () => {
    setIsSavingNow(true);
    try {
      const res = await fetch(`${API_URL}/api/pretraining/checkpoint/save-now`, {
        method: 'POST',
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Save failed');
      }
      // Refresh checkpoints after save
      setTimeout(fetchCheckpoints, 1000);
    } catch (err) {
      console.error('Failed to save checkpoint:', err);
      alert(err instanceof Error ? err.message : 'Failed to save checkpoint');
    } finally {
      setIsSavingNow(false);
    }
  };

  const canStart = status.state === 'idle' || status.state === 'completed' || status.state === 'error';
  const canPause = status.state === 'running';
  const canResume = status.state === 'paused';
  const canStop = status.state === 'running' || status.state === 'paused';
  const canSaveNow = status.state === 'running';

  const hasCheckpoints = checkpoints.length > 0;

  useEffect(() => {
    if (!showAdvancedGuide) return;
    guideCloseButtonRef.current?.focus();
  }, [showAdvancedGuide]);

  useEffect(() => {
    if (!showAdvancedGuide) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowAdvancedGuide(false);
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    document.addEventListener('keydown', onKeyDown);

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [showAdvancedGuide]);

  const applyMemoryPreset = () => {
    setConfig((prev) => ({
      ...prev,
      attention_impl: 'sdpa',
      precision: 'bf16',
      optimizer: 'adamw_8bit',
      gradient_checkpointing: true,
      tie_embeddings: true,
      grad_accum_steps: Math.max(prev.grad_accum_steps, 2),
    }));
    setShowAdvanced(true);
    setShowAdvancedGuide(false);
  };

  return (
    <div className="training-controls">
      <h3>Training Configuration</h3>

      <div className="config-form">
        {/* Resume from checkpoint dropdown */}
        <div className="form-row">
          <label>Resume from Checkpoint</label>
          <div className="checkpoint-select-row">
            <select
              value={config.resume_from || ''}
              onChange={(e) => handleResumeFromChange(e.target.value)}
              disabled={!canStart}
            >
              <option value="">
                {isLoadingCheckpoints
                  ? 'Loading...'
                  : hasCheckpoints
                  ? 'Start fresh (no checkpoint)'
                  : 'No checkpoints available'}
              </option>
              {checkpoints.map((ckpt) => (
                <option key={ckpt.id} value={ckpt.id}>
                  {ckpt.id} (step {ckpt.step})
                </option>
              ))}
            </select>
            <button
              className="btn btn-small btn-secondary"
              onClick={fetchCheckpoints}
              disabled={isLoadingCheckpoints}
              title="Refresh checkpoint list"
            >
              {isLoadingCheckpoints ? '...' : '↻'}
            </button>
          </div>
        </div>

        {/* Warning when config differs from checkpoint */}
        {configWarnings.length > 0 && (
          <div className="config-warning">
            Config differs from checkpoint: {configWarnings.join(', ')}
          </div>
        )}

        <div className="form-row">
          <label>Model Size</label>
          <select
            value={config.config_name}
            onChange={(e) => setConfig({ ...config, config_name: e.target.value })}
            disabled={!canStart}
          >
            {MODEL_CONFIGS.map((c) => (
              <option key={c} value={c}>{MODEL_DISPLAY_NAMES[c] || c}</option>
            ))}
          </select>
        </div>

        <div className="form-row">
          <label>Dataset</label>
          <select
            value={selectedDataset.name}
            onChange={(e) => handleDatasetChange(e.target.value)}
            disabled={!canStart}
          >
            {DATASETS.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.size})
              </option>
            ))}
          </select>
          <small className="form-hint">
            {selectedDataset.description}
            {selectedDataset.val_corpus && ' (official train/val split)'}
          </small>
        </div>

        <div className="form-row">
          <label>Epochs</label>
          <input
            type="number"
            value={config.epochs}
            onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 1 })}
            min={1}
            max={100}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Batch Size</label>
          <input
            type="number"
            value={config.batch_size}
            onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) || 1 })}
            min={1}
            max={32}
            disabled={!canStart}
          />
          <small className="form-hint">Effective batch size: {effectiveBatchSize}</small>
        </div>

        <div className="form-row">
          <label>Context Length</label>
          <input
            type="number"
            value={config.context_length}
            onChange={(e) => setConfig({ ...config, context_length: parseInt(e.target.value) || 64 })}
            min={64}
            max={2048}
            step={64}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Learning Rate</label>
          <input
            type="number"
            value={config.learning_rate}
            onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 1e-4 })}
            step={0.0001}
            min={0.00001}
            max={0.01}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Warmup Steps</label>
          <input
            type="number"
            value={config.warmup_steps}
            onChange={(e) => setConfig({ ...config, warmup_steps: parseInt(e.target.value) || 0 })}
            min={0}
            max={1000}
            disabled={!canStart}
          />
        </div>

        <div className="form-row checkbox">
          <label>
            <input
              type="checkbox"
              checked={config.save_checkpoints}
              onChange={(e) => setConfig({ ...config, save_checkpoints: e.target.checked })}
              disabled={!canStart}
            />
            Save Checkpoints (at 25%, 50%, 75%, 100%)
          </label>
        </div>

        <div className="form-row advanced-actions-row">
          <button
            type="button"
            className="btn btn-small btn-secondary"
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={!canStart}
          >
            {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
          </button>
          <button
            type="button"
            className="btn btn-small btn-ghost"
            onClick={() => setShowAdvancedGuide(true)}
            aria-haspopup="dialog"
            aria-expanded={showAdvancedGuide}
            aria-controls="advanced-settings-guide"
          >
            Advanced Guide
          </button>
        </div>

        {showAdvanced && (
          <>
            <div className="form-row">
              <label>Precision</label>
              <select
                value={config.precision}
                onChange={(e) => setConfig({ ...config, precision: e.target.value as TrainingConfig['precision'] })}
                disabled={!canStart}
              >
                <option value="fp32">FP32 (baseline)</option>
                <option value="bf16">BF16</option>
                <option value="fp16">FP16</option>
              </select>
            </div>

            <div className="form-row">
              <label>Optimizer</label>
              <select
                value={config.optimizer}
                onChange={(e) => setConfig({ ...config, optimizer: e.target.value as TrainingConfig['optimizer'] })}
                disabled={!canStart}
              >
                <option value="adamw">AdamW (full-precision states)</option>
                <option value="adamw_8bit">AdamW 8-bit</option>
                <option value="paged_adamw_8bit">Paged AdamW 8-bit</option>
              </select>
              <small className="form-hint">
                8-bit variants reduce optimizer memory usage; paged mode can reduce VRAM further with host-memory paging.
              </small>
            </div>

            <div className="form-row">
              <label>Attention Implementation</label>
              <select
                value={config.attention_impl}
                onChange={(e) => setConfig({
                  ...config,
                  attention_impl: e.target.value as TrainingConfig['attention_impl'],
                })}
                disabled={!canStart}
              >
                <option value="manual">Manual</option>
                <option value="sdpa">SDPA</option>
              </select>
            </div>

            <div className="form-row">
              <label>Grad Accumulation Steps</label>
              <input
                type="number"
                value={config.grad_accum_steps}
                onChange={(e) => setConfig({ ...config, grad_accum_steps: parseInt(e.target.value) || 1 })}
                min={1}
                max={128}
                disabled={!canStart}
              />
            </div>

            <div className="form-row checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={config.gradient_checkpointing}
                  onChange={(e) => setConfig({ ...config, gradient_checkpointing: e.target.checked })}
                  disabled={!canStart}
                />
                Enable Gradient Checkpointing
              </label>
            </div>

            <div className="form-row checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={config.tie_embeddings}
                  onChange={(e) => setConfig({ ...config, tie_embeddings: e.target.checked })}
                  disabled={!canStart}
                />
                Tie Input/Output Embeddings
              </label>
            </div>
          </>
        )}
      </div>

      {vramEstimate && (
        <div className={`vram-estimate ${vramEstimate.warning ? 'vram-warning' : ''}`}>
          <div className="vram-header">
            <span className="vram-label">Est. VRAM</span>
            <span className="vram-value">{vramEstimate.total_gb.toFixed(1)} GB</span>
          </div>
          {vramEstimate.warning && (
            <div className="vram-warning-text">{vramEstimate.warning}</div>
          )}
        </div>
      )}

      <div className="control-buttons">
        {canStart && (
          <button className="btn btn-primary" onClick={handleStart} disabled={isLoading}>
            {config.resume_from ? 'Resume Training' : 'Start Training'}
          </button>
        )}

        {canPause && (
          <button className="btn btn-warning" onClick={pauseTraining} disabled={isLoading}>
            Pause
          </button>
        )}

        {canResume && (
          <button className="btn btn-success" onClick={resumeTraining} disabled={isLoading}>
            Resume
          </button>
        )}

        {canStop && (
          <button className="btn btn-danger" onClick={stopTraining} disabled={isLoading}>
            Stop
          </button>
        )}

        {canSaveNow && (
          <button
            className="btn btn-secondary"
            onClick={handleSaveNow}
            disabled={isSavingNow}
            title="Save checkpoint immediately"
          >
            {isSavingNow ? 'Saving...' : 'Save Now'}
          </button>
        )}
      </div>

      {showAdvancedGuide && (
        <div
          className="modal-overlay"
          role="presentation"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setShowAdvancedGuide(false);
            }
          }}
        >
          <div
            id="advanced-settings-guide"
            className="modal-card settings-guide-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="advanced-guide-title"
            aria-describedby="advanced-guide-summary"
          >
            <div className="modal-header">
              <h4 id="advanced-guide-title">Pretraining Advanced Settings Guide</h4>
              <button
                ref={guideCloseButtonRef}
                type="button"
                className="btn btn-small btn-secondary"
                onClick={() => setShowAdvancedGuide(false)}
                aria-label="Close advanced settings guide"
              >
                Close
              </button>
            </div>

            <p id="advanced-guide-summary" className="guide-summary">
              Use this panel when you need to fit larger contexts/models on limited VRAM or want faster training.
              Start simple, then turn on memory-saving options one at a time.
            </p>

            <div className="guide-section">
              <h5>Recommended 16GB workflow</h5>
              <ol className="guide-list">
                <li>Keep your desired context length, but lower micro-batch if needed.</li>
                <li>Switch attention to SDPA first. It reduces attention-memory overhead.</li>
                <li>Use BF16 precision and enable gradient checkpointing.</li>
                <li>Switch optimizer to AdamW 8-bit (or paged AdamW 8-bit for tighter VRAM budgets).</li>
                <li>Use grad accumulation to recover effective batch size.</li>
              </ol>
            </div>

            <div className="guide-section">
              <h5>Setting reference</h5>
              <ul className="guide-list">
                <li><strong>Attention Implementation</strong>: <code>manual</code> is educational; <code>sdpa</code> uses PyTorch scaled dot-product kernels and is usually faster/lower-memory.</li>
                <li><strong>Precision</strong>: <code>fp32</code> is baseline; <code>bf16</code> is usually the best stability/speed tradeoff on modern GPUs; <code>fp16</code> can be faster but may be less stable.</li>
                <li><strong>Grad Accumulation Steps</strong>: increases effective batch without increasing per-step VRAM.</li>
                <li><strong>Gradient Checkpointing</strong>: recomputes activations during backward pass to reduce memory usage.</li>
                <li><strong>Tie Embeddings</strong>: shares input/output embedding weights to cut parameters and memory.</li>
                <li><strong>Optimizer</strong>: <code>adamw</code> is baseline; <code>adamw_8bit</code> and <code>paged_adamw_8bit</code> reduce optimizer-state memory.</li>
              </ul>
            </div>

            <div className="guide-section">
              <h5>Quick actions</h5>
              <div className="guide-actions">
                <button type="button" className="btn btn-small btn-primary" onClick={applyMemoryPreset}>
                  Apply 16GB-friendly preset
                </button>
                <button
                  type="button"
                  className="btn btn-small btn-secondary"
                  onClick={() => {
                    setShowAdvanced(true);
                    setShowAdvancedGuide(false);
                  }}
                >
                  Open advanced controls
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TrainingControls;
