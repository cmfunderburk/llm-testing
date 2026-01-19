/**
 * Training Controls Component
 */

import { useState, useEffect } from 'react';
import { useTraining } from '../../context/TrainingContext';
import type { TrainingConfig, CheckpointInfo } from '../../types';
import { MODEL_CONFIGS, CORPORA } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Model config display names with parameter counts
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  nano: 'nano (~10M)',
  small: 'small (~50M)',
  medium: 'medium (~124M)',
};

// Default context lengths for each model size
const DEFAULT_CONTEXT_LENGTHS: Record<string, number> = {
  nano: 256,
  small: 512,
  medium: 1024,
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

export function TrainingControls() {
  const { status, isLoading, startTraining, pauseTraining, resumeTraining, stopTraining } = useTraining();

  const [config, setConfig] = useState<TrainingConfig>({
    config_name: 'nano',
    corpus: 'verdict',
    epochs: 10,
    batch_size: 4,
    learning_rate: 3e-4,
    warmup_steps: 100,
    save_checkpoints: false,
    context_length: 256,
    resume_from: undefined,
  });

  const [vramEstimate, setVramEstimate] = useState<VRAMEstimate | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);
  const [isSavingNow, setIsSavingNow] = useState(false);
  const [configWarnings, setConfigWarnings] = useState<string[]>([]);

  // Fetch checkpoints for resume dropdown
  const fetchCheckpoints = async () => {
    setIsLoadingCheckpoints(true);
    try {
      const configs = ['nano', 'small', 'medium'];
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

  // Auto-update context_length when model size changes (only if not resuming)
  useEffect(() => {
    if (!config.resume_from) {
      const defaultContextLength = DEFAULT_CONTEXT_LENGTHS[config.config_name] || 256;
      setConfig(prev => ({ ...prev, context_length: defaultContextLength }));
    }
  }, [config.config_name, config.resume_from]);

  // Fetch VRAM estimate when config changes
  useEffect(() => {
    const fetchEstimate = async () => {
      try {
        const contextLength = config.context_length || DEFAULT_CONTEXT_LENGTHS[config.config_name] || 256;
        const res = await fetch(
          `${API_URL}/api/pretraining/estimate-vram?config_name=${config.config_name}&batch_size=${config.batch_size}&context_length=${contextLength}`
        );
        if (res.ok) {
          const data = await res.json();
          setVramEstimate(data);
        }
      } catch (err) {
        console.error('Failed to fetch VRAM estimate:', err);
      }
    };
    fetchEstimate();
  }, [config.config_name, config.batch_size, config.context_length]);

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
          <label>Corpus</label>
          <select
            value={config.corpus}
            onChange={(e) => setConfig({ ...config, corpus: e.target.value })}
            disabled={!canStart}
          >
            {CORPORA.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
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
    </div>
  );
}

export default TrainingControls;
