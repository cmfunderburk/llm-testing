/**
 * Training Controls Component
 */

import { useState, useEffect } from 'react';
import { useTraining } from '../../context/TrainingContext';
import type { TrainingConfig } from '../../types';
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
  });

  const [vramEstimate, setVramEstimate] = useState<VRAMEstimate | null>(null);

  // Auto-update context_length when model size changes
  useEffect(() => {
    const defaultContextLength = DEFAULT_CONTEXT_LENGTHS[config.config_name] || 256;
    setConfig(prev => ({ ...prev, context_length: defaultContextLength }));
  }, [config.config_name]);

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

  const handleStart = () => {
    startTraining(config);
  };

  const canStart = status.state === 'idle' || status.state === 'completed' || status.state === 'error';
  const canPause = status.state === 'running';
  const canResume = status.state === 'paused';
  const canStop = status.state === 'running' || status.state === 'paused';

  return (
    <div className="training-controls">
      <h3>Training Configuration</h3>

      <div className="config-form">
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
            Save Checkpoints
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
            Start Training
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
      </div>
    </div>
  );
}

export default TrainingControls;
