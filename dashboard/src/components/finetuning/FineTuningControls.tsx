/**
 * Fine-Tuning Controls Component
 */

import { useState, useEffect } from 'react';
import { useFineTuning } from '../../context/FineTuningContext';
import type {
  FineTuningConfig,
  AdapterCheckpointInfo,
  FineTuningModelOption,
  FineTuningVRAMEstimate,
} from '../../types';
import { LORA_RANKS, FINE_TUNING_MODELS } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function FineTuningControls() {
  const { status, isLoading, startTraining, pauseTraining, resumeTraining, stopTraining } = useFineTuning();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [config, setConfig] = useState<FineTuningConfig>({
    model_name: 'unsloth/Qwen2.5-7B-Instruct',
    max_seq_length: 1024,
    n_examples: 500,
    lora_r: 32,
    lora_alpha: 32,
    lora_dropout: 0.05,
    learning_rate: 2e-4,
    num_epochs: 1,
    batch_size: 4,
    gradient_accumulation: 4,
    warmup_ratio: 0.1,
    logging_steps: 10,
    eval_steps: 50,
    optim: 'adamw_8bit',
    fast_mode: false,
    use_gradient_checkpointing: true,
    save_adapter: true,
  });

  const [modelOptions, setModelOptions] = useState<FineTuningModelOption[]>(FINE_TUNING_MODELS);
  const [checkpoints, setCheckpoints] = useState<AdapterCheckpointInfo[]>([]);
  const [vramEstimate, setVramEstimate] = useState<FineTuningVRAMEstimate | null>(null);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);
  const [isSavingNow, setIsSavingNow] = useState(false);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/api/fine-tuning/models`);
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        setModelOptions(data);
      }
    } catch (err) {
      console.error('Failed to fetch fine-tuning model registry:', err);
    }
  };

  const fetchCheckpoints = async () => {
    setIsLoadingCheckpoints(true);
    try {
      const res = await fetch(`${API_URL}/api/fine-tuning/checkpoints`);
      if (res.ok) {
        const data = await res.json();
        setCheckpoints(data);
      }
    } catch (err) {
      console.error('Failed to fetch adapter checkpoints:', err);
    } finally {
      setIsLoadingCheckpoints(false);
    }
  };

  useEffect(() => {
    fetchModels();
    fetchCheckpoints();
  }, []);

  useEffect(() => {
    const fetchEstimate = async () => {
      try {
        const query = new URLSearchParams({
          model_name: config.model_name,
          max_seq_length: String(config.max_seq_length),
          batch_size: String(config.batch_size),
          lora_r: String(config.lora_r),
          optim: config.optim,
          use_gradient_checkpointing: String(config.use_gradient_checkpointing),
        });
        const res = await fetch(`${API_URL}/api/fine-tuning/estimate-vram?${query.toString()}`);
        if (!res.ok) return;
        const data = await res.json();
        setVramEstimate(data);
      } catch (err) {
        console.error('Failed to fetch fine-tuning VRAM estimate:', err);
      }
    };
    fetchEstimate();
  }, [
    config.model_name,
    config.max_seq_length,
    config.batch_size,
    config.lora_r,
    config.optim,
    config.use_gradient_checkpointing,
  ]);

  const handleStart = () => {
    startTraining(config);
  };

  const handleModelChange = (modelName: string) => {
    const selected = modelOptions.find((model) => model.hf_id === modelName);
    setConfig((prev) => ({
      ...prev,
      model_name: modelName,
      max_seq_length: selected ? Math.min(prev.max_seq_length, selected.recommended_max_seq) : prev.max_seq_length,
      lora_r: selected ? selected.recommended_lora_r : prev.lora_r,
    }));
  };

  const handleSaveNow = async () => {
    setIsSavingNow(true);
    try {
      const res = await fetch(`${API_URL}/api/fine-tuning/checkpoint/save-now`, {
        method: 'POST',
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Save failed');
      }
      setTimeout(fetchCheckpoints, 2000);
    } catch (err) {
      console.error('Failed to save checkpoint:', err);
      alert(err instanceof Error ? err.message : 'Failed to save checkpoint');
    } finally {
      setIsSavingNow(false);
    }
  };

  const effectiveBatchSize = config.batch_size * config.gradient_accumulation;

  const canStart = status.state === 'idle' || status.state === 'completed' || status.state === 'error';
  const canPause = status.state === 'running';
  const canResume = status.state === 'paused';
  const canStop = status.state === 'running' || status.state === 'paused';
  const canSaveNow = status.state === 'running';

  return (
    <div className="training-controls">
      <h3>Fine-Tuning Configuration</h3>

      <div className="config-form">
        <div className="form-row">
          <label>Model</label>
          <select
            value={config.model_name}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={!canStart}
          >
            {modelOptions.map((model) => (
              <option key={model.hf_id} value={model.hf_id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>

        <div className="form-row">
          <label>Max Sequence Length</label>
          <input
            type="number"
            value={config.max_seq_length}
            onChange={(e) => setConfig({ ...config, max_seq_length: parseInt(e.target.value) || 512 })}
            min={256}
            max={8192}
            step={128}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Dataset Examples</label>
          <input
            type="number"
            value={config.n_examples}
            onChange={(e) => setConfig({ ...config, n_examples: parseInt(e.target.value) || 100 })}
            min={50}
            max={5000}
            step={50}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>LoRA Rank</label>
          <select
            value={config.lora_r}
            onChange={(e) => setConfig({ ...config, lora_r: parseInt(e.target.value) })}
            disabled={!canStart}
          >
            {LORA_RANKS.map((r) => (
              <option key={r} value={r}>r = {r}</option>
            ))}
          </select>
        </div>

        <div className="form-row">
          <label>LoRA Alpha</label>
          <input
            type="number"
            value={config.lora_alpha}
            onChange={(e) => setConfig({ ...config, lora_alpha: parseInt(e.target.value) || 16 })}
            min={1}
            max={128}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Learning Rate</label>
          <input
            type="number"
            value={config.learning_rate}
            onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 2e-4 })}
            step={0.0001}
            min={0.00001}
            max={0.01}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Epochs</label>
          <input
            type="number"
            value={config.num_epochs}
            onChange={(e) => setConfig({ ...config, num_epochs: parseInt(e.target.value) || 1 })}
            min={1}
            max={10}
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
            max={16}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <label>Gradient Accumulation</label>
          <input
            type="number"
            value={config.gradient_accumulation}
            onChange={(e) => setConfig({ ...config, gradient_accumulation: parseInt(e.target.value) || 1 })}
            min={1}
            max={32}
            disabled={!canStart}
          />
          <small className="form-hint">Effective batch size: {effectiveBatchSize}</small>
        </div>

        <div className="form-row">
          <label>Eval Steps</label>
          <input
            type="number"
            value={config.eval_steps}
            onChange={(e) => setConfig({ ...config, eval_steps: parseInt(e.target.value) || 25 })}
            min={10}
            max={500}
            disabled={!canStart}
          />
        </div>

        <div className="form-row">
          <button
            type="button"
            className="btn btn-small btn-secondary"
            onClick={() => setShowAdvanced(!showAdvanced)}
            disabled={!canStart}
          >
            {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
          </button>
        </div>

        {showAdvanced && (
          <>
            <div className="form-row">
              <label>Optimizer</label>
              <select
                value={config.optim}
                onChange={(e) => setConfig({ ...config, optim: e.target.value as FineTuningConfig['optim'] })}
                disabled={!canStart}
              >
                <option value="adamw_8bit">AdamW 8-bit</option>
                <option value="paged_adamw_8bit">Paged AdamW 8-bit</option>
                <option value="adamw_torch">AdamW Torch</option>
              </select>
            </div>

            <div className="form-row checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={config.use_gradient_checkpointing}
                  onChange={(e) => setConfig({ ...config, use_gradient_checkpointing: e.target.checked })}
                  disabled={!canStart}
                />
                Use Gradient Checkpointing
              </label>
            </div>

            <div className="form-row checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={config.fast_mode}
                  onChange={(e) => setConfig({ ...config, fast_mode: e.target.checked })}
                  disabled={!canStart}
                />
                Fast Mode (enable Unsloth patching)
              </label>
            </div>
          </>
        )}

        <div className="form-row">
          <label>Resume from Checkpoint</label>
          <div className="checkpoint-select-row">
            <select
              value={config.resume_from || ''}
              onChange={(e) => setConfig({ ...config, resume_from: e.target.value || undefined })}
              disabled={!canStart}
            >
              <option value="">
                {isLoadingCheckpoints
                  ? 'Loading...'
                  : checkpoints.length > 0
                  ? 'Start fresh'
                  : 'No checkpoints available'}
              </option>
              {checkpoints.map((ckpt) => (
                <option key={ckpt.id} value={ckpt.path}>
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

        <div className="form-row checkbox">
          <label>
            <input
              type="checkbox"
              checked={config.save_adapter}
              onChange={(e) => setConfig({ ...config, save_adapter: e.target.checked })}
              disabled={!canStart}
            />
            Save adapter on completion
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
            {config.resume_from ? 'Resume Training' : 'Start Fine-Tuning'}
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
            title="Save adapter checkpoint immediately"
          >
            {isSavingNow ? 'Saving...' : 'Save Now'}
          </button>
        )}
      </div>
    </div>
  );
}

export default FineTuningControls;
