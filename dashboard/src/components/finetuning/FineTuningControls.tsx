/**
 * Fine-Tuning Controls Component
 */

import { useState, useEffect } from 'react';
import { useFineTuning } from '../../context/FineTuningContext';
import type { FineTuningConfig, AdapterCheckpointInfo } from '../../types';
import { LORA_RANKS } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function FineTuningControls() {
  const { status, isLoading, startTraining, pauseTraining, resumeTraining, stopTraining } = useFineTuning();

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
    save_adapter: true,
  });

  const [checkpoints, setCheckpoints] = useState<AdapterCheckpointInfo[]>([]);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);
  const [isSavingNow, setIsSavingNow] = useState(false);

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
    fetchCheckpoints();
  }, []);

  const handleStart = () => {
    startTraining(config);
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
            onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
            disabled={!canStart}
          >
            <option value="unsloth/Qwen2.5-7B-Instruct">Qwen 2.5 7B Instruct</option>
          </select>
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
