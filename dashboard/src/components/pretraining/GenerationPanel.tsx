/**
 * Generation Panel Component
 *
 * Allows users to generate text from trained model checkpoints.
 */

import { useState, useEffect } from 'react';
import type { CheckpointInfo, GenerateRequest, GenerateResponse } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function GenerationPanel() {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');
  const [prompt, setPrompt] = useState('Once upon a time');
  const [maxTokens, setMaxTokens] = useState(50);
  const [temperature, setTemperature] = useState(0.8);
  const [isGenerating, setIsGenerating] = useState(false);
  const [output, setOutput] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Fetch available checkpoints
  const fetchCheckpoints = async () => {
    setIsLoadingCheckpoints(true);
    try {
      // Fetch checkpoints for all config types
      const configs = ['nano', 'small', 'medium'];
      const allCheckpoints: CheckpointInfo[] = [];

      for (const config of configs) {
        const res = await fetch(`${API_URL}/api/pretraining/checkpoints?config_name=${config}`);
        if (res.ok) {
          const data = await res.json();
          allCheckpoints.push(...data);
        }
      }

      setCheckpoints(allCheckpoints);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
      setError('Failed to load checkpoints');
    } finally {
      setIsLoadingCheckpoints(false);
    }
  };

  useEffect(() => {
    fetchCheckpoints();
  }, []);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);

    const request: GenerateRequest = {
      prompt,
      max_tokens: maxTokens,
      temperature,
    };

    if (selectedCheckpoint) {
      request.checkpoint_id = selectedCheckpoint;
    }

    try {
      const res = await fetch(`${API_URL}/api/pretraining/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Generation failed');
      }

      const data: GenerateResponse = await res.json();
      setOutput(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
      setOutput(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopy = () => {
    if (output?.text) {
      navigator.clipboard.writeText(output.text);
    }
  };

  const handleDeleteAll = async () => {
    if (!confirm(`Delete all ${checkpoints.length} checkpoints? This cannot be undone.`)) {
      return;
    }

    setIsDeleting(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/pretraining/checkpoints/all`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Delete failed');
      }

      const data = await res.json();
      setSelectedCheckpoint('');
      await fetchCheckpoints();
      alert(`Deleted ${data.deleted_count} checkpoint files.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    } finally {
      setIsDeleting(false);
    }
  };

  const hasCheckpoints = checkpoints.length > 0;

  return (
    <div className="generation-panel">
      <h3>Text Generation</h3>

      <div className="generation-form">
        <div className="form-row">
          <label>Checkpoint</label>
          <div className="checkpoint-select-row">
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              disabled={isGenerating}
            >
              <option value="">
                {isLoadingCheckpoints
                  ? 'Loading...'
                  : hasCheckpoints
                  ? 'Random weights (untrained)'
                  : 'No checkpoints found'}
              </option>
              {checkpoints.map((ckpt) => (
                <option key={ckpt.id} value={ckpt.id}>
                  {ckpt.id}
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
            <button
              className="btn btn-small btn-danger"
              onClick={handleDeleteAll}
              disabled={isDeleting || !hasCheckpoints}
              title="Delete all checkpoints"
            >
              {isDeleting ? '...' : '🗑'}
            </button>
          </div>
        </div>

        <div className="form-row">
          <label>Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            disabled={isGenerating}
            placeholder="Enter your prompt..."
          />
        </div>

        <div className="form-row">
          <label>Temperature: {temperature.toFixed(2)}</label>
          <input
            type="range"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            min={0.1}
            max={2.0}
            step={0.1}
            disabled={isGenerating}
          />
        </div>

        <div className="form-row">
          <label>Max Tokens</label>
          <input
            type="number"
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value) || 50)}
            min={1}
            max={500}
            disabled={isGenerating}
          />
        </div>

        <button
          className="btn btn-primary generate-btn"
          onClick={handleGenerate}
          disabled={isGenerating || !prompt.trim()}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {error && <div className="generation-error">{error}</div>}

      {output && (
        <div className="generation-output">
          <div className="output-header">
            <span className="output-meta">
              {output.tokens_generated} tokens generated
            </span>
            <button className="btn btn-small btn-secondary" onClick={handleCopy}>
              Copy
            </button>
          </div>
          <pre className="output-text">{output.text}</pre>
        </div>
      )}
    </div>
  );
}

export default GenerationPanel;
