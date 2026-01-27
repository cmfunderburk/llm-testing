/**
 * Fine-Tuning Generation Panel Component
 *
 * Test fine-tuned models by generating text with saved adapters.
 */

import { useState, useEffect } from 'react';
import type { AdapterCheckpointInfo, GenerateResponse } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function GenerationPanel() {
  const [checkpoints, setCheckpoints] = useState<AdapterCheckpointInfo[]>([]);
  const [selectedAdapter, setSelectedAdapter] = useState<string>('');
  const [prompt, setPrompt] = useState('Explain what machine learning is in simple terms.');
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [isGenerating, setIsGenerating] = useState(false);
  const [output, setOutput] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingCheckpoints, setIsLoadingCheckpoints] = useState(false);

  const fetchCheckpoints = async () => {
    setIsLoadingCheckpoints(true);
    try {
      const res = await fetch(`${API_URL}/api/fine-tuning/checkpoints`);
      if (res.ok) {
        const data = await res.json();
        setCheckpoints(data);
      }
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
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

    try {
      const res = await fetch(`${API_URL}/api/fine-tuning/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          adapter_path: selectedAdapter || null,
          prompt,
          max_tokens: maxTokens,
          temperature,
        }),
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

  return (
    <div className="generation-panel">
      <h3>Test Generation</h3>

      <div className="generation-form">
        <div className="form-row">
          <label>Adapter</label>
          <div className="checkpoint-select-row">
            <select
              value={selectedAdapter}
              onChange={(e) => setSelectedAdapter(e.target.value)}
              disabled={isGenerating}
            >
              <option value="">
                {isLoadingCheckpoints
                  ? 'Loading...'
                  : 'Base model (no adapter)'}
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
              title="Refresh adapter list"
            >
              {isLoadingCheckpoints ? '...' : '↻'}
            </button>
          </div>
        </div>

        <div className="form-row">
          <label>Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
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
            min={0}
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
            onChange={(e) => setMaxTokens(parseInt(e.target.value) || 128)}
            min={1}
            max={1024}
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
