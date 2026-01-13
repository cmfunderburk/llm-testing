/**
 * Interactive Text Generation Component
 *
 * Allows users to generate text from trained model checkpoints
 * with configurable parameters (temperature, top-k, top-p).
 */

import { useState } from 'react';
import type { GenerateRequest, GenerateResponse } from '../../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function TextGeneration() {
  const [prompt, setPrompt] = useState('The meaning of life is');
  const [maxTokens, setMaxTokens] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState<number | undefined>(undefined);
  const [topP, setTopP] = useState<number | undefined>(undefined);

  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);

    const request: GenerateRequest = {
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      top_p: topP,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Generation failed');
      }

      const data: GenerateResponse = await response.json();
      setResult(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="text-generation">
      <h2>Interactive Text Generation</h2>
      <p className="section-description">
        Generate text using the trained model. Adjust parameters to explore
        different sampling strategies.
      </p>

      <div className="generation-form">
        <div className="form-section">
          <label htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="Enter your prompt..."
          />
        </div>

        <div className="form-grid">
          <div className="form-section">
            <label htmlFor="max-tokens">Max Tokens</label>
            <input
              id="max-tokens"
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value) || 50)}
              min={1}
              max={256}
            />
          </div>

          <div className="form-section">
            <label htmlFor="temperature">
              Temperature <span className="param-value">{temperature.toFixed(2)}</span>
            </label>
            <input
              id="temperature"
              type="range"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              min={0.1}
              max={2.0}
              step={0.1}
            />
          </div>

          <div className="form-section">
            <label htmlFor="top-k">
              Top-K <span className="param-value">{topK ?? 'off'}</span>
            </label>
            <input
              id="top-k"
              type="range"
              value={topK ?? 0}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setTopK(val === 0 ? undefined : val);
              }}
              min={0}
              max={100}
              step={5}
            />
          </div>

          <div className="form-section">
            <label htmlFor="top-p">
              Top-P <span className="param-value">{topP?.toFixed(2) ?? 'off'}</span>
            </label>
            <input
              id="top-p"
              type="range"
              value={topP ?? 0}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setTopP(val === 0 ? undefined : val);
              }}
              min={0}
              max={1}
              step={0.05}
            />
          </div>
        </div>

        <button
          className="btn btn-primary btn-generate"
          onClick={handleGenerate}
          disabled={isGenerating || !prompt.trim()}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {error && <div className="generation-error">{error}</div>}

      {result && (
        <div className="generation-result">
          <h3>Generated Text</h3>
          <div className="result-meta">
            <span>Tokens generated: {result.tokens_generated}</span>
          </div>
          <pre className="result-text">{result.text}</pre>
        </div>
      )}
    </div>
  );
}

export default TextGeneration;
