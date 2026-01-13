/**
 * Attention Visualization Component
 *
 * Displays attention patterns as a heatmap grid.
 * Shows how tokens attend to each other in the transformer.
 */

import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface AttentionData {
  tokens: string[];
  attention_weights: number[][];
  layer: number;
  head: number | null;
}

export function AttentionVisualization() {
  const [text, setText] = useState('The quick brown fox');
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [attentionData, setAttentionData] = useState<AttentionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze/attention`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          layer,
          head,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data: AttentionData = await response.json();
      setAttentionData(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  // Render attention heatmap
  const renderHeatmap = () => {
    if (!attentionData) return null;

    const { tokens, attention_weights } = attentionData;

    // Color scale from blue (low) to red (high)
    const getColor = (value: number): string => {
      const intensity = Math.min(Math.max(value, 0), 1);
      const r = Math.floor(intensity * 255);
      const b = Math.floor((1 - intensity) * 255);
      return `rgb(${r}, 50, ${b})`;
    };

    return (
      <div className="attention-heatmap">
        {/* Column headers (keys) */}
        <div className="heatmap-row header-row">
          <div className="heatmap-cell corner"></div>
          {tokens.map((token, i) => (
            <div key={i} className="heatmap-cell header" title={token}>
              {token.length > 6 ? token.slice(0, 5) + '…' : token}
            </div>
          ))}
        </div>

        {/* Rows (queries) */}
        {attention_weights.map((row, i) => (
          <div key={i} className="heatmap-row">
            <div className="heatmap-cell row-header" title={tokens[i]}>
              {tokens[i].length > 6 ? tokens[i].slice(0, 5) + '…' : tokens[i]}
            </div>
            {row.map((value, j) => (
              <div
                key={j}
                className="heatmap-cell"
                style={{ backgroundColor: getColor(value) }}
                title={`${tokens[i]} → ${tokens[j]}: ${value.toFixed(3)}`}
              />
            ))}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="attention-visualization">
      <h2>Attention Visualization</h2>
      <p className="section-description">
        Visualize how tokens attend to each other. Brighter colors indicate
        stronger attention weights.
      </p>

      <div className="attention-form">
        <div className="form-section">
          <label htmlFor="attn-text">Input Text</label>
          <input
            id="attn-text"
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
          />
        </div>

        <div className="form-grid">
          <div className="form-section">
            <label htmlFor="attn-layer">Layer</label>
            <input
              id="attn-layer"
              type="number"
              value={layer}
              onChange={(e) => setLayer(parseInt(e.target.value) || 0)}
              min={0}
              max={11}
            />
          </div>

          <div className="form-section">
            <label htmlFor="attn-head">
              Head <span className="param-value">{head ?? 'avg'}</span>
            </label>
            <input
              id="attn-head"
              type="range"
              value={head ?? -1}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setHead(val < 0 ? null : val);
              }}
              min={-1}
              max={7}
            />
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleAnalyze}
          disabled={isLoading || !text.trim()}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Attention'}
        </button>
      </div>

      {error && <div className="analysis-error">{error}</div>}

      {attentionData && (
        <div className="attention-result">
          <h3>
            Attention Pattern
            <span className="attention-info">
              Layer {attentionData.layer}, Head {attentionData.head ?? 'avg'}
            </span>
          </h3>
          {renderHeatmap()}
          <div className="heatmap-legend">
            <span className="legend-label">Low</span>
            <div className="legend-gradient" />
            <span className="legend-label">High</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default AttentionVisualization;
