/**
 * Attention Page
 *
 * Interface for extracting and visualizing attention patterns.
 */

import { useState } from 'react';
import { Card, LoadingSpinner } from '../components/shared';
import { AttentionViewer, HeatmapGrid } from '../components/attention';
import type { AttentionExtractResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function AttentionPage() {
  const [text, setText] = useState('The quick brown fox jumps over the lazy dog.');
  const [selectedLayers, setSelectedLayers] = useState<number[]>([0, 14, 27]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AttentionExtractResponse | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number | null>(null);

  const handleExtract = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/attention/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          layers: selectedLayers.length > 0 ? selectedLayers : null,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Extraction failed');
      }

      const data = await response.json();
      setResult(data);
      if (data.layers.length > 0) {
        setSelectedLayer(data.layers[0].layer_idx);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLayerSelect = (layer: number) => {
    if (selectedLayers.includes(layer)) {
      setSelectedLayers(selectedLayers.filter(l => l !== layer));
    } else {
      setSelectedLayers([...selectedLayers, layer].sort((a, b) => a - b));
    }
  };

  return (
    <div className="attention-page">
      <div className="page-header">
        <h2>Attention Visualization</h2>
        <p className="page-description">
          Extract and visualize attention patterns from transformer models.
          See how tokens attend to each other across different layers and heads.
        </p>
      </div>

      <Card title="Input">
        <div className="form-section">
          <label>Text to analyze</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder="Enter text to analyze attention patterns..."
          />
        </div>

        <div className="form-section">
          <label>Layers to extract (click to toggle)</label>
          <div className="layer-selector">
            {[0, 7, 14, 21, 27].map((layer) => (
              <button
                key={layer}
                className={`layer-btn ${selectedLayers.includes(layer) ? 'selected' : ''}`}
                onClick={() => handleLayerSelect(layer)}
              >
                {layer}
              </button>
            ))}
          </div>
          <p className="form-hint">
            Selected: {selectedLayers.length > 0 ? selectedLayers.join(', ') : 'All layers'}
          </p>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleExtract}
          disabled={isLoading || !text.trim()}
        >
          {isLoading ? 'Extracting...' : 'Extract Attention'}
        </button>
      </Card>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {isLoading && (
        <Card>
          <LoadingSpinner message="Loading model and extracting attention patterns..." />
        </Card>
      )}

      {result && !isLoading && (
        <>
          <AttentionViewer
            result={result}
            selectedLayer={selectedLayer}
            onLayerChange={setSelectedLayer}
            selectedHead={selectedHead}
            onHeadChange={setSelectedHead}
          />

          <Card title="Attention Heatmap">
            <HeatmapGrid
              tokens={result.tokens}
              weights={
                selectedHead !== null
                  ? result.layers.find(l => l.layer_idx === selectedLayer)?.heads[selectedHead]?.weights || []
                  : result.layers.find(l => l.layer_idx === selectedLayer)?.average_weights || []
              }
            />
          </Card>
        </>
      )}
    </div>
  );
}

export default AttentionPage;
