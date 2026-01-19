/**
 * Attention Viewer Component
 *
 * Controls for viewing attention data - layer/head selection.
 */

import type { AttentionExtractResponse } from '../../types';

interface AttentionViewerProps {
  result: AttentionExtractResponse;
  selectedLayer: number;
  onLayerChange: (layer: number) => void;
  selectedHead: number | null;
  onHeadChange: (head: number | null) => void;
}

export function AttentionViewer({
  result,
  selectedLayer,
  onLayerChange,
  selectedHead,
  onHeadChange,
}: AttentionViewerProps) {
  const currentLayerData = result.layers.find(l => l.layer_idx === selectedLayer);
  const numHeads = currentLayerData?.num_heads || 0;

  return (
    <div className="attention-viewer">
      <div className="viewer-header">
        <h3>Attention Patterns</h3>
        <div className="viewer-meta">
          <span>Tokens: {result.tokens.length}</span>
          <span>Layers captured: {result.num_layers_captured}</span>
        </div>
      </div>

      <div className="viewer-controls">
        <div className="control-group">
          <label>Layer</label>
          <div className="button-group">
            {result.layers.map((layer) => (
              <button
                key={layer.layer_idx}
                className={`btn-select ${selectedLayer === layer.layer_idx ? 'selected' : ''}`}
                onClick={() => onLayerChange(layer.layer_idx)}
              >
                {layer.layer_idx}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <label>Head</label>
          <div className="button-group">
            <button
              className={`btn-select ${selectedHead === null ? 'selected' : ''}`}
              onClick={() => onHeadChange(null)}
            >
              Avg
            </button>
            {Array.from({ length: Math.min(numHeads, 8) }, (_, i) => (
              <button
                key={i}
                className={`btn-select ${selectedHead === i ? 'selected' : ''}`}
                onClick={() => onHeadChange(i)}
              >
                {i}
              </button>
            ))}
            {numHeads > 8 && (
              <select
                value={selectedHead !== null && selectedHead >= 8 ? selectedHead : ''}
                onChange={(e) => onHeadChange(parseInt(e.target.value))}
                className="head-select"
              >
                <option value="" disabled>More...</option>
                {Array.from({ length: numHeads - 8 }, (_, i) => (
                  <option key={i + 8} value={i + 8}>
                    Head {i + 8}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
      </div>

      <div className="token-list">
        <label>Tokens:</label>
        <div className="tokens">
          {result.tokens.map((token, i) => (
            <span key={i} className="token-badge" title={`Position ${i}`}>
              {token}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AttentionViewer;
