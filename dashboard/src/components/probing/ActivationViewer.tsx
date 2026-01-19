/**
 * Activation Viewer Component
 *
 * Displays extracted activation metadata and tokens.
 */

import type { ActivationExtractResponse } from '../../types';

interface ActivationViewerProps {
  result: ActivationExtractResponse;
}

export function ActivationViewer({ result }: ActivationViewerProps) {
  return (
    <div className="activation-viewer">
      <div className="viewer-header">
        <h3>Extracted Activations</h3>
        <div className="viewer-meta">
          <span>Tokens: {result.tokens.length}</span>
          <span>Layers: {result.layers.length}</span>
          <span>Hidden size: {result.hidden_size}</span>
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

export default ActivationViewer;
