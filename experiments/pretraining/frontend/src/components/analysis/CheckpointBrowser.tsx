/**
 * Checkpoint Browser Component
 *
 * Browse and select model checkpoints for analysis and generation.
 */

import { useEffect } from 'react';
import { useTraining } from '../../context/TrainingContext';

export function CheckpointBrowser() {
  const { checkpoints, fetchCheckpoints, isLoading } = useTraining();

  useEffect(() => {
    fetchCheckpoints('nano');
  }, [fetchCheckpoints]);

  const formatDate = (timestamp: string | null): string => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  const formatLoss = (loss: number | null): string => {
    if (loss === null) return 'N/A';
    return loss.toFixed(4);
  };

  return (
    <div className="checkpoint-browser">
      <h2>Checkpoint Browser</h2>
      <p className="section-description">
        Browse saved model checkpoints. Select a checkpoint to use for
        generation or analysis.
      </p>

      <div className="checkpoint-controls">
        <button
          className="btn btn-secondary"
          onClick={() => fetchCheckpoints('nano')}
          disabled={isLoading}
        >
          {isLoading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {checkpoints.length === 0 ? (
        <div className="checkpoints-empty">
          <p>No checkpoints found.</p>
          <p className="hint">
            Enable "Save Checkpoints" in training configuration to save model
            snapshots during training.
          </p>
        </div>
      ) : (
        <div className="checkpoints-list">
          <div className="checkpoint-header">
            <span className="col-id">ID</span>
            <span className="col-step">Step</span>
            <span className="col-epoch">Epoch</span>
            <span className="col-loss">Train Loss</span>
            <span className="col-loss">Val Loss</span>
            <span className="col-time">Timestamp</span>
          </div>

          {checkpoints.map((ckpt) => (
            <div key={ckpt.id} className="checkpoint-row">
              <span className="col-id">{ckpt.id}</span>
              <span className="col-step">{ckpt.step}</span>
              <span className="col-epoch">{ckpt.epoch}</span>
              <span className="col-loss">{formatLoss(ckpt.train_loss)}</span>
              <span className="col-loss">{formatLoss(ckpt.val_loss)}</span>
              <span className="col-time">{formatDate(ckpt.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default CheckpointBrowser;
