/**
 * Metrics Display Component
 */

import { useTraining } from '../../context/TrainingContext';

export function MetricsDisplay() {
  const { status, metricsHistory } = useTraining();

  const latestMetrics = metricsHistory[metricsHistory.length - 1];
  const tokensPerSec = latestMetrics?.tokens_per_sec || 0;

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatNumber = (num: number | null | undefined): string => {
    if (num === null || num === undefined) return '-';
    if (num < 0.001) return num.toExponential(2);
    return num.toFixed(4);
  };

  return (
    <div className="metrics-display">
      <h3>Training Metrics</h3>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Step</div>
          <div className="metric-value">
            {status.current_step} / {status.total_steps || '?'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Epoch</div>
          <div className="metric-value">{status.current_epoch}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Train Loss</div>
          <div className="metric-value loss">{formatNumber(status.train_loss)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Val Loss</div>
          <div className="metric-value">{formatNumber(status.val_loss)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Tokens Seen</div>
          <div className="metric-value">{status.tokens_seen.toLocaleString()}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Tokens/sec</div>
          <div className="metric-value">{tokensPerSec.toFixed(1)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Learning Rate</div>
          <div className="metric-value">{formatNumber(latestMetrics?.learning_rate)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Elapsed</div>
          <div className="metric-value">{formatTime(status.elapsed_time)}</div>
        </div>
      </div>
    </div>
  );
}

export default MetricsDisplay;
