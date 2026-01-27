/**
 * Fine-Tuning Metrics Display Component
 */

import { useFineTuning } from '../../context/FineTuningContext';

export function FineTuningMetrics() {
  const { status, metricsHistory } = useFineTuning();

  const latestMetrics = metricsHistory[metricsHistory.length - 1];

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

  const trainableRatio = (status.trainable_params && status.total_params)
    ? ((status.trainable_params / status.total_params) * 100).toFixed(2)
    : null;

  return (
    <div className="metrics-display">
      <h3>Fine-Tuning Metrics</h3>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Step</div>
          <div className="metric-value">
            {status.current_step} / {status.total_steps || '?'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Epoch</div>
          <div className="metric-value">
            {typeof status.current_epoch === 'number' ? status.current_epoch.toFixed(1) : '-'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Train Loss</div>
          <div className="metric-value loss">{formatNumber(status.train_loss)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Eval Loss</div>
          <div className="metric-value">{formatNumber(status.eval_loss)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Learning Rate</div>
          <div className="metric-value">{formatNumber(latestMetrics?.learning_rate ?? status.learning_rate)}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Elapsed</div>
          <div className="metric-value">{formatTime(status.elapsed_time)}</div>
        </div>

        {trainableRatio && (
          <div className="metric-card">
            <div className="metric-label">Trainable</div>
            <div className="metric-value">{trainableRatio}%</div>
          </div>
        )}

        {status.trainable_params && (
          <div className="metric-card">
            <div className="metric-label">Params</div>
            <div className="metric-value">{(status.trainable_params / 1e6).toFixed(1)}M</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FineTuningMetrics;
