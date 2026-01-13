/**
 * Progress Indicators Component
 *
 * Shows training progress with step/epoch counters and progress bars.
 */

import { useTraining } from '../../context/TrainingContext';

export function ProgressIndicators() {
  const { status } = useTraining();

  const stepProgress =
    status.total_steps > 0 ? (status.current_step / status.total_steps) * 100 : 0;

  const epochProgress =
    status.config?.epochs && status.config.epochs > 0
      ? (status.current_epoch / status.config.epochs) * 100
      : 0;

  const formatTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${Math.floor(seconds)}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins < 60) {
      return `${mins}m ${secs}s`;
    }
    const hours = Math.floor(mins / 60);
    const remainMins = mins % 60;
    return `${hours}h ${remainMins}m`;
  };

  // Estimate time remaining
  const estimateRemaining = (): string => {
    if (status.current_step === 0 || status.elapsed_time === 0) return '--';
    const stepsRemaining = status.total_steps - status.current_step;
    const secsPerStep = status.elapsed_time / status.current_step;
    const remaining = stepsRemaining * secsPerStep;
    return formatTime(remaining);
  };

  return (
    <div className="progress-indicators">
      <h3>Training Progress</h3>

      <div className="progress-section">
        <div className="progress-header">
          <span className="progress-label">Steps</span>
          <span className="progress-value">
            {status.current_step} / {status.total_steps || '?'}
          </span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${stepProgress}%` }} />
        </div>
      </div>

      <div className="progress-section">
        <div className="progress-header">
          <span className="progress-label">Epochs</span>
          <span className="progress-value">
            {status.current_epoch} / {status.config?.epochs || '?'}
          </span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill epoch" style={{ width: `${epochProgress}%` }} />
        </div>
      </div>

      <div className="time-stats">
        <div className="time-stat">
          <span className="time-label">Elapsed</span>
          <span className="time-value">{formatTime(status.elapsed_time)}</span>
        </div>
        <div className="time-stat">
          <span className="time-label">Remaining</span>
          <span className="time-value">{estimateRemaining()}</span>
        </div>
      </div>
    </div>
  );
}

export default ProgressIndicators;
