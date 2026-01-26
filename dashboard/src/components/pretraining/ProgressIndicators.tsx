/**
 * Progress Indicators Component
 */

import { useTraining } from '../../context/TrainingContext';

export function ProgressIndicators() {
  const { status, loadingMessage, loadingProgress } = useTraining();

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

  const estimateRemaining = (): string => {
    if (status.current_step === 0 || status.elapsed_time === 0) return '--';
    const stepsRemaining = status.total_steps - status.current_step;
    const secsPerStep = status.elapsed_time / status.current_step;
    const remaining = stepsRemaining * secsPerStep;
    return formatTime(remaining);
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const formatTokens = (tokens: number): string => {
    if (tokens < 1000) return tokens.toString();
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`;
    if (tokens < 1000000000) return `${(tokens / 1000000).toFixed(1)}M`;
    return `${(tokens / 1000000000).toFixed(2)}B`;
  };

  return (
    <div className="progress-indicators">
      <h3>Training Progress</h3>

      {status.state === 'loading' && (
        <div className="loading-section">
          <div className="loading-message">
            <div className="loading-spinner" />
            <span>{loadingMessage || 'Loading...'}</span>
          </div>
          {loadingProgress && loadingProgress.total_bytes > 0 && (
            <div className="loading-progress">
              <div className="progress-bar loading">
                <div
                  className="progress-fill"
                  style={{ width: `${loadingProgress.percent}%` }}
                />
              </div>
              <div className="loading-stats">
                <span className="loading-stat">
                  {formatBytes(loadingProgress.bytes_read)} / {formatBytes(loadingProgress.total_bytes)}
                </span>
                <span className="loading-stat">
                  {formatTokens(loadingProgress.tokens)} tokens
                </span>
              </div>
            </div>
          )}
        </div>
      )}

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
