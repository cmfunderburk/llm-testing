/**
 * Token Counter Component
 *
 * Displays tokens seen and throughput (tokens/sec).
 * Includes animated progress visualization.
 */

import { useTraining } from '../../context/TrainingContext';

export function TokenCounter() {
  const { status, metricsHistory } = useTraining();

  // Get latest throughput
  const latestMetrics = metricsHistory[metricsHistory.length - 1];
  const tokensPerSec = latestMetrics?.tokens_per_sec ?? 0;

  // Format large numbers
  const formatTokens = (num: number): string => {
    if (num >= 1_000_000) {
      return `${(num / 1_000_000).toFixed(2)}M`;
    }
    if (num >= 1_000) {
      return `${(num / 1_000).toFixed(1)}K`;
    }
    return num.toString();
  };

  return (
    <div className="token-counter">
      <h3>Token Statistics</h3>

      <div className="token-stats">
        <div className="token-stat-item">
          <div className="token-stat-value">{formatTokens(status.tokens_seen)}</div>
          <div className="token-stat-label">Tokens Processed</div>
        </div>

        <div className="token-stat-item">
          <div className="token-stat-value">{tokensPerSec.toFixed(0)}</div>
          <div className="token-stat-label">Tokens/sec</div>
        </div>
      </div>

      {status.state === 'running' && (
        <div className="throughput-bar">
          <div
            className="throughput-fill"
            style={{
              width: `${Math.min((tokensPerSec / 5000) * 100, 100)}%`,
            }}
          />
        </div>
      )}
    </div>
  );
}

export default TokenCounter;
