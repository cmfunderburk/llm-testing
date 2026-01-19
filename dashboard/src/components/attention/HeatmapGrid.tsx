/**
 * Heatmap Grid Component
 *
 * Renders attention weights as a heatmap grid.
 */

interface HeatmapGridProps {
  tokens: string[];
  weights: number[][];
}

export function HeatmapGrid({ tokens, weights }: HeatmapGridProps) {
  if (!weights || weights.length === 0) {
    return (
      <div className="heatmap-empty">
        <p>No attention weights to display.</p>
      </div>
    );
  }

  // Color interpolation from blue (low) to red (high)
  const getColor = (value: number): string => {
    // Normalize value (attention weights are already 0-1 from softmax)
    const normalized = Math.min(Math.max(value, 0), 1);

    // Blue to white to red gradient
    if (normalized < 0.5) {
      const t = normalized * 2;
      const r = Math.round(t * 255);
      const g = Math.round(t * 255);
      const b = 255;
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = (normalized - 0.5) * 2;
      const r = 255;
      const g = Math.round((1 - t) * 255);
      const b = Math.round((1 - t) * 255);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // Truncate token display
  const formatToken = (token: string, maxLen: number = 6): string => {
    if (token.length <= maxLen) return token;
    return token.slice(0, maxLen - 1) + '...';
  };

  return (
    <div className="heatmap-container">
      <div className="attention-heatmap">
        {/* Header row */}
        <div className="heatmap-row">
          <div className="heatmap-cell corner"></div>
          {tokens.map((token, i) => (
            <div key={i} className="heatmap-cell header" title={token}>
              {formatToken(token)}
            </div>
          ))}
        </div>

        {/* Data rows */}
        {weights.map((row, i) => (
          <div key={i} className="heatmap-row">
            <div className="heatmap-cell row-header" title={tokens[i]}>
              {formatToken(tokens[i])}
            </div>
            {row.map((value, j) => (
              <div
                key={j}
                className="heatmap-cell data"
                style={{ backgroundColor: getColor(value) }}
                title={`${tokens[i]} -> ${tokens[j]}: ${value.toFixed(4)}`}
              >
                {value > 0.1 ? value.toFixed(2) : ''}
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="heatmap-legend">
        <span>Low attention</span>
        <div className="legend-gradient"></div>
        <span>High attention</span>
      </div>
    </div>
  );
}

export default HeatmapGrid;
