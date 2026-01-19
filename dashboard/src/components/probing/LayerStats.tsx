/**
 * Layer Stats Component
 *
 * Displays activation statistics for each layer.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { LayerActivationData } from '../../types';

interface LayerStatsProps {
  layers: LayerActivationData[];
}

export function LayerStats({ layers }: LayerStatsProps) {
  // Prepare chart data
  const chartData = layers.map((layer) => {
    const preAttn = layer.positions.find(p => p.position === 'pre_attn');
    const postFfn = layer.positions.find(p => p.position === 'post_ffn');

    return {
      layer: `L${layer.layer_idx}`,
      'Pre-Attention Norm': preAttn?.norm || 0,
      'Post-FFN Norm': postFfn?.norm || 0,
      'Attention Contrib': layer.attention_contrib_norm || 0,
      'FFN Contrib': layer.ffn_contrib_norm || 0,
    };
  });

  return (
    <div className="layer-stats">
      <div className="stats-card">
        <h3>Activation Norms by Layer</h3>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3f5f" />
              <XAxis dataKey="layer" stroke="#8b9dc3" fontSize={12} />
              <YAxis stroke="#8b9dc3" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2b47',
                  border: '1px solid #2a3f5f',
                  borderRadius: '4px',
                  color: '#e4e4e4',
                }}
              />
              <Legend />
              <Bar dataKey="Pre-Attention Norm" fill="#4a9eff" />
              <Bar dataKey="Post-FFN Norm" fill="#4ade80" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="stats-table">
        <h3>Detailed Statistics</h3>
        <table>
          <thead>
            <tr>
              <th>Layer</th>
              <th>Position</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Norm</th>
            </tr>
          </thead>
          <tbody>
            {layers.map((layer) =>
              layer.positions.map((pos, i) => (
                <tr key={`${layer.layer_idx}-${pos.position}`}>
                  {i === 0 && (
                    <td rowSpan={layer.positions.length}>{layer.layer_idx}</td>
                  )}
                  <td>{pos.position}</td>
                  <td>{pos.mean.toFixed(4)}</td>
                  <td>{pos.std.toFixed(4)}</td>
                  <td>{pos.norm.toFixed(2)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default LayerStats;
