/**
 * Fine-Tuning Loss Chart Component
 */

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useFineTuning } from '../../context/FineTuningContext';

const MAX_CHART_POINTS = 500;

function downsampleData<T>(data: T[], maxPoints: number): T[] {
  if (data.length <= maxPoints) return data;

  const step = Math.ceil(data.length / maxPoints);
  const sampled: T[] = [];

  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
  }

  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }

  return sampled;
}

export function FineTuningLossChart() {
  const { metricsHistory } = useFineTuning();

  const chartData = useMemo(() => {
    const mapped = metricsHistory.map((m) => ({
      step: m.step,
      'Train Loss': m.train_loss,
      'Eval Loss': m.eval_loss ?? null,
    }));
    return downsampleData(mapped, MAX_CHART_POINTS);
  }, [metricsHistory]);

  if (chartData.length === 0) {
    return (
      <div className="chart-container">
        <h3>Loss Curve</h3>
        <div className="chart-empty">
          <p>No fine-tuning data yet. Start training to see loss curves.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <h3>Loss Curve</h3>
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3f5f" />
            <XAxis
              dataKey="step"
              stroke="#8b9dc3"
              fontSize={12}
              tickFormatter={(value) => `${value}`}
            />
            <YAxis
              stroke="#8b9dc3"
              fontSize={12}
              domain={['auto', 'auto']}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2b47',
                border: '1px solid #2a3f5f',
                borderRadius: '4px',
                color: '#e4e4e4',
              }}
              labelFormatter={(label) => `Step ${label}`}
              formatter={(value) => [typeof value === 'number' ? value.toFixed(4) : 'N/A', '']}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="Train Loss"
              stroke="#4a9eff"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="Eval Loss"
              stroke="#4ade80"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default FineTuningLossChart;
