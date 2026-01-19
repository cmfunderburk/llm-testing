/**
 * Learning Rate Chart Component
 */

import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useTraining } from '../../context/TrainingContext';

// Maximum points to render for performance
const MAX_CHART_POINTS = 500;

function downsampleData<T>(data: T[], maxPoints: number): T[] {
  if (data.length <= maxPoints) return data;

  const step = Math.ceil(data.length / maxPoints);
  const sampled: T[] = [];

  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
  }

  // Always include the last point for current state
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }

  return sampled;
}

export function LearningRateChart() {
  const { metricsHistory } = useTraining();

  const chartData = useMemo(() => {
    const mapped = metricsHistory.map((m) => ({
      step: m.step,
      lr: m.learning_rate,
    }));
    return downsampleData(mapped, MAX_CHART_POINTS);
  }, [metricsHistory]);

  const maxLR = useMemo(() => {
    if (chartData.length === 0) return 0;
    return Math.max(...chartData.map((d) => d.lr));
  }, [chartData]);

  if (chartData.length === 0) {
    return (
      <div className="chart-container chart-small">
        <h3>Learning Rate Schedule</h3>
        <div className="chart-empty">
          <p>No data yet.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-container chart-small">
      <h3>Learning Rate Schedule</h3>
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3f5f" />
            <XAxis
              dataKey="step"
              stroke="#8b9dc3"
              fontSize={10}
              tickFormatter={(value) => `${value}`}
            />
            <YAxis
              stroke="#8b9dc3"
              fontSize={10}
              domain={[0, maxLR * 1.1]}
              tickFormatter={(value) => value.toExponential(0)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2b47',
                border: '1px solid #2a3f5f',
                borderRadius: '4px',
                color: '#e4e4e4',
                fontSize: '12px',
              }}
              labelFormatter={(label) => `Step ${label}`}
              formatter={(value) => [typeof value === 'number' ? value.toExponential(2) : 'N/A', 'LR']}
            />
            <Area
              type="monotone"
              dataKey="lr"
              stroke="#818cf8"
              fill="#818cf8"
              fillOpacity={0.3}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default LearningRateChart;
