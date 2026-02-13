/**
 * Loss Curve Chart Component
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
import { useTraining } from '../../context/TrainingContext';
import type { PretrainingRunDetail } from '../../types';

interface LossChartProps {
  comparisonRuns?: PretrainingRunDetail[];
}

// Maximum points to render for performance
const MAX_CHART_POINTS = 500;
const COMPARISON_COLORS = ['#f97316', '#facc15', '#22d3ee', '#f472b6'];

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

export function LossChart({ comparisonRuns = [] }: LossChartProps) {
  const { metricsHistory } = useTraining();

  const comparisonSeries = useMemo(() => {
    return comparisonRuns.map((run, index) => {
      const shortId = run.run_id.slice(-8);
      const label = `${run.config?.config_name || 'run'}:${shortId}`;
      return {
        run,
        label,
        trainKey: `${label} Train`,
        valKey: `${label} Val`,
        color: COMPARISON_COLORS[index % COMPARISON_COLORS.length],
      };
    });
  }, [comparisonRuns]);

  const chartData = useMemo(() => {
    const rowsByStep = new Map<number, Record<string, number | string | null>>();

    const ensureRow = (step: number) => {
      if (!rowsByStep.has(step)) {
        rowsByStep.set(step, { step });
      }
      return rowsByStep.get(step)!;
    };

    for (const metric of metricsHistory) {
      const row = ensureRow(metric.step);
      row['Current Train'] = metric.train_loss;
      row['Current Val'] = metric.val_loss ?? null;
    }

    for (const series of comparisonSeries) {
      for (const metric of series.run.metrics) {
        const row = ensureRow(metric.step);
        row[series.trainKey] = metric.train_loss;
        row[series.valKey] = metric.val_loss;
      }
    }

    const sorted = Array.from(rowsByStep.values()).sort((a, b) => Number(a.step) - Number(b.step));
    return downsampleData(sorted, MAX_CHART_POINTS);
  }, [metricsHistory, comparisonSeries]);

  const hasCurrentData = metricsHistory.length > 0;
  const hasComparisonData = comparisonSeries.some((series) => series.run.metrics.length > 0);

  if (!hasCurrentData && !hasComparisonData) {
    return (
      <div className="chart-container">
        <h3>Loss Curve</h3>
        <div className="chart-empty">
          <p>No training data yet. Start training to see loss curves.</p>
        </div>
      </div>
    );
  }

  const hasAnyCurrentVal = chartData.some((row) => row['Current Val'] !== null && row['Current Val'] !== undefined);

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
              formatter={(value, name) => [typeof value === 'number' ? value.toFixed(4) : 'N/A', name]}
            />
            <Legend />

            <Line
              type="monotone"
              dataKey="Current Train"
              stroke="#4a9eff"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
              connectNulls
            />

            {hasAnyCurrentVal && (
              <Line
                type="monotone"
                dataKey="Current Val"
                stroke="#4ade80"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
                strokeDasharray="4 3"
                connectNulls={false}
              />
            )}

            {comparisonSeries.map((series) => {
              return (
                <Line
                  key={series.trainKey}
                  type="monotone"
                  dataKey={series.trainKey}
                  stroke={series.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 3 }}
                  isAnimationActive={false}
                  connectNulls
                />
              );
            })}

            {comparisonSeries.map((series) => {
              const hasVal = chartData.some(
                (row) => row[series.valKey] !== null && row[series.valKey] !== undefined,
              );
              if (!hasVal) return null;

              return (
                <Line
                  key={series.valKey}
                  type="monotone"
                  dataKey={series.valKey}
                  stroke={series.color}
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 3 }}
                  isAnimationActive={false}
                  strokeDasharray="5 3"
                  connectNulls={false}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default LossChart;
