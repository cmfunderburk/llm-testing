/**
 * Run History Panel
 *
 * Lists persisted pretraining runs and allows loading up to two for chart comparison.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { PretrainingRunDetail, PretrainingRunSummary } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface RunHistoryPanelProps {
  activeRunId?: string | null;
  onComparisonChange: (runs: PretrainingRunDetail[]) => void;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function formatLoss(value: number | null): string {
  if (value === null || value === undefined) return '-';
  if (value < 0.001) return value.toExponential(2);
  return value.toFixed(4);
}

export function RunHistoryPanel({ activeRunId, onComparisonChange }: RunHistoryPanelProps) {
  const [runs, setRuns] = useState<PretrainingRunSummary[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRuns = useCallback(async () => {
    setIsLoadingRuns(true);
    try {
      const res = await fetch(`${API_URL}/api/pretraining/runs?limit=100`);
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to load runs');
      }
      const data: PretrainingRunSummary[] = await res.json();
      setRuns(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load runs');
    } finally {
      setIsLoadingRuns(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  useEffect(() => {
    setSelectedRunIds((prev) => prev.filter((runId) => runs.some((run) => run.run_id === runId)));
  }, [runs]);

  useEffect(() => {
    let cancelled = false;

    const fetchDetails = async () => {
      if (selectedRunIds.length === 0) {
        onComparisonChange([]);
        return;
      }

      setIsLoadingDetails(true);
      try {
        const details = await Promise.all(
          selectedRunIds.map(async (runId) => {
            const res = await fetch(`${API_URL}/api/pretraining/runs/${runId}`);
            if (!res.ok) {
              const data = await res.json();
              throw new Error(data.detail || `Failed to load run ${runId}`);
            }
            return (await res.json()) as PretrainingRunDetail;
          }),
        );

        if (!cancelled) {
          onComparisonChange(details);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load run details');
          onComparisonChange([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingDetails(false);
        }
      }
    };

    fetchDetails();

    return () => {
      cancelled = true;
    };
  }, [selectedRunIds, onComparisonChange]);

  const toggleRunSelection = (runId: string) => {
    setSelectedRunIds((prev) => {
      if (prev.includes(runId)) {
        return prev.filter((id) => id !== runId);
      }
      if (prev.length < 2) {
        return [...prev, runId];
      }
      return [prev[1], runId];
    });
  };

  const runCountLabel = useMemo(() => {
    if (isLoadingRuns) return 'Loading...';
    return `${runs.length} runs`;
  }, [isLoadingRuns, runs.length]);

  return (
    <div className="run-history-panel">
      <div className="run-history-header">
        <h3>Run History</h3>
        <div className="run-history-actions">
          <span className="run-count-label">{runCountLabel}</span>
          <button
            type="button"
            className="btn btn-small btn-secondary"
            onClick={fetchRuns}
            disabled={isLoadingRuns}
            title="Refresh run list"
          >
            {isLoadingRuns ? '...' : '↻'}
          </button>
        </div>
      </div>

      <div className="run-history-help">
        Select up to 2 runs to overlay on the loss chart.
      </div>

      {isLoadingDetails && selectedRunIds.length > 0 && (
        <div className="run-history-loading">Loading selected runs...</div>
      )}

      {error && <div className="run-history-error">{error}</div>}

      {runs.length === 0 && !isLoadingRuns ? (
        <div className="run-history-empty">No persisted runs yet.</div>
      ) : (
        <div className="run-list">
          {runs.map((run) => {
            const selected = selectedRunIds.includes(run.run_id);
            const isActive = activeRunId === run.run_id;
            return (
              <button
                key={run.run_id}
                type="button"
                className={`run-row ${selected ? 'selected' : ''}`}
                onClick={() => toggleRunSelection(run.run_id)}
              >
                <div className="run-row-top">
                  <span className="run-row-id">{run.run_id}</span>
                  <span className={`run-row-state state-${run.state}`}>{run.state}</span>
                </div>
                <div className="run-row-meta">
                  {run.config?.config_name || 'unknown'} · {run.config?.corpus || 'unknown corpus'}
                </div>
                <div className="run-row-meta">
                  Step {run.current_step}/{run.total_steps || '?'} · Train {formatLoss(run.final_train_loss)} · Val {formatLoss(run.final_val_loss)}
                </div>
                <div className="run-row-meta">
                  Updated {formatDate(run.updated_at)}{isActive ? ' · Active' : ''}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default RunHistoryPanel;
