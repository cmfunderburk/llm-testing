/**
 * Pretraining Page
 *
 * Dashboard view for GPT pretraining with real-time metrics.
 */

import { useEffect, useCallback, useMemo } from 'react';
import { useTraining } from '../context/TrainingContext';
import { useWebSocket } from '../hooks/useWebSocket';
import {
  MetricsDisplay,
  TrainingControls,
  LossChart,
  LearningRateChart,
  TokenCounter,
  SampleTextDisplay,
  ProgressIndicators,
  GenerationPanel,
} from '../components/pretraining';
import { ErrorBanner } from '../components/shared';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/training';

export function PretrainingPage() {
  const {
    handleWebSocketMessage,
    setConnected,
    setConnectionError,
    fetchStatus,
    error,
    status,
  } = useTraining();

  // Stable callbacks to prevent WebSocket reconnection loops
  const handleOpen = useCallback(() => {
    setConnected(true);
    setConnectionError(null);
  }, [setConnected, setConnectionError]);

  const handleClose = useCallback(() => {
    setConnected(false);
  }, [setConnected]);

  const handleError = useCallback(() => {
    setConnectionError('WebSocket connection failed');
  }, [setConnectionError]);

  // Memoize WebSocket options to prevent recreation
  const wsOptions = useMemo(() => ({
    url: WS_URL,
    onMessage: handleWebSocketMessage,
    onOpen: handleOpen,
    onClose: handleClose,
    onError: handleError,
    reconnectAttempts: 3,
    reconnectInterval: 2000,
  }), [handleWebSocketMessage, handleOpen, handleClose, handleError]);

  const { isConnected, reconnect } = useWebSocket(wsOptions);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return (
    <div className="pretraining-page">
      {error && (
        <ErrorBanner
          message={error}
          action={!isConnected ? { label: 'Reconnect', onClick: reconnect } : undefined}
        />
      )}

      <div className="dashboard-view">
        <div className="dashboard-main">
          <MetricsDisplay />
          <LossChart />
          <div className="dashboard-charts-row">
            <LearningRateChart />
            <TokenCounter />
          </div>
          <SampleTextDisplay />
        </div>
        <div className="dashboard-right-column">
          <TrainingControls />
          <ProgressIndicators />
          {status.state !== 'idle' && (
            <div className={`training-state-badge training-state-${status.state}`}>
              {status.state.charAt(0).toUpperCase() + status.state.slice(1)}
            </div>
          )}
          <GenerationPanel />
        </div>
      </div>
    </div>
  );
}

export default PretrainingPage;
