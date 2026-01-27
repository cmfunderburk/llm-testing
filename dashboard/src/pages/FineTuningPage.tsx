/**
 * Fine-Tuning Page
 *
 * Dashboard view for QLoRA fine-tuning with real-time metrics.
 */

import { useEffect, useCallback, useMemo } from 'react';
import { useFineTuning } from '../context/FineTuningContext';
import { useWebSocket } from '../hooks/useWebSocket';
import {
  FineTuningMetrics,
  FineTuningControls,
  FineTuningLossChart,
  GenerationPanel,
} from '../components/finetuning';
import { ErrorBanner } from '../components/shared';

const WS_URL = import.meta.env.VITE_WS_URL_FT || 'ws://localhost:8000/ws/fine-tuning';

export function FineTuningPage() {
  const {
    handleWebSocketMessage,
    setConnected,
    setConnectionError,
    fetchStatus,
    error,
    status,
    loadingMessage,
  } = useFineTuning();

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

      {status.state === 'loading' && loadingMessage && (
        <div className="loading-banner">
          <div className="loading-spinner" />
          <span>{loadingMessage}</span>
        </div>
      )}

      <div className="dashboard-view">
        <div className="dashboard-main">
          <FineTuningMetrics />
          <FineTuningLossChart />
        </div>
        <div className="dashboard-right-column">
          <FineTuningControls />
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

export default FineTuningPage;
