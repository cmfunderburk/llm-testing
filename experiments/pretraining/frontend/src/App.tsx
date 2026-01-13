/**
 * LLM Pretraining Lab - Main Application
 *
 * React application for interactive GPT pretraining with real-time visualization.
 */

import { useState, useEffect } from 'react';
import { TrainingProvider, useTraining } from './context/TrainingContext';
import { useWebSocket } from './hooks/useWebSocket';
import { Layout } from './components/layout';
import {
  TrainingControls,
  MetricsDisplay,
  LossChart,
  LearningRateChart,
  TokenCounter,
  SampleTextDisplay,
  ProgressIndicators,
} from './components/dashboard';
import './App.css';

// WebSocket URL
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/training';

/**
 * Main App Content - uses hooks that require TrainingProvider
 */
function AppContent() {
  const [activeSection, setActiveSection] = useState('dashboard');

  const {
    handleWebSocketMessage,
    setConnected,
    setConnectionError,
    fetchStatus,
    error,
  } = useTraining();

  // Set up WebSocket connection
  const { isConnected, reconnect } = useWebSocket({
    url: WS_URL,
    onMessage: handleWebSocketMessage,
    onOpen: () => {
      setConnected(true);
      setConnectionError(null);
    },
    onClose: () => {
      setConnected(false);
    },
    onError: () => {
      setConnectionError('WebSocket connection failed');
    },
    reconnectAttempts: 3,
    reconnectInterval: 2000,
  });

  // Fetch initial status on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Update connected state
  useEffect(() => {
    setConnected(isConnected);
  }, [isConnected, setConnected]);

  // Render section content
  const renderContent = () => {
    switch (activeSection) {
      case 'dashboard':
        return (
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
            </div>
          </div>
        );

      case 'training':
        return (
          <div className="training-view">
            <h2>Training Configuration</h2>
            <TrainingControls />
          </div>
        );

      case 'generate':
        return (
          <div className="generate-view">
            <h2>Text Generation</h2>
            <p className="coming-soon">Generation interface coming in Phase 6</p>
          </div>
        );

      case 'analysis':
        return (
          <div className="analysis-view">
            <h2>Model Analysis</h2>
            <p className="coming-soon">Analysis tools coming in Phase 6</p>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Layout activeSection={activeSection} onSectionChange={setActiveSection}>
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          {!isConnected && (
            <button onClick={reconnect} className="btn btn-small">
              Reconnect
            </button>
          )}
        </div>
      )}
      {renderContent()}
    </Layout>
  );
}

/**
 * Root App Component with Providers
 */
function App() {
  return (
    <TrainingProvider>
      <AppContent />
    </TrainingProvider>
  );
}

export default App;
