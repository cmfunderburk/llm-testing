/**
 * Header Component
 *
 * Top navigation bar with branding and connection status indicator.
 */

import { useTraining } from '../../context/TrainingContext';

export function Header() {
  const { isConnected, connectionError, status } = useTraining();

  return (
    <header className="header">
      <div className="header-brand">
        <h1>LLM Pretraining Lab</h1>
        <span className="header-subtitle">Interactive GPT Training Dashboard</span>
      </div>

      <div className="header-status">
        <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {connectionError ? 'Error' : isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {status.state !== 'idle' && (
          <div className={`training-state training-state-${status.state}`}>
            {status.state.charAt(0).toUpperCase() + status.state.slice(1)}
          </div>
        )}
      </div>
    </header>
  );
}

export default Header;
