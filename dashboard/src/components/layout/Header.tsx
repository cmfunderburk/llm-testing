/**
 * Header Component
 *
 * Top navigation bar with branding and connection status.
 */

import type { Track } from '../../types';

interface HeaderProps {
  activeTrack: Track;
  isConnected: boolean;
  connectionError: string | null;
}

export function Header({ activeTrack, isConnected, connectionError }: HeaderProps) {
  const trackLabels: Record<Track, string> = {
    pretraining: 'Pretraining',
    'fine-tuning': 'Fine-Tuning',
    attention: 'Attention',
    probing: 'Probing',
  };

  return (
    <header className="header">
      <div className="header-brand">
        <h1>LLM Learning Lab</h1>
        <span className="header-subtitle">{trackLabels[activeTrack]} Track</span>
      </div>

      <div className="header-status">
        <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {connectionError ? 'Error' : isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
