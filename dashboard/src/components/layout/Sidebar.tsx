/**
 * Sidebar Component
 *
 * Track-based navigation sidebar.
 */

import type { Track } from '../../types';

interface TrackInfo {
  id: Track;
  label: string;
  icon: string;
  description: string;
}

const tracks: TrackInfo[] = [
  { id: 'pretraining', label: 'Pretraining', icon: 'P', description: 'Train GPT from scratch' },
  { id: 'attention', label: 'Attention', icon: 'A', description: 'Visualize attention patterns' },
  { id: 'probing', label: 'Probing', icon: 'R', description: 'Analyze representations' },
];

interface SidebarProps {
  activeTrack: Track;
  onTrackChange: (track: Track) => void;
}

export function Sidebar({ activeTrack, onTrackChange }: SidebarProps) {
  return (
    <nav className="sidebar">
      <div className="sidebar-section">
        <h3 className="sidebar-section-title">Learning Tracks</h3>
        <ul className="nav-list">
          {tracks.map((track) => (
            <li key={track.id}>
              <button
                className={`nav-item ${activeTrack === track.id ? 'active' : ''}`}
                onClick={() => onTrackChange(track.id)}
              >
                <span className="nav-icon">{track.icon}</span>
                <div className="nav-text">
                  <span className="nav-label">{track.label}</span>
                  <span className="nav-description">{track.description}</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="sidebar-footer">
        <div className="version-info">v0.2.0</div>
      </div>
    </nav>
  );
}

export default Sidebar;
