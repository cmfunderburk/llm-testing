/**
 * Layout Component
 *
 * Main layout wrapper with header, sidebar, and content area.
 */

import type { ReactNode } from 'react';
import type { Track } from '../../types';
import { Header } from './Header';
import { Sidebar } from './Sidebar';

interface LayoutProps {
  children: ReactNode;
  activeTrack: Track;
  onTrackChange: (track: Track) => void;
  isConnected: boolean;
  connectionError: string | null;
}

export function Layout({
  children,
  activeTrack,
  onTrackChange,
  isConnected,
  connectionError,
}: LayoutProps) {
  return (
    <div className="app-layout">
      <Header
        activeTrack={activeTrack}
        isConnected={isConnected}
        connectionError={connectionError}
      />
      <div className="app-body">
        <Sidebar activeTrack={activeTrack} onTrackChange={onTrackChange} />
        <main className="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}

export default Layout;
