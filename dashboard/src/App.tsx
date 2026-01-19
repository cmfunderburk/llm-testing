/**
 * LLM Learning Lab - Main Application
 *
 * Unified dashboard for all learning tracks.
 */

import { useState } from 'react';
import { TrainingProvider, useTraining } from './context/TrainingContext';
import { Layout } from './components/layout';
import { PretrainingPage, AttentionPage, ProbingPage } from './pages';
import type { Track } from './types';
import './App.css';

/**
 * Main App Content
 */
function AppContent() {
  const [activeTrack, setActiveTrack] = useState<Track>('pretraining');
  const { isConnected, connectionError } = useTraining();

  const renderPage = () => {
    switch (activeTrack) {
      case 'pretraining':
        return <PretrainingPage />;
      case 'attention':
        return <AttentionPage />;
      case 'probing':
        return <ProbingPage />;
      default:
        return <PretrainingPage />;
    }
  };

  return (
    <Layout
      activeTrack={activeTrack}
      onTrackChange={setActiveTrack}
      isConnected={isConnected}
      connectionError={connectionError}
    >
      {renderPage()}
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
