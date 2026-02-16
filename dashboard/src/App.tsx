/**
 * LLM Learning Lab - Main Application
 *
 * Unified dashboard for all learning tracks.
 */

import { useState } from 'react';
import { TrainingProvider, useTraining } from './context/TrainingContext';
import { FineTuningProvider } from './context/FineTuningContext';
import { Layout } from './components/layout';
import {
  PretrainingPage,
  FineTuningPage,
  AttentionPage,
  ProbingPage,
  MicroGPTPage,
} from './pages';
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
      case 'fine-tuning':
        return <FineTuningPage />;
      case 'attention':
        return <AttentionPage />;
      case 'probing':
        return <ProbingPage />;
      case 'microgpt':
        return <MicroGPTPage />;
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
      <FineTuningProvider>
        <AppContent />
      </FineTuningProvider>
    </TrainingProvider>
  );
}

export default App;
