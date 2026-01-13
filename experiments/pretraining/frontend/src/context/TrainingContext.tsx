/**
 * Training Context - Global State Management
 *
 * Provides centralized state for training status, metrics history,
 * and API interactions throughout the application.
 */

import { createContext, useContext, useReducer, useCallback } from 'react';
import type { ReactNode } from 'react';
import type {
  TrainingConfig,
  TrainingStatus,
  TrainingMetrics,
  CheckpointInfo,
} from '../types';

// =============================================================================
// State Types
// =============================================================================

interface MetricsDataPoint {
  step: number;
  epoch: number;
  train_loss: number;
  val_loss?: number;
  learning_rate: number;
  tokens_seen: number;
  tokens_per_sec: number;
  elapsed_time: number;
}

interface TrainingContextState {
  // Connection status
  isConnected: boolean;
  connectionError: string | null;

  // Training status
  status: TrainingStatus;

  // Metrics history for charts
  metricsHistory: MetricsDataPoint[];

  // Generated text samples
  generations: { step: number; epoch: number; text: string }[];

  // Checkpoints
  checkpoints: CheckpointInfo[];

  // UI state
  isLoading: boolean;
  error: string | null;
}

// =============================================================================
// Actions
// =============================================================================

type TrainingAction =
  | { type: 'SET_CONNECTED'; payload: boolean }
  | { type: 'SET_CONNECTION_ERROR'; payload: string | null }
  | { type: 'SET_STATUS'; payload: TrainingStatus }
  | { type: 'ADD_METRICS'; payload: MetricsDataPoint }
  | { type: 'ADD_GENERATION'; payload: { step: number; epoch: number; text: string } }
  | { type: 'SET_CHECKPOINTS'; payload: CheckpointInfo[] }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET_METRICS' };

// =============================================================================
// Reducer
// =============================================================================

const initialState: TrainingContextState = {
  isConnected: false,
  connectionError: null,
  status: {
    state: 'idle',
    current_step: 0,
    current_epoch: 0,
    total_steps: 0,
    train_loss: null,
    val_loss: null,
    tokens_seen: 0,
    elapsed_time: 0,
    config: null,
  },
  metricsHistory: [],
  generations: [],
  checkpoints: [],
  isLoading: false,
  error: null,
};

function trainingReducer(state: TrainingContextState, action: TrainingAction): TrainingContextState {
  switch (action.type) {
    case 'SET_CONNECTED':
      return { ...state, isConnected: action.payload };

    case 'SET_CONNECTION_ERROR':
      return { ...state, connectionError: action.payload };

    case 'SET_STATUS':
      return { ...state, status: action.payload };

    case 'ADD_METRICS':
      return {
        ...state,
        metricsHistory: [...state.metricsHistory, action.payload],
        status: {
          ...state.status,
          current_step: action.payload.step,
          current_epoch: action.payload.epoch,
          train_loss: action.payload.train_loss,
          tokens_seen: action.payload.tokens_seen,
          elapsed_time: action.payload.elapsed_time,
        },
      };

    case 'ADD_GENERATION':
      return {
        ...state,
        generations: [...state.generations, action.payload],
      };

    case 'SET_CHECKPOINTS':
      return { ...state, checkpoints: action.payload };

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload };

    case 'RESET_METRICS':
      return {
        ...state,
        metricsHistory: [],
        generations: [],
        status: { ...initialState.status },
      };

    default:
      return state;
  }
}

// =============================================================================
// Context
// =============================================================================

interface TrainingContextValue extends TrainingContextState {
  // Actions
  handleWebSocketMessage: (data: TrainingMetrics) => void;
  setConnected: (connected: boolean) => void;
  setConnectionError: (error: string | null) => void;

  // API calls
  startTraining: (config: TrainingConfig) => Promise<void>;
  pauseTraining: () => Promise<void>;
  resumeTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  fetchStatus: () => Promise<void>;
  fetchCheckpoints: (configName?: string) => Promise<void>;
}

const TrainingContext = createContext<TrainingContextValue | null>(null);

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// =============================================================================
// Provider
// =============================================================================

interface TrainingProviderProps {
  children: ReactNode;
}

export function TrainingProvider({ children }: TrainingProviderProps) {
  const [state, dispatch] = useReducer(trainingReducer, initialState);

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = useCallback((data: TrainingMetrics) => {
    switch (data.type) {
      case 'metrics':
        if (data.step !== undefined && data.train_loss !== undefined) {
          dispatch({
            type: 'ADD_METRICS',
            payload: {
              step: data.step,
              epoch: data.epoch || 0,
              train_loss: data.train_loss,
              val_loss: data.val_loss,
              learning_rate: data.learning_rate || 0,
              tokens_seen: data.tokens_seen || 0,
              tokens_per_sec: data.tokens_per_sec || 0,
              elapsed_time: data.elapsed_time || 0,
            },
          });
        }
        break;

      case 'generation':
        if (data.text) {
          dispatch({
            type: 'ADD_GENERATION',
            payload: {
              step: data.step || 0,
              epoch: data.epoch || 0,
              text: data.text,
            },
          });
        }
        break;

      case 'status':
        if (data.state) {
          dispatch({
            type: 'SET_STATUS',
            payload: {
              ...state.status,
              state: data.state,
            },
          });
        }
        break;

      case 'complete':
        dispatch({
          type: 'SET_STATUS',
          payload: { ...state.status, state: 'completed' },
        });
        break;

      case 'error':
        dispatch({
          type: 'SET_STATUS',
          payload: { ...state.status, state: 'error' },
        });
        dispatch({ type: 'SET_ERROR', payload: data.message || 'Training error' });
        break;
    }
  }, [state.status]);

  const setConnected = useCallback((connected: boolean) => {
    dispatch({ type: 'SET_CONNECTED', payload: connected });
  }, []);

  const setConnectionError = useCallback((error: string | null) => {
    dispatch({ type: 'SET_CONNECTION_ERROR', payload: error });
  }, []);

  // API calls
  const startTraining = useCallback(async (config: TrainingConfig) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    dispatch({ type: 'RESET_METRICS' });

    try {
      const response = await fetch(`${API_BASE_URL}/api/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to start training');
      }

      const status = await response.json();
      dispatch({ type: 'SET_STATUS', payload: status });
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);

  const pauseTraining = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/train/pause`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to pause training');
      }

      const status = await response.json();
      dispatch({ type: 'SET_STATUS', payload: status });
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);

  const resumeTraining = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/train/resume`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to resume training');
      }

      const status = await response.json();
      dispatch({ type: 'SET_STATUS', payload: status });
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);

  const stopTraining = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/train/stop`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to stop training');
      }

      const status = await response.json();
      dispatch({ type: 'SET_STATUS', payload: status });
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: (e as Error).message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/train/status`);
      if (response.ok) {
        const status = await response.json();
        dispatch({ type: 'SET_STATUS', payload: status });
      }
    } catch (e) {
      console.error('Failed to fetch status:', e);
    }
  }, []);

  const fetchCheckpoints = useCallback(async (configName: string = 'nano') => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/checkpoints?config_name=${configName}`);
      if (response.ok) {
        const checkpoints = await response.json();
        dispatch({ type: 'SET_CHECKPOINTS', payload: checkpoints });
      }
    } catch (e) {
      console.error('Failed to fetch checkpoints:', e);
    }
  }, []);

  const value: TrainingContextValue = {
    ...state,
    handleWebSocketMessage,
    setConnected,
    setConnectionError,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    fetchStatus,
    fetchCheckpoints,
  };

  return (
    <TrainingContext.Provider value={value}>
      {children}
    </TrainingContext.Provider>
  );
}

// =============================================================================
// Hook
// =============================================================================

export function useTraining(): TrainingContextValue {
  const context = useContext(TrainingContext);
  if (!context) {
    throw new Error('useTraining must be used within a TrainingProvider');
  }
  return context;
}

export default TrainingContext;
