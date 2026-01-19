/**
 * Training Context - Global State Management for Pretraining Track
 */

import { createContext, useContext, useReducer, useCallback, useRef } from 'react';
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
  isConnected: boolean;
  connectionError: string | null;
  status: TrainingStatus;
  loadingMessage: string | null;  // Message shown during initialization
  metricsHistory: MetricsDataPoint[];
  generations: { step: number; epoch: number; text: string }[];
  checkpoints: CheckpointInfo[];
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
  | { type: 'SET_LOADING_MESSAGE'; payload: string | null }
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
  loadingMessage: null,
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

    case 'SET_LOADING_MESSAGE':
      return { ...state, loadingMessage: action.payload };

    case 'ADD_METRICS': {
      // Deduplicate by step - only add if this step doesn't exist
      const existingIndex = state.metricsHistory.findIndex(m => m.step === action.payload.step);
      let newHistory: MetricsDataPoint[];

      if (existingIndex >= 0) {
        // Update existing entry
        newHistory = [...state.metricsHistory];
        newHistory[existingIndex] = action.payload;
      } else {
        // Add new entry and keep sorted by step
        newHistory = [...state.metricsHistory, action.payload].sort((a, b) => a.step - b.step);
      }

      // Only update displayed status if this is the latest step (avoid flashing from history replay)
      const isLatestStep = action.payload.step >= state.status.current_step;

      return {
        ...state,
        metricsHistory: newHistory,
        status: isLatestStep ? {
          ...state.status,
          current_step: action.payload.step,
          current_epoch: action.payload.epoch,
          train_loss: action.payload.train_loss,
          tokens_seen: action.payload.tokens_seen,
          elapsed_time: action.payload.elapsed_time,
        } : state.status,
      };
    }

    case 'ADD_GENERATION': {
      // Deduplicate by step - only add if this step doesn't exist
      const genExists = state.generations.some(g => g.step === action.payload.step);
      if (genExists) {
        return state;
      }
      return {
        ...state,
        generations: [...state.generations, action.payload].sort((a, b) => a.step - b.step),
      };
    }

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
  handleWebSocketMessage: (data: TrainingMetrics) => void;
  setConnected: (connected: boolean) => void;
  setConnectionError: (error: string | null) => void;
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

  // Use ref to access current status without causing callback recreation
  const statusRef = useRef(state.status);
  statusRef.current = state.status;

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
              ...statusRef.current,
              state: data.state,
            },
          });
          // Handle loading message
          if (data.state === 'loading' && data.message) {
            dispatch({ type: 'SET_LOADING_MESSAGE', payload: data.message });
          } else if (data.state === 'running') {
            dispatch({ type: 'SET_LOADING_MESSAGE', payload: null });
          }
        }
        break;

      case 'complete':
        dispatch({
          type: 'SET_STATUS',
          payload: { ...statusRef.current, state: 'completed' },
        });
        break;

      case 'error':
        dispatch({
          type: 'SET_STATUS',
          payload: { ...statusRef.current, state: 'error' },
        });
        dispatch({ type: 'SET_ERROR', payload: data.message || 'Training error' });
        break;
    }
  }, []);

  const setConnected = useCallback((connected: boolean) => {
    dispatch({ type: 'SET_CONNECTED', payload: connected });
  }, []);

  const setConnectionError = useCallback((error: string | null) => {
    dispatch({ type: 'SET_CONNECTION_ERROR', payload: error });
  }, []);

  const startTraining = useCallback(async (config: TrainingConfig) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    dispatch({ type: 'RESET_METRICS' });

    try {
      const response = await fetch(`${API_BASE_URL}/api/pretraining/start`, {
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
      const response = await fetch(`${API_BASE_URL}/api/pretraining/pause`, {
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
      const response = await fetch(`${API_BASE_URL}/api/pretraining/resume`, {
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
      const response = await fetch(`${API_BASE_URL}/api/pretraining/stop`, {
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
      const response = await fetch(`${API_BASE_URL}/api/pretraining/status`);
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
      const response = await fetch(`${API_BASE_URL}/api/pretraining/checkpoints?config_name=${configName}`);
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
