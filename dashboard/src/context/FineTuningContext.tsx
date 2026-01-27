/**
 * Fine-Tuning Context - Global State Management for Fine-Tuning Track
 */

import { createContext, useContext, useReducer, useCallback, useRef } from 'react';
import type { ReactNode } from 'react';
import type {
  FineTuningConfig,
  FineTuningStatus,
  FineTuningMetrics,
  AdapterCheckpointInfo,
} from '../types';

// =============================================================================
// State Types
// =============================================================================

interface FineTuningMetricsDataPoint {
  step: number;
  epoch: number;
  train_loss: number;
  eval_loss?: number;
  learning_rate?: number;
}

interface FineTuningContextState {
  isConnected: boolean;
  connectionError: string | null;
  status: FineTuningStatus;
  loadingMessage: string | null;
  metricsHistory: FineTuningMetricsDataPoint[];
  checkpoints: AdapterCheckpointInfo[];
  isLoading: boolean;
  error: string | null;
}

// =============================================================================
// Actions
// =============================================================================

type FineTuningAction =
  | { type: 'SET_CONNECTED'; payload: boolean }
  | { type: 'SET_CONNECTION_ERROR'; payload: string | null }
  | { type: 'SET_STATUS'; payload: FineTuningStatus }
  | { type: 'SET_LOADING_MESSAGE'; payload: string | null }
  | { type: 'ADD_METRICS'; payload: FineTuningMetricsDataPoint }
  | { type: 'SET_CHECKPOINTS'; payload: AdapterCheckpointInfo[] }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET_METRICS' };

// =============================================================================
// Reducer
// =============================================================================

const initialState: FineTuningContextState = {
  isConnected: false,
  connectionError: null,
  status: {
    state: 'idle',
    current_step: 0,
    total_steps: 0,
    current_epoch: 0,
    train_loss: null,
    eval_loss: null,
    learning_rate: null,
    elapsed_time: 0,
    config: null,
    trainable_params: null,
    total_params: null,
  },
  loadingMessage: null,
  metricsHistory: [],
  checkpoints: [],
  isLoading: false,
  error: null,
};

function fineTuningReducer(state: FineTuningContextState, action: FineTuningAction): FineTuningContextState {
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
      const existingIndex = state.metricsHistory.findIndex(m => m.step === action.payload.step);
      let newHistory: FineTuningMetricsDataPoint[];

      if (existingIndex >= 0) {
        newHistory = [...state.metricsHistory];
        newHistory[existingIndex] = { ...newHistory[existingIndex], ...action.payload };
      } else {
        newHistory = [...state.metricsHistory, action.payload].sort((a, b) => a.step - b.step);
      }

      const isLatestStep = action.payload.step >= state.status.current_step;

      return {
        ...state,
        metricsHistory: newHistory,
        status: isLatestStep ? {
          ...state.status,
          current_step: action.payload.step,
          current_epoch: action.payload.epoch,
          train_loss: action.payload.train_loss,
          ...(action.payload.learning_rate !== undefined && { learning_rate: action.payload.learning_rate }),
        } : state.status,
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
        status: { ...initialState.status },
      };

    default:
      return state;
  }
}

// =============================================================================
// Context
// =============================================================================

interface FineTuningContextValue extends FineTuningContextState {
  handleWebSocketMessage: (data: FineTuningMetrics) => void;
  setConnected: (connected: boolean) => void;
  setConnectionError: (error: string | null) => void;
  startTraining: (config: FineTuningConfig) => Promise<void>;
  pauseTraining: () => Promise<void>;
  resumeTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;
  fetchStatus: () => Promise<void>;
  fetchCheckpoints: () => Promise<void>;
}

const FineTuningContext = createContext<FineTuningContextValue | null>(null);

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// =============================================================================
// Provider
// =============================================================================

interface FineTuningProviderProps {
  children: ReactNode;
}

export function FineTuningProvider({ children }: FineTuningProviderProps) {
  const [state, dispatch] = useReducer(fineTuningReducer, initialState);
  const statusRef = useRef(state.status);
  statusRef.current = state.status;

  const handleWebSocketMessage = useCallback((data: FineTuningMetrics) => {
    switch (data.type) {
      case 'metrics':
        if (data.step !== undefined && data.train_loss !== undefined) {
          dispatch({
            type: 'ADD_METRICS',
            payload: {
              step: data.step,
              epoch: data.epoch || 0,
              train_loss: data.train_loss,
              learning_rate: data.learning_rate,
            },
          });
        }
        break;

      case 'validation':
        if (data.step !== undefined && data.eval_loss !== undefined) {
          // Update existing metrics point with eval_loss, or add new one
          dispatch({
            type: 'ADD_METRICS',
            payload: {
              step: data.step,
              epoch: data.epoch || 0,
              train_loss: statusRef.current.train_loss || 0,
              eval_loss: data.eval_loss,
            },
          });
          dispatch({
            type: 'SET_STATUS',
            payload: { ...statusRef.current, eval_loss: data.eval_loss },
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
              ...(data.total_steps !== undefined && { total_steps: data.total_steps }),
              ...(data.trainable_params !== undefined && { trainable_params: data.trainable_params }),
              ...(data.total_params !== undefined && { total_params: data.total_params }),
            },
          });
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
          payload: {
            ...statusRef.current,
            state: 'completed',
            ...(data.elapsed_time !== undefined && { elapsed_time: data.elapsed_time }),
          },
        });
        break;

      case 'error':
        dispatch({
          type: 'SET_STATUS',
          payload: { ...statusRef.current, state: 'error' },
        });
        dispatch({ type: 'SET_ERROR', payload: data.message || 'Fine-tuning error' });
        break;
    }
  }, []);

  const setConnected = useCallback((connected: boolean) => {
    dispatch({ type: 'SET_CONNECTED', payload: connected });
  }, []);

  const setConnectionError = useCallback((error: string | null) => {
    dispatch({ type: 'SET_CONNECTION_ERROR', payload: error });
  }, []);

  const startTraining = useCallback(async (config: FineTuningConfig) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    dispatch({ type: 'RESET_METRICS' });

    try {
      const response = await fetch(`${API_BASE_URL}/api/fine-tuning/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to start fine-tuning');
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
      const response = await fetch(`${API_BASE_URL}/api/fine-tuning/pause`, { method: 'POST' });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to pause');
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
      const response = await fetch(`${API_BASE_URL}/api/fine-tuning/resume`, { method: 'POST' });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to resume');
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
      const response = await fetch(`${API_BASE_URL}/api/fine-tuning/stop`, { method: 'POST' });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to stop');
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
      const response = await fetch(`${API_BASE_URL}/api/fine-tuning/status`);
      if (response.ok) {
        const status = await response.json();
        dispatch({ type: 'SET_STATUS', payload: status });
      }
    } catch (e) {
      console.error('Failed to fetch fine-tuning status:', e);
    }
  }, []);

  const fetchCheckpoints = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/fine-tuning/checkpoints`);
      if (response.ok) {
        const checkpoints = await response.json();
        dispatch({ type: 'SET_CHECKPOINTS', payload: checkpoints });
      }
    } catch (e) {
      console.error('Failed to fetch adapter checkpoints:', e);
    }
  }, []);

  const value: FineTuningContextValue = {
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
    <FineTuningContext.Provider value={value}>
      {children}
    </FineTuningContext.Provider>
  );
}

// =============================================================================
// Hook
// =============================================================================

export function useFineTuning(): FineTuningContextValue {
  const context = useContext(FineTuningContext);
  if (!context) {
    throw new Error('useFineTuning must be used within a FineTuningProvider');
  }
  return context;
}

export default FineTuningContext;
