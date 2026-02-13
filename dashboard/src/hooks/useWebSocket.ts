/**
 * WebSocket Client Hook for Training Metrics
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { TrainingMetrics } from '../types';

interface UseWebSocketOptions<T = TrainingMetrics> {
  url: string;
  onMessage?: (data: T) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface UseWebSocketReturn<T = TrainingMetrics> {
  isConnected: boolean;
  lastMessage: T | null;
  error: string | null;
  send: (data: string) => void;
  reconnect: () => void;
}

export function useWebSocket<T = TrainingMetrics>({
  url,
  onMessage,
  onOpen,
  onClose,
  onError,
  reconnectAttempts = 3,
  reconnectInterval = 2000,
}: UseWebSocketOptions<T>): UseWebSocketReturn<T> {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const connectRef = useRef<() => void>(() => {});

  // Store callbacks in refs to avoid reconnection when they change
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    onOpenRef.current = onOpen;
  }, [onOpen]);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        reconnectCountRef.current = 0;
        onOpenRef.current?.();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as T;
          setLastMessage(data);
          onMessageRef.current?.(data);
        } catch {
          if (event.data !== 'pong') {
            console.warn('Failed to parse WebSocket message:', event.data);
          }
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        onCloseRef.current?.();

        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current += 1;
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connectRef.current();
          }, reconnectInterval);
        } else {
          setError(`Failed to connect after ${reconnectAttempts} attempts`);
        }
      };

      ws.onerror = (event) => {
        setError('WebSocket connection error');
        onErrorRef.current?.(event);
      };
    } catch (e) {
      setError(`Failed to create WebSocket: ${e}`);
    }
  }, [url, reconnectAttempts, reconnectInterval]);
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const send = useCallback((data: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectCountRef.current = 0;
    setError(null);
    connect();
  }, [connect]);

  useEffect(() => {
    const connectTimer = window.setTimeout(() => {
      connect();
    }, 0);

    return () => {
      clearTimeout(connectTimer);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      send('ping');
    }, 25000);

    return () => clearInterval(pingInterval);
  }, [isConnected, send]);

  return {
    isConnected,
    lastMessage,
    error,
    send,
    reconnect,
  };
}

export default useWebSocket;
