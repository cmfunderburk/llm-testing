/**
 * WebSocket Client Hook for Training Metrics
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { TrainingMetrics } from '../types';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: TrainingMetrics) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: TrainingMetrics | null;
  error: string | null;
  send: (data: string) => void;
  reconnect: () => void;
}

export function useWebSocket({
  url,
  onMessage,
  onOpen,
  onClose,
  onError,
  reconnectAttempts = 3,
  reconnectInterval = 2000,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<TrainingMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Store callbacks in refs to avoid reconnection when they change
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  // Update refs when callbacks change
  onMessageRef.current = onMessage;
  onOpenRef.current = onOpen;
  onCloseRef.current = onClose;
  onErrorRef.current = onError;

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
          const data = JSON.parse(event.data) as TrainingMetrics;
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
            connect();
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
    connect();

    return () => {
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
