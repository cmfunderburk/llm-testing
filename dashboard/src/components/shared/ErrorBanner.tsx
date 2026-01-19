/**
 * Error Banner Component
 */

interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function ErrorBanner({ message, onDismiss, action }: ErrorBannerProps) {
  return (
    <div className="error-banner">
      <span>{message}</span>
      <div className="error-banner-actions">
        {action && (
          <button className="btn btn-small" onClick={action.onClick}>
            {action.label}
          </button>
        )}
        {onDismiss && (
          <button className="btn btn-small btn-ghost" onClick={onDismiss}>
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}

export default ErrorBanner;
