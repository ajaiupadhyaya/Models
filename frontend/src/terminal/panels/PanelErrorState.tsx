import React from "react";

interface PanelErrorStateProps {
  /** Panel section title (e.g. "Fundamental: AAPL"). */
  title: string;
  /** Error message to show. */
  error: string;
  /** Optional hint (e.g. "Try again or ensure the service is reachable."). */
  hint?: string;
  /** If provided, a Retry button is shown. */
  onRetry?: () => void;
  /** Optional class for the section (e.g. "panel panel-main", "panel panel-main-secondary"). */
  sectionClassName?: string;
}

export const PanelErrorState: React.FC<PanelErrorStateProps> = ({
  title,
  error,
  hint,
  onRetry,
  sectionClassName = "panel panel-main",
}) => (
  <section className={sectionClassName}>
    <div className="panel-title">{title}</div>
    <div className="panel-error-box">
      <span>{error}{hint ? ` ${hint}` : ""}</span>
      {onRetry && (
        <button type="button" className="ai-button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  </section>
);
