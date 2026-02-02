import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Terminal error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "var(--bg)",
            color: "var(--text)",
            fontFamily: "var(--font-mono)",
            padding: 24,
          }}
        >
          <h2 style={{ color: "var(--accent)", marginBottom: 12 }}>Something went wrong</h2>
          <p style={{ color: "var(--text-soft)", marginBottom: 16, maxWidth: 400, textAlign: "center" }}>
            The terminal encountered an error. Reload the page to try again.
          </p>
          <button
            type="button"
            className="ai-button"
            onClick={() => window.location.reload()}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
