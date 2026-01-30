"""
Legacy Dash dashboard entrypoint (deprecated).

This project has been refocused around a single FastAPI backend and a
React + D3.js web terminal. The old Dash/Plotly dashboard is no longer
part of the supported stack and this script is kept only to avoid
breaking imports.

To use the system:
  1. Start the FastAPI backend (see README).
  2. Start the React/D3 terminal frontend.
"""

if __name__ == "__main__":
    banner = "=" * 60
    print(banner)
    print("Legacy Dashboard Deprecated")
    print(banner)
    print(
        "\nThe Dash-based financial dashboard has been retired.\n"
        "Use the FastAPI-powered Bloomberg-style web terminal instead.\n"
        "See the updated README for backend/frontend startup commands."
    )
