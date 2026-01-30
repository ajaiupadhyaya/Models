"""
Legacy Bloomberg Terminal Dash UI entrypoint (deprecated).

The interactive UI is now provided by a React + D3.js web terminal
talking to the FastAPI backend. This Dash-based UI is no longer
supported and this script is kept only as a stub for backwards
compatibility.

To run the platform:
  1. Start the FastAPI backend (see README).
  2. Start the React/D3 terminal frontend.
"""


def main() -> None:
    banner = "=" * 80
    print(banner)
    print("BLOOMBERG-STYLE TERMINAL UI HAS MOVED")
    print(banner)
    print(
        "\nThe legacy Dash-based Bloomberg terminal UI has been retired.\n"
        "Use the FastAPI backend together with the React + D3 web terminal.\n"
        "See the updated README for how to launch the new experience."
    )


if __name__ == "__main__":
    main()
