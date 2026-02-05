"""
Shared requests session for yfinance so Yahoo Finance returns real data on cloud (e.g. Render).
Without a browser-like User-Agent, Yahoo often returns empty data or blocks requests from server IPs.
"""
import requests

_session: requests.Session | None = None

# Browser User-Agent so Yahoo Finance serves real data instead of blocking/empty on Render
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def get_yfinance_session() -> requests.Session:
    """Return a session with browser User-Agent for yf.download(session=...)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session
