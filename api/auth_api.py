"""
Auth API for terminal sign-in.

Env-based credentials (TERMINAL_USER, TERMINAL_PASSWORD, AUTH_SECRET).
JWT tokens for session; no database for MVP.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config.settings import get_settings

router = APIRouter()
security = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: str


class MeResponse(BaseModel):
    user: str


def _create_token(username: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.auth.token_expire_minutes)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(
        payload,
        settings.auth.auth_secret,
        algorithm="HS256",
    )


def _verify_token(token: str) -> str:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.auth.auth_secret,
            algorithms=["HS256"],
        )
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token")
        return str(sub)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> str:
    if not credentials or credentials.credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _verify_token(credentials.credentials)


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    settings = get_settings()
    if not settings.auth.terminal_user or not settings.auth.terminal_password:
        raise HTTPException(status_code=503, detail="Auth not configured")
    if body.username != settings.auth.terminal_user or body.password != settings.auth.terminal_password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _create_token(body.username)
    return LoginResponse(token=token, user=body.username)


@router.get("/me", response_model=MeResponse)
async def me(username: Annotated[str, Depends(get_current_user)]) -> MeResponse:
    return MeResponse(user=username)


@router.post("/logout")
async def logout() -> dict:
    return {"status": "ok"}


@router.get("/status")
async def auth_status() -> dict:
    """
    Return whether auth is configured. Frontend can allow access without login when False.
    Considered configured only when AUTH_SECRET is set and not the default placeholder.
    """
    settings = get_settings()
    secret_ok = (
        settings.auth.auth_secret
        and settings.auth.auth_secret.strip()
        and "change-me-in-production" not in settings.auth.auth_secret
    )
    configured = bool(
        settings.auth.terminal_user
        and settings.auth.terminal_password
        and secret_ok
    )
    return {"configured": configured}
