"""
auth.py
=======
WebSocket JWT validation for the people-counting API.

Behaviour:
  * If env ``PC_JWT_SECRET`` is set, tokens MUST validate as HS256 against it.
    The decoded claims are returned as a dict.
  * If ``PC_JWT_SECRET`` is unset, any non-empty token is accepted (dev mode).
    A warning is logged on the first such request.
  * If the token is missing/empty, validation fails.

Optional claims used downstream when present:
  - ``sub``     : caller identifier
  - ``org_id``  : tenant identifier (cross-checked against request body)
  - ``user_id`` : caller user id (cross-checked against request body)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("people_counting.auth")

_warned_dev_mode = False


class AuthError(Exception):
    pass


def _try_jwt_decode(token: str, secret: str) -> dict[str, Any]:
    try:
        import jwt  # PyJWT
    except ImportError as exc:
        raise AuthError(
            "PyJWT is required for JWT validation. Install with: pip install PyJWT"
        ) from exc

    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except Exception as exc:
        raise AuthError(f"invalid token: {exc}") from exc


def validate_token(token: str | None) -> dict[str, Any]:
    """Returns the decoded claims (or ``{"_dev": True}`` in dev mode).
    Raises ``AuthError`` if the token is missing or invalid."""
    global _warned_dev_mode

    if token is None or not token.strip():
        raise AuthError("missing token")

    secret = os.environ.get("PC_JWT_SECRET", "").strip()
    if not secret:
        if not _warned_dev_mode:
            logger.warning(
                "PC_JWT_SECRET is not set: accepting any non-empty token "
                "(dev mode). Set PC_JWT_SECRET to enforce HS256 JWTs."
            )
            _warned_dev_mode = True
        return {"_dev": True, "raw": token}

    return _try_jwt_decode(token.strip(), secret)
