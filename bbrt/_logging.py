"""Small logging helpers so the library uses stdlib ``logging``, not ``print``."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once (idempotent-ish via ``basicConfig``)."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, datefmt=_DATEFMT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (``get_logger(__name__)``)."""
    return logging.getLogger(name)
