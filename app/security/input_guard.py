from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

def _load_guard() -> Any | None:
    """Try to import llm-guard scanner; return None if unavailable."""
    try:
        from llm_guard import scan_prompt
        return scan_prompt
    except Exception:
        logger.debug("llm-guard not available; input guard will use fallback")
        return None


_SCAN_PROMPT: Any | None = None
_SCAN_PROMPT_LOADED: bool = False

def _get_scan_prompt() -> Any | None:
    """Lazy-load llm-guard scanner function."""
    global _SCAN_PROMPT, _SCAN_PROMPT_LOADED
    if not _SCAN_PROMPT_LOADED:
        _SCAN_PROMPT = _load_guard()
        _SCAN_PROMPT_LOADED = True
    return _SCAN_PROMPT

_scanners: list[Any] | None = None

def _get_scanners() -> list[Any]:
    """Lazy-build llm-guard input scanner instances from settings."""
    global _scanners
    if _scanners is not None:
        return _scanners

    from llm_guard.input_scanners import PromptInjection, Toxicity, BanTopics, TokenLimit

    _scanners = [
        PromptInjection(threshold=settings.prompt_injection_threshold),
        Toxicity(threshold=settings.toxicity_threshold),
        BanTopics(
            topics=["violence", "self-harm", "illegal activities"],
            threshold=settings.toxicity_threshold,
        ),
        TokenLimit(limit=4096),
    ]
    return _scanners


def scan_input(text: str) -> dict[str, Any]:
    scan_fn = _get_scan_prompt()
    if scan_fn is None:
        return {
            "is_safe": True,
            "failed_checks": [],
            "scores": {},
            "sanitized": text,
        }

    try:
        scanners = _get_scanners()
        sanitized, is_valid, scores = scan_fn(scanners, text)
        failed = [name for name, valid in is_valid.items() if not valid]
        return {
            "is_safe": len(failed) == 0,
            "failed_checks": failed,
            "scores": dict(scores),
            "sanitized": str(sanitized),
        }
    except Exception:
        logger.exception("llm-guard scan failed; allowing input")
        return {
            "is_safe": True,
            "failed_checks": [],
            "scores": {},
            "sanitized": text,
        }


def check_input_safe(text: str) -> tuple[bool, str | None]:
    result = scan_input(text)
    if result["is_safe"]:
        return True, None

    checks = ", ".join(result["failed_checks"]) if result["failed_checks"] else "security scan"
    return False, f"Input blocked by {checks}"

    