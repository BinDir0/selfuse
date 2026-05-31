import os

# NOTE: kept for backward compatibility. This module-level constant is captured
# at import time, so it does NOT reflect runtime changes to $HAWOR_QUIET. Prefer
# `is_quiet()` (live) or the structured logger in `lib.pipeline.logging_setup`.
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"

_TRUTHY = {"1", "true", "yes", "y", "on"}


def is_quiet():
    """Live read of $HAWOR_QUIET (re-evaluated on every call, never frozen)."""
    return os.environ.get("HAWOR_QUIET", "").strip().lower() in _TRUTHY


def vprint(*args, **kwargs):
    """Print routine progress only when not in quiet mode.

    Reads the quiet flag live so toggling $HAWOR_QUIET at runtime takes effect.
    For warnings/errors that must surface even under quiet mode, use the logger
    from `lib.pipeline.logging_setup` instead of vprint.
    """
    if not is_quiet():
        print(*args, **kwargs)
