import os

QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"


def vprint(*args, **kwargs):
    """Print only when HAWOR_QUIET is not set."""
    if not QUIET_MODE:
        print(*args, **kwargs)
