"""Tests for structured logging setup and the live-quiet vprint fix."""

import io
import logging
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline import logging_setup
from hawor.utils import logging as hawor_logging


class _QuietEnv:
    def __enter__(self):
        self._saved = os.environ.get("HAWOR_QUIET")
        self._saved_level = os.environ.get("HAWOR_LOG_LEVEL")
        return self

    def set(self, value):
        if value is None:
            os.environ.pop("HAWOR_QUIET", None)
        else:
            os.environ["HAWOR_QUIET"] = value

    def __exit__(self, *exc):
        for key, val in (("HAWOR_QUIET", self._saved), ("HAWOR_LOG_LEVEL", self._saved_level)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


class LiveQuietTests(unittest.TestCase):
    def test_vprint_reads_live_env_not_frozen(self):
        with _QuietEnv() as env:
            env.set(None)
            self.assertFalse(hawor_logging.is_quiet())
            env.set("1")
            # Live read: changed after import -> reflects immediately.
            self.assertTrue(hawor_logging.is_quiet())
            env.set("0")
            self.assertFalse(hawor_logging.is_quiet())

    def test_vprint_silent_when_quiet(self):
        with _QuietEnv() as env:
            env.set("1")
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                hawor_logging.vprint("should not appear")
            finally:
                sys.stdout = old
            self.assertEqual(buf.getvalue(), "")

    def test_vprint_emits_when_not_quiet(self):
        with _QuietEnv() as env:
            env.set("0")
            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                hawor_logging.vprint("hello")
            finally:
                sys.stdout = old
            self.assertIn("hello", buf.getvalue())


class ConfigureLoggingTests(unittest.TestCase):
    def test_level_from_env_quiet_is_warning(self):
        with _QuietEnv() as env:
            env.set("1")
            os.environ.pop("HAWOR_LOG_LEVEL", None)
            self.assertEqual(logging_setup._level_from_env(), "WARNING")

    def test_explicit_log_level_wins(self):
        with _QuietEnv() as env:
            env.set("1")
            os.environ["HAWOR_LOG_LEVEL"] = "debug"
            self.assertEqual(logging_setup._level_from_env(), "DEBUG")

    def test_configure_is_idempotent_handler_count(self):
        logger = logging_setup.configure_logging(force=True)
        before = len([h for h in logger.handlers if getattr(h, "_hawor_handler", False)])
        logging_setup.configure_logging()
        logging_setup.configure_logging()
        after = len([h for h in logger.handlers if getattr(h, "_hawor_handler", False)])
        self.assertEqual(before, after)
        self.assertEqual(after, 1)

    def test_warning_visible_even_when_quiet(self):
        # WARNING+ must surface regardless of quiet (that's the whole point).
        with _QuietEnv() as env:
            env.set("1")
            os.environ.pop("HAWOR_LOG_LEVEL", None)
            logger = logging_setup.configure_logging(force=True)
            buf = io.StringIO()
            handler = logging.StreamHandler(buf)
            handler._hawor_handler = True
            logger.addHandler(handler)
            try:
                logger.warning("danger")
                logger.info("routine")
            finally:
                logger.removeHandler(handler)
            out = buf.getvalue()
            self.assertIn("danger", out)
            self.assertNotIn("routine", out)  # INFO suppressed under quiet


if __name__ == "__main__":
    unittest.main()
