"""Guard the boundary between the canonical pipeline and the demo fork.

The repository has two stage implementations:

* ``lib/pipeline/`` -- the canonical, maintained pipeline used by the batch
  path (``scripts/batch_infer.py`` -> ``lib/pipeline/batch/`` ->
  ``lib/pipeline/stages/``) and the orchestrator.
* ``scripts/scripts_test_video/`` + ``scripts/batch_worker.py`` -- a divergent
  fork kept alive only for the demo/visualization entrypoints (``demo.py``,
  ``demo_offline.py``, ``app.py``) and the legacy infiller subprocess shelled
  out by ``lib/pipeline/exporters/webdataset_features.py``.

These two must stay decoupled: the canonical ``lib/pipeline`` package must never
*import* the fork, or the divergence silently leaks back into maintained code.
This test fails if it ever does.

A subprocess path string (``PROJECT_ROOT / "scripts" / "batch_worker.py"``) or a
prose comment referencing the fork is allowed -- only real ``import``
statements are forbidden. We parse the AST so strings and comments never trip
the guard.
"""

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_PIPELINE = PROJECT_ROOT / "lib" / "pipeline"

FORBIDDEN_PREFIXES = (
    "scripts.scripts_test_video",
    "scripts.batch_worker",
)

# Files allowed to import the fork: the demo/visualization entrypoints and the
# fork itself. Paths are relative to the repo root.
ALLOWED_IMPORTERS = {
    "demo.py",
    "demo_offline.py",
    "app.py",
}


def _imported_modules(tree: ast.AST):
    """Yield the dotted module name of every import statement in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in FORBIDDEN_PREFIXES
    )


class ForkBoundaryTests(unittest.TestCase):
    def test_canonical_pipeline_does_not_import_demo_fork(self):
        offenders = []
        for path in LIB_PIPELINE.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            rel = path.relative_to(PROJECT_ROOT)
            for module in _imported_modules(tree):
                if _is_forbidden(module):
                    offenders.append(f"{rel} -> import {module}")

        self.assertEqual(
            offenders,
            [],
            "Canonical lib/pipeline must not import the demo fork "
            f"(scripts.scripts_test_video / scripts.batch_worker): {offenders}",
        )

    def test_fork_importers_are_only_the_known_demo_entrypoints(self):
        """Document/enforce exactly which top-level files may import the fork.

        If a new file starts importing the fork, this fails so the boundary stays
        an intentional decision rather than accidental coupling.
        """
        importers = set()
        for path in PROJECT_ROOT.rglob("*.py"):
            rel = path.relative_to(PROJECT_ROOT)
            parts = rel.parts
            if any(p in {".git", "__pycache__", "deprecated", "thirdparty"} for p in parts):
                continue
            # The fork is allowed to import itself.
            if parts[0] == "scripts":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for module in _imported_modules(tree):
                if _is_forbidden(module):
                    importers.add(str(rel))
                    break

        unexpected = importers - ALLOWED_IMPORTERS
        self.assertEqual(
            unexpected,
            set(),
            "Unexpected non-script files import the demo fork; either route them "
            f"through lib/pipeline or add them to ALLOWED_IMPORTERS: {sorted(unexpected)}",
        )


if __name__ == "__main__":
    unittest.main()
