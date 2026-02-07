from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """
    Make the package importable without requiring an editable install.

    This repo uses a src-layout (`src/cutlery_classifier/...`). Adding `./src`
    to `sys.path` keeps test execution simple and avoids dependency bloat.
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

