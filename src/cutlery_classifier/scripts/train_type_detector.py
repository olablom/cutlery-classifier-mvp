#!/usr/bin/env python3
"""
Console script entry point for training (dev tool).

Training is intentionally out-of-scope for the offline inference MVP runtime.
This entrypoint remains as a shim so existing installs don't break.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Dev-only training entrypoint (not part of offline inference MVP)."
    )


def main(argv: list[str] | None = None) -> None:
    _build_parser().parse_args(argv)

    try:
        from scripts.train_type_detector import main as legacy_main  # type: ignore

    except Exception as e:
        print(
            "cutlery-train is a dev tool and is not part of the offline inference MVP runtime.\n"
            "If you want to train, use `python scripts/train_type_detector.py ...` in the repo\n"
            "and install the dev dependencies.\n"
            f"\nImport error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    legacy_main()


if __name__ == "__main__":
    main()
