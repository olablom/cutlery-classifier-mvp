#!/usr/bin/env python3
"""
Console script entry point for dataset evaluation (dev tool).

This command is intentionally treated as dev-only in this repo's current scope.
If you want to run it, install the extra dependencies and use the scripts under
`scripts/`.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dev-only dataset evaluation helper (artifact-dependent)."
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--test_dir", required=False, help="Path to test dataset directory")
    p.add_argument("--model", required=False, help="Path to model checkpoint")
    return p


def main(argv: list[str] | None = None) -> None:
    _build_parser().parse_args(argv)

    try:
        from scripts.test_dataset_inference import main as legacy_main  # type: ignore

    except Exception as e:
        print(
            "cutlery-test is a dev tool and is not part of the offline inference MVP runtime.\n"
            "If you want to run it, use `python scripts/test_dataset_inference.py ...` in the repo,\n"
            "and install the dev/eval dependencies (Grad-CAM, plotting, etc.).\n"
            f"\nImport error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    legacy_main()


if __name__ == "__main__":
    main()
