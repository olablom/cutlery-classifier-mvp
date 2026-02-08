#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cutlery-test",
        description="Dev-only tool (not part of offline inference MVP runtime).",
    )
    p.add_argument(
        "--info",
        action="store_true",
        help="Print how to run the legacy dev script.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    msg = (
        "cutlery-test is dev-only and not part of the offline inference MVP runtime.\n"
        "Use the legacy script instead:\n"
        "  python legacy/dev/scripts/test_dataset_inference.py ...\n"
    )

    if args.info:
        print(msg)
        return

    # For any non-help invocation: treat as dev-only command.
    print(msg, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
