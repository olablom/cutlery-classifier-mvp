"""Dev-only shim for cutlery-train.

Not part of the offline inference MVP.
Must remain import-safe (stdlib only).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cutlery-train",
        description="Dev-only tool (not part of offline inference MVP runtime).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print how to run the legacy dev script.",
    )
    args = parser.parse_args()

    if args.info:
        print("Legacy dev script: legacy/dev/scripts/train_type_detector.py")
        return 0

    print(
        "cutlery-train is dev-only and not included in the runtime MVP.\n"
        "Use --info to see the legacy entrypoint.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
