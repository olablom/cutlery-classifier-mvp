#!/usr/bin/env python3
"""
Console script entry point for offline inference.

This is intentionally lightweight: importing `cutlery_classifier` should not
pull in training/plotting dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..inference.inferencer import CutleryInferencer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run offline inference on one image.")
    p.add_argument("--image", required=True, help="Path to an input image")
    p.add_argument(
        "--model", required=True, help="Path to a model checkpoint (.pth from trainer)"
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device (default: auto)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to print (default: 3)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON (default: false)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {model_path}")

    device = None if args.device == "auto" else args.device
    inferencer = CutleryInferencer(model_path=str(model_path), device=device)
    result: dict[str, Any] = inferencer.predict(image_path, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    top = result["predictions"][0]
    print(f"pred: {top['class_name']} ({top['percentage']:.1f}%)")
    print(f"time_ms: {result['inference_time_ms']:.2f}")
    for i, p in enumerate(result["predictions"], 1):
        print(f"{i}. {p['class_name']}: {p['percentage']:.2f}%")


if __name__ == "__main__":
    main()
