"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from .pipeline import run_analysis


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process microtremor array observations"
    )
    parser.add_argument("params", help="Path to params.json")
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Do not remove existing result files",
    )
    return parser


def build_visualize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neobd visualize-fk", description="Visualize an FK map"
    )
    parser.add_argument("file", help="Path to an FK CSV map")
    parser.add_argument("--output", help="Optional image output path")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open an interactive window"
    )
    parser.add_argument(
        "--db", action="store_true", help="Plot normalized power in decibels"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "visualize-fk":
        from .visualization import visualize_fk

        arguments = build_visualize_parser().parse_args(values[1:])
        destination = visualize_fk(
            arguments.file, arguments.output, not arguments.no_show, arguments.db
        )
        if destination is not None:
            print(destination)
        return 0
    arguments = build_run_parser().parse_args(values)
    run_analysis(
        arguments.params,
        replace_results=not arguments.keep_results,
        reporter=print,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
