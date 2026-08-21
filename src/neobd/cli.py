"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from .pipeline import run_analysis


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


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
    parser.add_argument(
        "--npara",
        type=_nonnegative_int,
        default=None,
        metavar="N",
        help=(
            "Override n_para from params.json; use 0 for all available CPUs "
            "(effective default: 1)"
        ),
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


def build_visualize_fv_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neobd visualize-fv", description="Visualize an FK F-V spectrum"
    )
    parser.add_argument("file", help="Path to an fv.csv file")
    parser.add_argument("--output", help="Optional image output path")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open an interactive window"
    )
    parser.add_argument(
        "--db", action="store_true", help="Plot normalized power in decibels"
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=-30.0,
        help="Lower color limit for --db (default: -30)",
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
    if values and values[0] == "visualize-fv":
        from .fv_visualization import visualize_fv

        arguments = build_visualize_fv_parser().parse_args(values[1:])
        destination = visualize_fv(
            arguments.file,
            arguments.output,
            not arguments.no_show,
            arguments.db,
            arguments.min_db,
        )
        if destination is not None:
            print(destination)
        return 0
    arguments = build_run_parser().parse_args(values)
    run_analysis(
        arguments.params,
        replace_results=not arguments.keep_results,
        reporter=print,
        n_para=arguments.npara,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
