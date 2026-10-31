"""HeadShell command line (MIT).

Subcommands:
  headshell version
  headshell params-validate <params.json>
  headshell stats <mesh.stl>

Exit codes: 0 success, 1 invalid input, 2 usage error (argparse default).
Print JSON to stdout for version, params-validate and stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.geometry import mesh_stats, parse_binary_stl
from core.params import Params

NAME = "headshell"
VERSION = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI (version, params-validate, stats)."""
    parser = argparse.ArgumentParser(prog=NAME, description="HeadShell product kernel CLI")
    # Not required: main() returns 2 when no subcommand is given.
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print name and version as JSON")
    p_validate = sub.add_parser("params-validate", help="validate a params JSON file")
    p_validate.add_argument("path", type=Path, help="path to params JSON")
    p_stats = sub.add_parser("stats", help="print mesh statistics of a binary STL")
    p_stats.add_argument("path", type=Path, help="path to binary STL")
    return parser


def _emit(payload: dict[str, Any]) -> None:
    """Print one JSON object on stdout."""
    print(json.dumps(payload))


def _errors(exc: BaseException) -> list[str]:
    """Split an aggregated message into individual error strings."""
    parts = [p for p in str(exc).split("; ") if p]
    return parts or [type(exc).__name__]


def _cmd_version() -> int:
    _emit({"name": NAME, "version": VERSION})
    return 0


def _cmd_params_validate(path: Path) -> int:
    """Validate a params file with Params.from_dict and report JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("params file must contain a JSON object")
        Params.from_dict(data)
    except (OSError, ValueError) as exc:  # ValueError also covers json.JSONDecodeError
        _emit({"ok": False, "errors": _errors(exc)})
        return 1
    except (TypeError, AttributeError) as exc:  # wrong JSON value types inside the object
        _emit({"ok": False, "errors": _errors(exc)})
        return 1
    _emit({"ok": True, "errors": []})
    return 0


def _cmd_stats(path: Path) -> int:
    """Parse a binary STL and print mesh_stats as JSON."""
    try:
        triangles = parse_binary_stl(path)
    except (OSError, ValueError) as exc:
        _emit({"ok": False, "errors": _errors(exc)})
        return 1
    _emit(mesh_stats(triangles))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the exit code (never raises SystemExit)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage errors exit with code 2
        return exc.code if isinstance(exc.code, int) else 0
    if args.command is None:
        return 2
    if args.command == "version":
        return _cmd_version()
    if args.command == "params-validate":
        return _cmd_params_validate(args.path)
    return _cmd_stats(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
