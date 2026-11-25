"""Start-up banner and input summary printed by the main egqca entry points."""
from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

LOGO = r"""
  _____ ____  ___   ____  _
 | ____/ ___|/ _ \ / ___|/ \
 |  _|| |  _| | | | |   / _ \
 | |__| |_| | |_| | |__/ ___ \
 |_____\____|\__\_\\__/_/   \_\
"""
TITLE = "Extended Generalized Quasichemical Approximation"
#AUTHORS = "Pedro Nunes Ferreira, Dominik Spath, Luka Wibmer"


def _get_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("egqca")
    except PackageNotFoundError:
        pass
    # not installed, e.g. run from the source tree: read pyproject.toml
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        import tomllib
        with open(pyproject, "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, ImportError, ValueError):
        return "unknown"


__version__ = _get_version()

_depth = threading.local()


def header_lines() -> list[str]:
    return [LOGO.strip("\n"), "", f" {TITLE}", #f" Authors: {AUTHORS}"
     f" Version: {__version__}"]


def format_summary(task: str, items: Iterable[tuple[str, object]]) -> list[str]:
    items = [(k, v) for k, v in items if v is not None]
    width = max((len(k) for k, _ in items), default=0)
    lines = [f" {task}  ({datetime.now():%Y-%m-%d %H:%M:%S})", " " + "-" * 60]
    lines += [f"   {k:<{width}} : {v}" for k, v in items]
    return lines


def print_banner(task: str, items: Iterable[tuple[str, object]]) -> list[str]:
    """Print logo, authors, version and the input summary; return the summary lines."""
    summary = format_summary(task, items)
    print("\n".join(header_lines() + [""] + summary + [""]), flush=True)
    return summary


def describe_grid(values, unit: str = "") -> str | None:
    """Short description of a 1D grid: 'min ... max (N points)'."""
    if values is None:
        return None
    import numpy as np
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return "empty"
    if arr.size <= 8:
        return ", ".join(f"{v:g}" for v in arr) + (f" {unit}" if unit else "")
    return f"{arr.min():g} ... {arr.max():g}{(' ' + unit) if unit else ''} ({arr.size} points)"


def entry_point(task: str, summary: Callable[[inspect.BoundArguments], Iterable[tuple[str, object]]] | Iterable[str]):
    """Decorator printing the banner and an input summary when ``func`` is called.

    ``summary`` is either a list of argument names to show, or a function that
    takes the bound call arguments and returns ``(label, value)`` pairs. Calls
    made from inside another entry point print nothing, so a workflow prints
    its banner once.
    """

    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            depth = getattr(_depth, "value", 0)
            if depth == 0:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if callable(summary):
                    # may replace arguments, e.g. turn an iterator into a list it can read twice
                    items = list(summary(bound))
                else:
                    items = [(name, bound.arguments[name]) for name in summary]
                print_banner(task, items)
                args, kwargs = bound.args, bound.kwargs
            _depth.value = depth + 1
            try:
                return func(*args, **kwargs)
            finally:
                _depth.value = depth

        return wrapper

    return decorator
