#!/usr/bin/env python3
"""Relax the non-converged clusters of an existing ML run again, with more optimizer steps.

Every setting except ``--steps`` and ``--device`` comes from the run folder's
``run_settings.json`` (see :func:`egqca.ml.rerun_nonconverged_ml`). The clusters
are reset first (rows, per-cluster result files and phonon DOS removed) and continue
from their last structure in ``cif_relaxed/`` unless ``--from-original`` is given.
For the parallel workflow use ``collect_ml_results.py --reset-nonconverged`` and
``ml_relaxation_parallel.py --start-from-relaxed`` instead.

Example usage::

    python ml_rerun_nonconverged.py /path/to/clusters_folder --model mattersim-5m
    python ml_rerun_nonconverged.py /path/to/clusters_folder --model mace-mp-small --device cuda --steps 100000
    python ml_rerun_nonconverged.py /path/to/clusters_folder --model py:my_calcs.py:mace_large --model-name mace-large
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from egqca._ml_utils import _resolve_model
    from egqca.ml import rerun_nonconverged_ml
except ImportError as exc:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))
        from egqca._ml_utils import _resolve_model
        from egqca.ml import rerun_nonconverged_ml
    else:
        raise exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relax the non-converged clusters of an existing ML run again."
    )
    parser.add_argument("clusters_folder", help="Folder containing the S*/ cluster folders and the run folder.")
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "ML model of the original run: a model string, or py:<module or file.py>:<function> "
            "for a calculator factory, as in ml_relaxation_parallel.py."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="ML_model_name of the original run. Required for a calculator factory.",
    )
    parser.add_argument(
        "--out-folder",
        default=None,
        help="Name of the run folder in clusters_folder. Default: the model name.",
    )
    parser.add_argument("--steps", type=int, default=20000, help="Maximum optimizer steps. Default: 20000.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Calculator device, e.g. cuda; not recorded with the run. Default: cpu.",
    )
    parser.add_argument(
        "--from-original",
        action="store_true",
        help="Start from the original cluster CIFs instead of the last relaxed structures.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    rerun = rerun_nonconverged_ml(
        args.clusters_folder,
        _resolve_model(args.model),
        ML_model_name=args.model_name,
        out_folder_name=args.out_folder,
        steps=args.steps,
        device=args.device,
        start_from_relaxed=not args.from_original,
    )
    print(f"Relaxed again: {len(rerun)} clusters" + (f" ({', '.join(rerun)})" if rerun else ""))


if __name__ == "__main__":
    main()
