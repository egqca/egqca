#!/usr/bin/env python3
"""Aggregate per-cluster *_ml_result.dat files into EGQCA input files.

After running ``ml_relaxation_parallel.py`` for every cluster (e.g. via a
SLURM job array), call this script once to collect the individual result files
into the ``clusters_<model>.dat`` and ``volume_<model>.dat`` files that EGQCA
expects.

Example usage::

    python collect_ml_results.py /path/to/clusters_folder/pet-oam-xl
    python collect_ml_results.py /path/to/clusters_folder/pet-oam-xl --model pet-oam-xl
    python collect_ml_results.py /path/to/run_folder --clusters-folder /path/to/clusters_folder

To relax the non-converged clusters again in the parallel workflow, collect with
``--reset-nonconverged``, queue the clusters listed in ``rerun_clusters.txt`` with
``ml_relaxation_parallel.py`` (e.g. with a higher ``--steps``), and collect again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from egqca.ml import collect_ml_results, reset_ml_clusters
except ImportError as exc:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))
        from egqca.ml import collect_ml_results, reset_ml_clusters
    else:
        raise exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-cluster *_ml_result.dat files into "
            "clusters_<model>.dat and volume_<model>.dat."
        )
    )
    parser.add_argument(
        "run_folder",
        help=(
            "Model run folder written by ml_relaxation_parallel.py, e.g. "
            "/path/to/clusters_folder/mattersim-5m/. Result files are read from "
            "its per_cluster/ subfolder and the aggregated files are written into it."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "ML model name used during relaxation. "
            "Auto-detected from result filenames when omitted; "
            "required if multiple models are present."
        ),
    )
    parser.add_argument(
        "--clusters-folder",
        default=None,
        help=(
            "Root cluster folder containing S*/S*_i*_w*/ subdirectories and "
            "clusters_template.dat. Default: the parent of run_folder."
        ),
    )
    parser.add_argument(
        "--clusters-file",
        default=None,
        help="Output filename for clusters data. Default: clusters_<model>.dat",
    )
    parser.add_argument(
        "--volume-file",
        default=None,
        help="Output filename for volume data. Default: volume_<model>.dat",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Path for the convergence summary log. "
            "Default: convergence_<model>.log in output_dir"
        ),
    )
    parser.add_argument(
        "--clusters-template",
        default=None,
        metavar="FILE",
        help=(
            "Clusters template to use instead of <clusters folder>/clusters_template.dat. "
            "Reads permutable_sites, sites, Nat, compound and the list of all clusters."
        ),
    )
    parser.add_argument(
        "--permutable-sites",
        type=int,
        default=None,
        help=(
            "Number of permutable sites n (x = n_j / n). Default: the clusters template "
            "(--clusters-template, else <clusters folder>/clusters_template.dat, else "
            "DFT/clusters.dat of an older tree), else the highest S* folder number. A value "
            "that differs from the template is an error."
        ),
    )
    parser.add_argument(
        "--reset-nonconverged",
        action="store_true",
        help=(
            "After collecting, remove the rows, result files and phonon DOS of the "
            "non-converged clusters and list them in <clusters folder>/rerun_clusters.txt, "
            "so that ml_relaxation_parallel.py relaxes them again (e.g. with a higher "
            "--steps). Collect again afterwards."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    clusters_path, volume_path, log_path = collect_ml_results(
        args.run_folder,
        clusters_folder=args.clusters_folder,
        model=args.model,
        clusters_template=args.clusters_template,
        ml_clusters_filename=args.clusters_file,
        ml_volume_filename=args.volume_file,
        log_file=args.log_file,
        permutable_sites=args.permutable_sites,
    )
    print(f"Clusters file:     {clusters_path}")
    print(f"Volume file:       {volume_path}")
    print(f"Convergence log:   {log_path}")
    if args.reset_nonconverged:
        reset_ml_clusters(
            args.run_folder,
            args.model,
            clusters_folder=args.clusters_folder,
            ml_clusters_filename=args.clusters_file,
            ml_volume_filename=args.volume_file,
        )


if __name__ == "__main__":
    main()
