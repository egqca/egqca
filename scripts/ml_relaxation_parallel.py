#!/usr/bin/env python3
"""Run the EGQCA ML relaxation (and optional phonon DOS) for one or more cluster subfolders.

This script is intended for parallel cluster workflows, e.g. a job array with one
cluster or a small group of clusters per job. Several clusters given in one call
are relaxed one after another in the same process, so the ML model is loaded only
once. Each cluster writes only its own per-cluster files, avoiding concurrent writes
to shared ``clusters_*.dat`` or ``volume_*.dat`` files. Use
``collect_ml_results.py`` afterwards to aggregate the per-cluster result files
into the final EGQCA input format.

A cluster is skipped (no relaxation, no ML model load) if it already has a
per-cluster result file, or if it is already present in a shared
``clusters_<model>.dat``/``volume_<model>.dat`` pair in ``output_dir`` -- e.g.
left behind by a prior ``run_ml_relaxation`` (sequential) run on the same run
folder. This lets a parallel condor sweep resume a sequential run, or itself
be resubmitted, without recomputing already-completed clusters.
"""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:
    from egqca._ml_utils import (
        _Tee,
        _ml_model_display_name,
        _resolve_model,
        _atomic_write,
        _original_cluster_cif,
        _cluster_info_from_id,
        _relaxation_constraint_arg,
        _parse_relax_cell_mask,
        _parse_bravais_supercell,
        _parse_phonon_supercell,
        _parse_optimizer,
        _parse_optimizer_kwargs,
        _setup_matplotlib_cache,
        _runtime_line,
        _model_identity,
        _check_model_identity,
        _run_settings,
        _phonon_settings,
        _check_run_settings,
        DEFAULT_RELAXATION_CONSTRAINT,
        DEFAULT_BRAVAIS_TOL,
    )
    from egqca.ml import (
        _RESULT_HEADER,
        _read_completed_ids,
        _read_ml_result_file,
        _ml_model_name,
        _ml_model_description,
        _PHONONS_ONLY_SEPARATOR,
        _ml_model_source,
        _resolve_permutable_sites,
        geometry_relaxation,
        phonons as ml_phonons,
    )
    from egqca.constants import Ry_to_mev
except ImportError as exc:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))
        from egqca._ml_utils import (
            _Tee,
            _ml_model_display_name,
            _resolve_model,
            _atomic_write,
            _original_cluster_cif,
            _cluster_info_from_id,
            _relaxation_constraint_arg,
            _parse_relax_cell_mask,
            _parse_bravais_supercell,
            _parse_phonon_supercell,
            _parse_optimizer,
            _parse_optimizer_kwargs,
            _setup_matplotlib_cache,
            _runtime_line,
            _model_identity,
            _check_model_identity,
            _run_settings,
            _phonon_settings,
            _check_run_settings,
            DEFAULT_RELAXATION_CONSTRAINT,
            DEFAULT_BRAVAIS_TOL,
        )
        from egqca.ml import (
            _RESULT_HEADER,
            _read_completed_ids,
            _read_ml_result_file,
            _ml_model_name,
            _ml_model_description,
            _PHONONS_ONLY_SEPARATOR,
            _ml_model_source,
            _resolve_permutable_sites,
            geometry_relaxation,
            phonons as ml_phonons,
        )
        from egqca.constants import Ry_to_mev
    else:
        raise exc

_setup_matplotlib_cache()

import time

import numpy as np
from ase.io import read, write


def _result_file(cluster_dir: Path, model: str, output_dir: Path) -> Path:
    return output_dir / "per_cluster" / f"{cluster_dir.name}_{_ml_model_display_name(model)}_ml_result.dat"


def _write_result(
    path: Path,
    cluster_id: str,
    n_j: int | None,
    Nat: int,
    x: float | None,
    g_j: int | None,
    energy_ry: float,
    latpar,
    atoms,
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    converged = atoms.info.get("egqca_ml_converged", True)
    final_fmax = atoms.info.get("egqca_ml_final_fmax", float("nan"))
    with _atomic_write(path) as f:
        f.write(f"# ML model: {model}\n")
        f.write(f"# {_RESULT_HEADER}\n")
        f.write(
            "\t".join(
                map(str, [cluster_id, n_j, Nat, x, g_j, energy_ry, *latpar, converged, final_fmax])
            )
            + "\n"
        )


def _run_one_cluster(args: argparse.Namespace) -> None:
    if geometry_relaxation is None:
        raise RuntimeError(
            "egqca.geometry_relaxation is unavailable. Install the ML extras, "
            "for example: pip install 'egqca[upet]'"
        )

    cluster_dir = Path(args.cluster_dir).resolve()
    if not cluster_dir.is_dir():
        raise NotADirectoryError(f"Cluster directory does not exist: {cluster_dir}")

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "per_cluster").mkdir(parents=True, exist_ok=True)
    (output_dir / "ML_logs" / "clusters").mkdir(parents=True, exist_ok=True)

    cluster_id = cluster_dir.name
    # as in run_ml_relaxation: ML_model is the calculator (or model string),
    # ML_model_name names the files and identifies the run
    ml_model = _resolve_model(args.model)
    model_name = _ml_model_name(ml_model, args.model_name)
    model_description = _ml_model_description(ml_model, args.model_name)
    safe_model = _ml_model_display_name(model_name)
    result_file = _result_file(cluster_dir, model_name, output_dir)
    clusters_file = output_dir / f"clusters_{safe_model}.dat"
    volume_file = output_dir / f"volume_{safe_model}.dat"
    relaxed_cif_file = output_dir / "cif_relaxed" / f"{cluster_id}_{safe_model}.cif"
    phonon_path = output_dir / "phonons" / f"{cluster_id}.dos"

    # file-based models (eddp): refuse to add results of another model file to output_dir
    model_identity = _model_identity(ml_model)
    _check_model_identity(output_dir, model_identity)
    if model_identity:
        print(f"Model file: {model_identity['path']} (sha256 {model_identity['sha256']})")
    # every job of a run folder must use the settings the folder was started with
    _check_run_settings(
        output_dir,
        _run_settings(
            model=model_name,
            pressure_GPa=args.pressure_gpa,
            relaxation_constraint=args.relaxation_constraint,
            CellFilter_mask=args.cell_filter_mask,
            bravais_supercell=args.bravais_supercell,
            fix_bravais_positions=args.fix_bravais_positions,
            bravais_tol=args.bravais_tol,
            fmax=args.fmax,
            optimizer=args.optimizer,
            optimizer_kwargs=args.optimizer_kwargs,
            model_source=_ml_model_source(ml_model),
        ),
        _phonon_settings(args.supercell) if args.phonons else None,
    )

    # A prior run of this script, or a prior run_ml_relaxation (sequential) run
    # sharing this output_dir, may have already relaxed this cluster -- either
    # recorded here in per_cluster/, or in the shared clusters_<model>.dat /
    # volume_<model>.dat files. Recognizing the latter lets a parallel condor
    # sweep pick up where a sequential run left off without recomputing
    # already-done clusters.
    relaxation_recorded = (
        (result_file.exists() and _read_ml_result_file(result_file) is not None)
        or (
            cluster_id in _read_completed_ids(clusters_file)
            and cluster_id in _read_completed_ids(volume_file)
        )
    )
    # The relaxation may be recorded but --phonons was turned on afterwards (or
    # a previous phonon step crashed), leaving the DOS file missing even though
    # the cluster looks "done". Don't skip in that case.
    phonons_pending = args.phonons and not phonon_path.exists()

    if relaxation_recorded and not phonons_pending:
        print(f"Already complete: {cluster_id}. Skipping.")
        return

    # Reuse the already-relaxed structure instead of paying for a full
    # relaxation again when only the phonon DOS is missing.
    resume_phonons_only = relaxation_recorded and phonons_pending and relaxed_cif_file.exists()
    if relaxation_recorded and phonons_pending and not resume_phonons_only:
        print(
            f"{cluster_id}: relaxation already recorded but phonon DOS "
            f"{phonon_path} is missing, and no relaxed CIF ({relaxed_cif_file}) "
            f"is available to resume from; relaxing again from scratch."
        )

    if not resume_phonons_only:
        cif_file = _original_cluster_cif(cluster_dir)
        if cif_file is None:
            raise FileNotFoundError(f"No original S*.cif file found in {cluster_dir}")
        if args.start_from_relaxed and relaxed_cif_file.exists():
            print(f"{cluster_id}: starting from the relaxed structure {relaxed_cif_file}")
            atoms = read(relaxed_cif_file)
        else:
            atoms = read(cif_file)

        # n as in run_ml_relaxation: clusters template, else --permutable-sites, else
        # the highest S* folder; the cluster folder is <clusters folder>/S<n_j>/<cluster_id>
        permutable_sites, _ = _resolve_permutable_sites(cluster_dir.parent.parent, args.permutable_sites)
        n_j, x, g_j = _cluster_info_from_id(cluster_id, permutable_sites)
        Nat = len(atoms)

        save_cif = relaxed_cif_file if not args.dont_save_relaxed_cif else None

        def _relax():
            print(f"----- Geometry relaxation for {cluster_id} -----")
            return geometry_relaxation(
                atoms.copy(),
                ML_model=ml_model,
                ML_model_name=args.model_name,
                pressure_GPa=args.pressure_gpa,
                relaxation_constraint=args.relaxation_constraint,
                CellFilter_mask=args.cell_filter_mask,
                bravais_supercell=args.bravais_supercell,
                fix_bravais_positions=args.fix_bravais_positions,
                bravais_tol=args.bravais_tol,
                steps=args.steps,
                device=args.device,
                fmax=args.fmax,
                optimizer=args.optimizer,
                optimizer_kwargs=args.optimizer_kwargs,
                save_cif=save_cif,
            )

    def _do_work():
        t_total0 = time.monotonic()
        if resume_phonons_only:
            print(f"----- Reusing relaxed structure for {cluster_id} (phonons pending) -----")
            atoms_relaxed = read(relaxed_cif_file)
            t_relax = 0.0
        else:
            atoms_relaxed = _relax()
            t_relax = time.monotonic() - t_total0

            energy_ry = atoms_relaxed.get_potential_energy() / (Ry_to_mev/1000)
            latpar = np.hstack((atoms_relaxed.cell.lengths(), atoms_relaxed.cell.angles()))

            _write_result(
                result_file,
                cluster_id,
                n_j, Nat, x, g_j,
                energy_ry,
                latpar,
                atoms_relaxed,
                model_description,
            )

        if args.phonons:
            print(f"----- Phonon calculation for {cluster_id} -----")
            t_phonon0 = time.monotonic()
            # Keeping the finite-displacement forces makes the DOS re-samplable
            # with egqca.phonon_dos_from_cache and lets a killed job resume.
            cache_dir = (
                phonon_path.parent / f"{cluster_id}_cache"
                if args.phonon_cache else None
            )
            dos = ml_phonons(
                atoms_relaxed,
                ML_model=ml_model,
                ML_model_name=args.model_name,
                N=args.supercell,
                device=args.device,
                cache_dir=cache_dir,
            )
            dos_data = np.column_stack((dos.get_energies(), dos.get_weights()))
            phonon_path.parent.mkdir(parents=True, exist_ok=True)
            with _atomic_write(phonon_path) as f_dos:
                np.savetxt(f_dos, dos_data, fmt="%.6e", header="Energy(eV) DOS")
            t_phonon = time.monotonic() - t_phonon0
            print(f"\nRelaxation runtime: {t_relax:.2f} s")
            print(f"Phonon runtime: {t_phonon:.2f} s")

        # last line of the cluster log; collect_ml_results.py reads the runtime from it
        print(_runtime_line(time.monotonic() - t_total0))

    # the cluster log is always written: collect_ml_results reads the settings header,
    # the Bravais check and the runtime from it
    log_file = output_dir / "ML_logs" / "clusters" / f"{cluster_dir.name}.log"
    # a phonons-only pass keeps the relaxation's log (runtime, Bravais check)
    with open(log_file, "a" if resume_phonons_only else "w") as f:
        if resume_phonons_only:
            f.write(f"\n{_PHONONS_ONLY_SEPARATOR} (reusing {relaxed_cif_file.name}) =====\n")
        f.write(f"cluster_dir:           {cluster_dir}\n")
        f.write(f"ML_model:              {model_description}\n")
        f.write(f"device:                {args.device if isinstance(ml_model, str) else 'set by the user-supplied calculator'}\n")
        f.write(f"steps:                 {args.steps}\n")
        f.write(f"fmax:                  {args.fmax}\n")
        f.write(f"optimizer:             {args.optimizer.__name__}\n")
        f.write(f"optimizer_kwargs:      {args.optimizer_kwargs or {}}\n")
        f.write(f"relaxation_constraint: {args.relaxation_constraint}\n")
        f.write(f"cell_filter_mask:      {args.cell_filter_mask}\n")
        f.write(f"bravais_supercell:     {'x'.join(str(n) for n in args.bravais_supercell)}\n")
        f.write(f"fix_bravais_positions: {args.fix_bravais_positions}\n")
        f.write(f"bravais_tol:           {args.bravais_tol}\n")
        f.write(f"phonons:               {args.phonons}\n")
        f.write(f"phonon_supercell:      {args.supercell}\n")
        f.write(f"phonon_cache:          {args.phonon_cache}\n")
        f.write(f"pressure_GPa:          {args.pressure_gpa}\n")
        f.write("\n")
        with redirect_stdout(_Tee(sys.stdout, f)), redirect_stderr(_Tee(sys.stderr, f)):
            _do_work()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EGQCA ML relaxation for one or more cluster subfolders."
    )
    parser.add_argument(
        "cluster_dirs",
        nargs="+",
        metavar="cluster_dir",
        help=(
            "Cluster subfolder(s), e.g. /path/to/clusters/S2/S2_i03_w4. Several clusters "
            "are relaxed one after another in this process with one ML model load."
        ),
    )
    parser.add_argument(
        "output_dir",
        help=(
            "Output directory for this model run (e.g. /path/to/clusters/mattersim-5m/). "
            "Subdirectories per_cluster/ and ML_logs/clusters/ are created automatically."
        ),
    )
    parser.add_argument(
        "--model",
        default="mattersim-5m",
        help=(
            "ML_model: a model string (e.g. mattersim-5m, mace-medium-mpa-0, eddp:<path>), or a "
            "custom calculator as py:<module or file.py>:<function>, a function returning an ASE "
            "calculator when called without arguments. Default: mattersim-5m."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "ML_model_name: name of the model in the run folder, file names and logs. Required "
            "for a custom calculator, defaults to the model string otherwise. Only the class of a "
            "custom calculator is recorded, so give each checkpoint its own name."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Calculator device")
    parser.add_argument("--steps", type=int, default=5000, help="Maximum optimizer steps. Default: 5000.")
    parser.add_argument("--fmax", type=float, default=1e-4, help="Force threshold in eV/Å")
    parser.add_argument(
        "--optimizer",
        type=_parse_optimizer,
        default="FIRE",
        metavar="NAME",
        help="ASE optimizer class name, e.g. FIRE, BFGS, LBFGS, GPMin. Default: FIRE.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        type=_parse_optimizer_kwargs,
        default=None,
        metavar="JSON",
        help=(
            "Arguments of the optimizer as a JSON object, e.g. '{\"maxstep\": 0.1}' "
            "(optimizer_kwargs of run_ml_relaxation). Default: none."
        ),
    )
    parser.add_argument(
        "--pressure-gpa",
        type=float,
        default=None,
        help="Optional scalar pressure in GPa",
    )
    parser.add_argument(
        "--relaxation-constraint",
        type=_relaxation_constraint_arg,
        default=DEFAULT_RELAXATION_CONSTRAINT,
        metavar="CONSTRAINT",
        help=(
            "Space-group / Bravais-lattice ASE constraint. One of:\n"
            "  'FixSymmetry' (default)  — preserve space group via ASE FixSymmetry;\n"
            "  'auto'                   — fix Bravais lattice auto-detected from the\n"
            "                               structure's own cell via FixBravaisCell;\n"
            "  'rhombohedral', 'RHL', etc. — fix Bravais lattice/crystal system\n"
            "                               via FixBravaisCell;\n"
            "  'none'                   — no ASE constraint.\n"
            "Use --cell-filter-mask to restrict individual strain components."
        ),
    )
    parser.add_argument(
        "--cell-filter-mask",
        type=_parse_relax_cell_mask,
        default=None,
        dest="cell_filter_mask",
        metavar="xx,yy,zz,yz,xz,xy",
        help=(
            "Strain mask for FrechetCellFilter: six comma-separated bools for the Cartesian\n"
            "strain components xx,yy,zz,yz,xz,xy (ASE Voigt order), e.g.\n"
            "'True,True,True,False,False,False'. True = free to relax, False = fixed.\n"
            "Not the cell parameters a,b,c,alpha,beta,gamma: the two only coincide for an\n"
            "orthogonal, axis-aligned cell. Default: all free."
        ),
    )
    parser.add_argument(
        "--permutable-sites",
        type=int,
        default=None,
        help=(
            "Number of permutable sites n (x = n_j / n). Default: <clusters folder>/"
            "clusters_template.dat (or DFT/clusters.dat of an older tree), else the "
            "highest S* folder number. Needed only if neither is right (no template and "
            "only a sub-range of S* folders). A value that differs from the template is an error."
        ),
    )
    parser.add_argument(
        "--dont-save-relaxed-cif",
        dest="dont_save_relaxed_cif",
        action="store_true",
        default=False,
        help="Skip writing the relaxed CIF to <output_dir>/cif_relaxed/. By default the CIF is always saved.",
    )
    parser.add_argument(
        "--bravais-supercell",
        type=_parse_bravais_supercell,
        default=[1, 1, 1],
        metavar="NxNxN",
        help=(
            "Diagonal supercell sizes for the Bravais cell constraint, e.g. '2x2x1'. "
            "Each lattice vector is divided by the corresponding size before applying "
            "Bravais symmetry operations. Only used when --relaxation-constraint is a "
            "Bravais key or crystal system. Default: 1x1x1 (primitive cell)."
        ),
    )
    parser.add_argument(
        "--start-from-relaxed",
        action="store_true",
        help=(
            "Start the relaxation from the cluster's structure in <output_dir>/cif_relaxed/ "
            "if there is one, e.g. to continue non-converged clusters reset with "
            "collect_ml_results.py --reset-nonconverged, instead of from its original CIF."
        ),
    )
    parser.add_argument(
        "--fix-bravais-positions",
        dest="fix_bravais_positions",
        action="store_true",
        default=False,
        help=(
            "With a Bravais lattice relaxation constraint, also freeze the atomic positions "
            "at their starting fractional coordinates (no internal relaxation, only the cell "
            "relaxes). Off by default (cell/stress constraint only, positions relax freely)."
        ),
    )
    parser.add_argument(
        "--bravais-tol",
        type=float,
        default=DEFAULT_BRAVAIS_TOL,
        help=(
            "Relative tolerance of the Bravais lattice detection for the Bravais cell "
            "constraint. A start cell within it of a higher-symmetry lattice is projected "
            "onto that lattice. Only used when --relaxation-constraint is 'auto', a Bravais "
            "key or a crystal system. Default: %(default)g (about 1 %%)."
        ),
    )
    parser.add_argument(
        "--phonons",
        action="store_true",
        default=False,
        help="Compute phonon DOS after relaxation",
    )
    parser.add_argument(
        "--supercell",
        type=_parse_phonon_supercell,
        default=2,
        metavar="N | N1,N2,N3",
        help=(
            "Phonon supercell size (only used with --phonons). Either a single "
            "integer N for an N×N×N supercell, or three comma-separated integers "
            "'N1,N2,N3'. Default: 2."
        ),
    )
    parser.add_argument(
        "--no-phonon-cache",
        dest="phonon_cache",
        action="store_false",
        help=(
            "Do not keep the finite-displacement forces in "
            "<output_dir>/phonons/<cluster_id>_cache/. Without the cache the DOS "
            "can only be re-sampled by repeating the whole phonon calculation."
        ),
    )
    parser.set_defaults(phonon_cache=True)
    return parser


def _run_clusters(args: argparse.Namespace) -> list[str]:
    """Relax every cluster in ``args.cluster_dirs``; return the clusters that failed.

    A failing cluster does not stop the others. The ML calculator is cached per
    (model, device) in egqca.ml, so it is loaded once for all clusters.
    """
    failed = []
    for i, cluster_dir in enumerate(args.cluster_dirs, 1):
        if len(args.cluster_dirs) > 1:
            print(f"\n===== Cluster {i}/{len(args.cluster_dirs)}: {cluster_dir} =====", flush=True)
        cluster_args = copy.copy(args)
        cluster_args.cluster_dir = cluster_dir
        try:
            _run_one_cluster(cluster_args)
        except Exception:
            if len(args.cluster_dirs) == 1:
                raise
            traceback.print_exc()
            failed.append(cluster_dir)
    return failed


def main() -> None:
    failed = _run_clusters(_build_parser().parse_args())
    if failed:
        print(f"\n{len(failed)} cluster(s) failed:\n  " + "\n  ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
