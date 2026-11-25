#!/usr/bin/env python3
"""Relax an arbitrary CIF file with an ML potential and optionally compute its phonon DOS.

Unlike ``ml_relaxation_parallel.py``, this script is not tied to the EGQCA
cluster directory structure. It operates on any CIF file and writes a relaxed
structure and (with --phonons) a phonon DOS file directly beside the input (or
into --output-dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from egqca._ml_utils import (
        _parse_optimizer,
        _parse_optimizer_kwargs,
        _ml_model_display_name,
        _relaxation_constraint_arg,
        _parse_relax_cell_mask,
        _parse_bravais_supercell,
        _parse_phonon_supercell,
        _setup_matplotlib_cache,
        DEFAULT_RELAXATION_CONSTRAINT,
        DEFAULT_BRAVAIS_TOL,
    )
    from egqca.ml import geometry_relaxation, phonons as ml_phonons
except ImportError as exc:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.is_dir():
        sys.path.insert(0, str(repo_src))
        from egqca._ml_utils import (
        _parse_optimizer,
        _parse_optimizer_kwargs,
            _ml_model_display_name,
            _relaxation_constraint_arg,
            _parse_relax_cell_mask,
            _parse_bravais_supercell,
        _parse_phonon_supercell,
            _setup_matplotlib_cache,
            DEFAULT_RELAXATION_CONSTRAINT,
            DEFAULT_BRAVAIS_TOL,
        )
        from egqca.ml import geometry_relaxation, phonons as ml_phonons
    else:
        raise exc

_setup_matplotlib_cache()

import numpy as np
from ase.io import read, write


def _default_relaxed_path(cif_file: Path, model: str, output_dir: Path | None) -> Path:
    folder = cif_file.parent if output_dir is None else output_dir
    return folder / f"{cif_file.stem}_{_ml_model_display_name(model)}_relaxed{cif_file.suffix}"


def _default_phonon_path(cif_file: Path, model: str, output_dir: Path | None) -> Path:
    folder = cif_file.parent if output_dir is None else output_dir
    return folder / f"{cif_file.stem}_{_ml_model_display_name(model)}_phonons.dos"


def _resolve_output_dir(output_dir: str | None, cif_file: Path) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir)
    return path if path.is_absolute() else cif_file.parent / path


def _run(args: argparse.Namespace) -> None:
    cif_file = Path(args.cif_file).resolve()
    if not cif_file.is_file():
        raise FileNotFoundError(f"CIF file does not exist: {cif_file}")

    output_dir = _resolve_output_dir(args.output_dir, cif_file)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    relaxed_file = (
        Path(args.relaxed_file).resolve()
        if args.relaxed_file is not None
        else _default_relaxed_path(cif_file, args.model, output_dir)
    )
    relaxed_file.parent.mkdir(parents=True, exist_ok=True)

    atoms = read(cif_file)
    print(f"----- Geometry relaxation for {cif_file.name} -----")
    atoms_relaxed = geometry_relaxation(
        atoms.copy(),
        ML_model=args.model,
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
    )
    write(relaxed_file, atoms_relaxed)
    print(f"Relaxed structure: {relaxed_file}")

    if args.phonons:
        phonon_file = (
            Path(args.phonon_file).resolve()
            if args.phonon_file is not None
            else _default_phonon_path(cif_file, args.model, output_dir)
        )
        phonon_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"----- Phonon calculation for {relaxed_file.name} -----")
        # Keeping the finite-displacement forces makes the DOS re-samplable
        # with egqca.phonon_dos_from_cache and lets a killed run resume.
        cache_dir = (
            phonon_file.parent / f"{phonon_file.stem}_cache"
            if args.phonon_cache else None
        )
        dos = ml_phonons(
            atoms_relaxed,
            ML_model=args.model,
            N=args.supercell,
            device=args.device,
            cache_dir=cache_dir,
        )
        dos_data = np.column_stack((dos.get_energies(), dos.get_weights()))
        np.savetxt(phonon_file, dos_data, fmt="%.6e", header="Energy(eV) DOS")
        print(f"Phonon DOS: {phonon_file}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relax one CIF with an EGQCA ML potential and optionally compute phonon DOS."
    )
    parser.add_argument("cif_file", help="Input CIF file")
    parser.add_argument("--model", default="mattersim-5m", help="ML model name")
    parser.add_argument("--device", default="cpu", help="Calculator device")
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
        "--output-dir",
        default=None,
        help=(
            "Output folder. Relative paths are resolved beside the CIF. "
            "Default: folder containing the CIF"
        ),
    )
    parser.add_argument("--relaxed-file", default=None, help="Explicit relaxed CIF output path")
    parser.add_argument("--phonon-file", default=None, help="Explicit phonon DOS output path")
    parser.add_argument(
        "--no-phonon-cache",
        dest="phonon_cache",
        action="store_false",
        help=(
            "Do not keep the finite-displacement forces next to the DOS file. "
            "Without the cache the DOS can only be re-sampled by repeating the "
            "whole phonon calculation."
        ),
    )
    parser.set_defaults(phonon_cache=True)
    return parser


def main() -> None:
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
