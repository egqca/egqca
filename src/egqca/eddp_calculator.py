"""ASE Calculator wrapping AIRSS's ``repose`` eddp potential evaluator.

``repose`` (see https://github.com/airss-org) normally does its own
relaxation, which only supports a fully fixed unit cell (``-f``) or a fully
free one -- there is no equivalent of egqca's ``FixBravaisCell`` constraint.
This module instead uses ``repose -n -o -q`` as a *single-point* energy,
force and stress evaluator (``-n`` disables its internal relaxation; ``-o``
makes it print forces and stress), so an eddp potential can be plugged into
egqca's own ASE-based ``geometry_relaxation``/``run_ml_relaxation`` pipeline
exactly like MACE or MatterSim -- reusing its optimizer and constraints
(including ``FixBravaisCell``) unchanged.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

_SEED = "structure"


def _write_cell_file(path: Path, atoms) -> None:
    """Write a minimal CASTEP-style .cell file repose can read.

    Written by hand instead of via ``ase.io.write(format="castep-cell")``,
    which internally builds a full ``ase.calculators.castep.Castep``
    instance on *every* call -- regenerating its keywords JSON cache and
    probing for a CASTEP binary that doesn't exist here -- purely to
    serialize a handful of lattice/position lines. That's expensive and
    noisy (a "Generating CASTEP keywords JSON file" + "Could not determine
    the version of your CASTEP binary" warning on every relaxation step),
    and it also tries and fails to serialize any ASE constraint attached to
    ``atoms`` (e.g. FixSymmetry), which repose has no use for anyway since
    it only ever evaluates one fixed geometry per call.
    """
    lines = ["%BLOCK LATTICE_CART"]
    for vector in atoms.cell:
        lines.append(f"{vector[0]:.10f} {vector[1]:.10f} {vector[2]:.10f}")
    lines.append("%ENDBLOCK LATTICE_CART")
    lines.append("")
    lines.append("%BLOCK POSITIONS_FRAC")
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions()):
        lines.append(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
    lines.append("%ENDBLOCK POSITIONS_FRAC")
    path.write_text("\n".join(lines) + "\n")


def _read_eddp_ddps_dirname(model_path: Path) -> str:
    """The ddps subdirectory name referenced by an .eddp ensemble manifest.

    Each line of a trained ``.eddp`` file is ``<ddps_dir>/<file>.ddp
    <weight>`` -- a path (relative to repose's working directory) to one
    ensemble member plus its weight. Every line shares the same directory
    component; that directory of ``.ddp`` potential files must sit next to
    the ``.eddp`` manifest and the seed ``.cell`` file for repose to
    evaluate the potential.
    """
    with open(model_path) as f:
        first_line = f.readline()
    if not first_line.strip():
        raise ValueError(f"eddp model file {model_path} is empty.")
    rel_path = first_line.split()[0]
    if "/" not in rel_path:
        raise ValueError(
            f"eddp model file {model_path} doesn't reference a ddps subdirectory "
            f"(expected lines like '<ddps_dir>/<file>.ddp <weight>', got {rel_path!r})."
        )
    return rel_path.split("/")[0]


def _parse_repose_output(stdout: str, n_atoms: int) -> dict:
    """Parse the Energy:/Stress:/Forces: block written by ``repose -n -o -q``."""
    lines = stdout.splitlines()
    energy_eV = None
    stress = np.zeros((3, 3))
    forces = np.zeros((n_atoms, 3))

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Energy:"):
            energy_eV = float(stripped.split(":")[1].strip().split()[0])
        elif stripped.startswith("Stress:"):
            for row in range(3):
                stress[row] = [float(v) for v in lines[i + 1 + row].split()]
        elif stripped.startswith("Forces:"):
            for atom_i in range(n_atoms):
                forces[atom_i] = [float(v) for v in lines[i + 1 + atom_i].split()[1:4]]

    if energy_eV is None:
        raise RuntimeError(f"No 'Energy:' line found in repose output:\n{stdout}")

    return {
        "energy": energy_eV,
        "forces": forces,
        # repose's stress is already in eV/Angstrom^3 (constants.f90's
        # evbyang3 GPa factor is only applied to the derived Pressure: line)
        # but uses the opposite sign convention from ASE: repose.f90 prints
        # Pressure = +trace(stress)/3 (so positive trace/stress there means
        # compressive), whereas ASE expects tensile-positive stress with
        # pressure = -trace(stress)/3.
        "stress": full_3x3_to_voigt_6_stress(-stress),
    }


class EddpCalculator(Calculator):
    """Single-point ASE calculator for a trained AIRSS eddp potential.

    Runs ``repose -n -o -q`` in a persistent scratch directory holding a
    copy of ``model_path`` and a symlink to its ddps directory, rewriting
    only the ``.cell`` file on every ``calculate()`` call -- the model/ddps
    setup is done once, at construction.

    Stress sign convention has not been cross-checked against ASE's own
    (only that repose's stress is in eV/Angstrom^3, not GPa); verify against
    a reference calculator before relying on cell relaxation results.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str | Path,
        repose_bin: str = "repose",
        omp_threads: int = 1,
        rmin: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"eddp model file not found: {self.model_path}")

        ddps_dirname = _read_eddp_ddps_dirname(self.model_path)
        self.ddps_dir = self.model_path.parent / ddps_dirname
        if not self.ddps_dir.is_dir():
            raise FileNotFoundError(
                f"eddp ddps directory {self.ddps_dir} (referenced by {self.model_path}) not found."
            )

        if shutil.which(repose_bin) is None:
            raise FileNotFoundError(
                f"repose binary {repose_bin!r} not found on PATH. "
                "Add AIRSS's repose/ddp bin directories to PATH before using EddpCalculator."
            )
        self.repose_bin = repose_bin
        self.omp_threads = omp_threads
        self.rmin = rmin

        self._scratch = tempfile.TemporaryDirectory(prefix="egqca_eddp_")
        self._scratch_dir = Path(self._scratch.name)
        shutil.copy(self.model_path, self._scratch_dir / f"{_SEED}.eddp")
        (self._scratch_dir / ddps_dirname).symlink_to(self.ddps_dir)

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        cell_path = self._scratch_dir / f"{_SEED}.cell"
        _write_cell_file(cell_path, self.atoms)

        cmd = [
            self.repose_bin, "-n", "-o", "-q",
            "-ompnp", str(self.omp_threads),
            "-rmin", str(self.rmin),
            _SEED,
        ]
        result = subprocess.run(cmd, cwd=self._scratch_dir, capture_output=True, text=True)
        if result.returncode != 0 or "Energy:" not in result.stdout:
            raise RuntimeError(
                f"repose single-point evaluation failed (return code {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )

        self.results = _parse_repose_output(result.stdout, len(self.atoms))
