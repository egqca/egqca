"""Shared utilities for ML relaxation scripts and the ml module."""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import tempfile
import re
import socket
import warnings
from pathlib import Path

import numpy as np
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from ase.spacegroup.symmetrize import symmetrize_rank2
from ase import Atoms

DEFAULT_RELAXATION_CONSTRAINT = "FixSymmetry"
DEFAULT_BRAVAIS_TOL = 1e-2   # relative tolerance of the Bravais lattice detection


class _Tee:
    """Write to multiple streams simultaneously.

    Streams that are already closed are skipped instead of raising. Libraries with
    asynchronous logging keep a reference to the stream they were given: mattersim
    registers its loguru handler (``enqueue=True``) on first use, which happens
    inside the redirect of the first cluster, so it keeps writing to that first
    ``_Tee`` after the per-cluster log file has been closed. Those late messages
    still reach the real stdout/stderr, they are only dropped for the closed file.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.write(text)
            except ValueError:
                # stream was closed between the check and the write
                pass
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.flush()
            except ValueError:
                pass


def _runtime_line(seconds: float) -> str:
    """Total-runtime line of a per-cluster log; parsed by ``ml._extract_log_runtime``."""
    return f"Runtime: {seconds:.2f} s"


def _setup_matplotlib_cache() -> None:
    """Ensure a writable matplotlib config dir exists (needed on shared HPC nodes).

    Keeps a usable ``MPLCONFIGDIR``; otherwise points it to a cache folder in the
    system temp dir. If that cannot be created either, matplotlib falls back to
    its own temporary config dir.
    """
    configured = os.environ.get("MPLCONFIGDIR")
    if configured is not None and Path(configured).is_dir() and os.access(configured, os.W_OK):
        return
    cache_dir = Path(tempfile.gettempdir()) / "egqca_matplotlib_cache"
    try:
        cache_dir.mkdir(exist_ok=True)
    except OSError:
        return
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def _safe_ml_name(value: str) -> str:
    return re.sub(r"[\\/]+", "_", value)


def _ml_model_display_name(ML_model: str) -> str:
    """Short, filesystem-safe name for ML_model, used in folder/file/log names.

    eddp models are identified by ``ML_model="eddp:<path/to/model.eddp>"`` --
    using that verbatim would embed the whole model path in every generated
    name, so it's shortened to ``"eddp-<file name without .eddp>"``. Two eddp
    files with the same name are told apart by :func:`_check_model_identity`.
    Other model names (e.g. ``"mattersim-5m"``) are only slash-safed.
    """
    if ML_model.startswith("eddp:"):
        return "eddp-" + _safe_ml_name(Path(ML_model.removeprefix("eddp:")).stem)
    return _safe_ml_name(ML_model)


FACTORY_PREFIX = "py:"


@functools.lru_cache(maxsize=None)
def _resolve_model(model: str):
    """``ML_model`` for a ``--model`` value: the model string itself, or a calculator factory.

    ``py:<module>:<function>`` or ``py:<file.py>:<function>`` imports ``<function>``,
    which must return an ASE calculator when called without arguments, as for
    ``ML_model`` in :func:`egqca.run_ml_relaxation`. Cached, so all clusters of one
    call get the same factory and egqca builds the calculator only once.
    """
    if not model.startswith(FACTORY_PREFIX):
        return model
    target, sep, name = model[len(FACTORY_PREFIX):].rpartition(":")
    if not sep or not target or not name:
        raise SystemExit(f"--model {model!r}: expected py:<module or file.py>:<function>.")
    if target.endswith(".py"):
        path = Path(target).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or not path.is_file():
            raise SystemExit(f"--model {model!r}: file {path} not found.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    factory = getattr(module, name, None)
    if not callable(factory):
        raise SystemExit(f"--model {model!r}: {name!r} in {target!r} is not a function.")
    return factory


@contextlib.contextmanager
def _atomic_write(path: str | Path, mode: str = "w"):
    """Open a temporary file next to ``path`` that replaces ``path`` only once the block is done.

    A job killed while writing leaves the old file (or none), never a partial one,
    so a partial file can never mark a cluster as done.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    try:
        with open(tmp, mode) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


MODEL_IDENTITY_NAME = "model_identity.json"
RUN_SETTINGS_NAME = "run_settings.json"


def _create_json_once(path: Path, data: dict) -> bool:
    """Write ``data`` to ``path`` unless it exists; True if this call created it.

    The file is written to a temporary name and hard-linked into place: atomic,
    so parallel per-cluster jobs never read a half-written file, and only the
    first one wins.
    """
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink()


def _model_identity(ML_model) -> dict | None:
    """Path and sha256 of a file-based model, or None for named models.

    Only eddp models (``"eddp:<path>"``) are files; the hash covers the ``.eddp``
    manifest and every ``.ddp`` ensemble member it lists, so retraining a model
    under the same file name is detected.
    """
    if not (isinstance(ML_model, str) and ML_model.startswith("eddp:")):
        return None
    manifest = Path(ML_model.removeprefix("eddp:")).expanduser().resolve()
    h = hashlib.sha256(manifest.read_bytes())
    for line in manifest.read_text().splitlines():
        if line.strip():
            h.update((manifest.parent / line.split()[0]).read_bytes())
    return {"model": ML_model, "path": str(manifest), "sha256": h.hexdigest()}


def _check_model_identity(run_folder: str | Path, identity: dict | None) -> None:
    """Record the model of a run folder, or refuse to resume it with another model.

    The first run writes ``<run_folder>/model_identity.json``; later runs (resume,
    parallel per-cluster jobs) must use a model with the same sha256.
    """
    if identity is None:
        return
    path = Path(run_folder) / MODEL_IDENTITY_NAME
    if _create_json_once(path, identity):
        return
    recorded = json.loads(path.read_text())
    if recorded.get("sha256") != identity["sha256"]:
        raise ValueError(
            f"{run_folder} was started with the model {recorded.get('path')} "
            f"(sha256 {str(recorded.get('sha256'))[:12]}...), but this run uses "
            f"{identity['path']} (sha256 {identity['sha256'][:12]}...). "
            "Use another out_folder_name / output_dir for a different model."
        )


def _run_settings(
    *,
    model: str,
    pressure_GPa: float | None,
    relaxation_constraint: bool | str | None,
    CellFilter_mask: list[bool] | None,
    bravais_supercell,
    fix_bravais_positions: bool,
    fmax: float,
    optimizer,
    optimizer_kwargs: dict | None,
    bravais_tol: float = DEFAULT_BRAVAIS_TOL,
    model_source: str | None = None,
) -> dict:
    """Settings that determine the relaxed structures of a run, in comparable form.

    Every cluster of a run folder must be relaxed with these. ``steps`` is not
    among them: it only limits the optimizer, so a rerun of the non-converged
    clusters may raise it. ``device`` and the output switches do not change the
    results either. ``bravais_tol`` is only recorded for Bravais constraints.
    ``model`` is the model name (``ML_model_name`` or the model string),
    ``model_source`` the model behind it (see :func:`egqca.ml._ml_model_source`),
    so that two models under one name are not mixed in a run folder.
    """
    relaxation_constraint = _normalize_relaxation_constraint(relaxation_constraint)
    settings = {
        "model": model,
        "model_source": model_source,
        "pressure_GPa": 0.0 if pressure_GPa is None else float(pressure_GPa),
        "relaxation_constraint": relaxation_constraint,
        "CellFilter_mask": [True] * 6 if CellFilter_mask is None else [bool(m) for m in CellFilter_mask],
        "bravais_supercell": [int(n) for n in bravais_supercell],
        "fix_bravais_positions": bool(fix_bravais_positions),
        "fmax": float(fmax),
        "optimizer": getattr(optimizer, "__name__", type(optimizer).__name__),
        # through JSON, so that a fresh dict compares equal to the recorded one
        "optimizer_kwargs": json.loads(json.dumps(optimizer_kwargs or {}, default=repr)),
    }
    if relaxation_constraint not in (None, "FixSymmetry"):
        settings["bravais_tol"] = float(bravais_tol)
    not_json = sorted(k for k, v in (optimizer_kwargs or {}).items() if not _is_json(v))
    if not_json:
        # recorded as repr strings: enough to compare runs, not to restore the values
        settings["optimizer_kwargs_not_json"] = not_json
        warnings.warn(
            f"optimizer_kwargs {not_json} are not plain JSON values and are recorded only as text; "
            "a rerun of this run folder needs them passed again as optimizer_kwargs.",
            UserWarning,
        )
    return settings


def _is_json(value) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _phonon_settings(N_phonon_supercell) -> dict:
    """Phonon settings every cluster of a run folder must share."""
    supercell = [N_phonon_supercell] * 3 if isinstance(N_phonon_supercell, int) else N_phonon_supercell
    return {"phonon_supercell": [int(n) for n in supercell]}


def _run_folder_has_results(run_folder: Path) -> bool:
    if any(
        (run_folder / sub).is_dir() and any((run_folder / sub).iterdir())
        for sub in ("per_cluster", "cif_relaxed", "phonons")
    ):
        return True
    # rows in clusters_<model>.dat of a sequential run
    return any(
        line.strip() and not line.lstrip().startswith("#")
        for dat in run_folder.glob("clusters_*.dat")
        for line in dat.read_text().splitlines()
    )


# Settings recorded only by later versions; a run folder from before cannot be checked for them
_SETTINGS_ADDED_LATER = ("model_source", "bravais_tol")


def _check_settings_record(path: Path, settings: dict, run_folder: Path, what: str) -> None:
    """Record ``settings`` in ``path``, or refuse to continue with different ones.

    A key of :data:`_SETTINGS_ADDED_LATER` that an older record lacks only gives a
    warning.
    """
    if _create_json_once(path, settings):
        return
    recorded = json.loads(path.read_text())
    if "relaxation_constraint" in recorded:
        # records written before the normalisation may hold e.g. "fcc" for "FCC"
        try:
            recorded["relaxation_constraint"] = _normalize_relaxation_constraint(recorded["relaxation_constraint"])
        except ValueError:
            pass    # left as recorded; the comparison below reports it
    unchecked = [key for key in _SETTINGS_ADDED_LATER if key in settings and key not in recorded]
    for key in unchecked:
        warnings.warn(
            f"{path} was written before {key} was recorded, so it cannot be checked. Make sure "
            f"this run uses the same {key} as the existing results ({key} = {settings[key]!r}).",
            UserWarning,
        )
    differences = [
        f"  {key}: recorded {recorded.get(key)!r}, this run {settings.get(key)!r}"
        for key in sorted(set(recorded) | set(settings))
        if key not in unchecked and recorded.get(key) != settings.get(key)
    ]
    if differences:
        raise ValueError(
            f"The run folder {run_folder} was started with different {what}:\n"
            + "\n".join(differences)
            + f"\nContinuing or redoing calculations in a run folder must use the settings "
            f"it was started with (recorded in {path}).\n"
            "- To continue or rerun this run, use the recorded settings.\n"
            "- To start a new run with these settings, choose another out_folder_name / output_dir."
        )


def _check_run_settings(
    run_folder: str | Path,
    settings: dict,
    phonon_settings: dict | None = None,
) -> None:
    """Record the settings of a run folder, or refuse to continue it with others.

    The first run writes ``<run_folder>/run_settings.json`` (see
    :func:`_run_settings`); every later run in the folder, a continued run, a
    parallel per-cluster job, or a rerun of the non-converged clusters, must use
    the same settings. The phonon settings are recorded separately in
    ``<run_folder>/phonons/phonon_settings.json`` when phonons are first
    requested, so that phonons can still be added to a run that relaxed without.
    """
    run_folder = Path(run_folder)
    path = run_folder / RUN_SETTINGS_NAME
    if not path.exists() and _run_folder_has_results(run_folder):
        warnings.warn(
            f"{run_folder} contains results but no {RUN_SETTINGS_NAME} (made before settings "
            "were recorded). The settings of this run are recorded now; make sure they are "
            "the ones the existing results were made with.",
            UserWarning,
        )
    _check_settings_record(path, settings, run_folder, "relaxation settings")
    if phonon_settings is not None:
        phonon_dir = run_folder / "phonons"
        phonon_dir.mkdir(parents=True, exist_ok=True)
        _check_settings_record(phonon_dir / "phonon_settings.json", phonon_settings, run_folder, "phonon settings")


def _load_run_settings(run_folder: str | Path) -> tuple[dict, dict | None]:
    """Recorded ``(settings, phonon_settings)`` of a run folder; phonon settings None without phonons."""
    run_folder = Path(run_folder)
    path = run_folder / RUN_SETTINGS_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{run_folder} has no {RUN_SETTINGS_NAME}, so the settings of this run are unknown."
        )
    phonon_path = run_folder / "phonons" / "phonon_settings.json"
    phonon_settings = json.loads(phonon_path.read_text()) if phonon_path.exists() else None
    return json.loads(path.read_text()), phonon_settings


def _original_cluster_cif(cluster_dir: str | Path) -> Path | None:
    cluster_dir = Path(cluster_dir)
    preferred_cif = cluster_dir / f"{cluster_dir.name}.cif"
    if preferred_cif.exists():
        return preferred_cif
    cif_files = sorted(
        cif_file
        for cif_file in cluster_dir.glob("S*.cif")
        if not cif_file.stem.startswith(f"{cluster_dir.name}_")
    )
    return cif_files[0] if cif_files else None


def _extract_snumber_from_name(name: str) -> int:
    match = re.search(r"S(\d+)", name)
    if match is None:
        raise ValueError(f"Could not extract S number from {name}.")
    return int(match.group(1))


def _cluster_info_from_id(
    cluster_id: str, permutable_sites: int
) -> tuple[int | None, float | None, int | None]:
    match = re.search(r"S(\d+)_", cluster_id)
    if match:
        n_j = int(match.group(1))
        x = n_j / permutable_sites
    else:
        n_j = None
        x = None
    match = re.search(r"_w(\d+)", cluster_id)
    g_j = int(match.group(1)) if match else None
    return n_j, x, g_j


def _parse_relax_cell_mask(value: str) -> list[bool]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "relax cell mask must contain six comma-separated bools"
        )
    bools = []
    for part in parts:
        lowered = part.lower()
        if lowered in {"true", "t", "1", "yes", "y"}:
            bools.append(True)
        elif lowered in {"false", "f", "0", "no", "n"}:
            bools.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid bool value: {part!r}")
    return bools


_CRYSTAL_SYSTEMS: frozenset[str] = frozenset({
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "rhombohedral",
    "hexagonal",
    "cubic",
})

_BRAVAIS_LATTICE_SYSTEMS: dict[str, set[str]] = {
    "CUB": {"cubic"},
    "FCC": {"cubic"},
    "BCC": {"cubic"},
    "TET": {"tetragonal"},
    "BCT": {"tetragonal"},
    "ORC": {"orthorhombic"},
    "ORCF": {"orthorhombic"},
    "ORCI": {"orthorhombic"},
    "ORCC": {"orthorhombic"},
    "HEX": {"hexagonal", "trigonal"},
    "RHL": {"rhombohedral", "trigonal"},
    "MCL": {"monoclinic"},
    "MCLC": {"monoclinic"},
    "TRI": {"triclinic"},
}

_BRAVAIS_FULL_NAMES: dict[str, str] = {
    "CUB":  "Cubic - Primitive",
    "FCC":  "Cubic - Face-Centred",
    "BCC":  "Cubic - Body-Centred",
    "TET":  "Tetragonal - Primitive",
    "BCT":  "Tetragonal - Body-Centred",
    "ORC":  "Orthorhombic - Primitive",
    "ORCF": "Orthorhombic - Face-Centred",
    "ORCI": "Orthorhombic - Body-Centred",
    "ORCC": "Orthorhombic - Base-Centred",
    "HEX":  "Hexagonal",
    "RHL":  "Rhombohedral",
    "MCL":  "Monoclinic - Primitive",
    "MCLC": "Monoclinic - Base-Centred",
    "TRI":  "Triclinic",
}


def _bravais_full_name(key: str) -> str:
    """Return the full human-readable Bravais lattice name, e.g. 'Cubic - Face-Centred'."""
    return _BRAVAIS_FULL_NAMES.get(key, key)


# Bravais types sharing one holohedry; a start cell of any of them is accepted for a target.
_HOLOHEDRY_FAMILIES: dict[str, tuple[str, ...]] = {
    "cubic": ("CUB", "FCC", "BCC"),
    "tetragonal": ("TET", "BCT"),
    "orthorhombic": ("ORC", "ORCF", "ORCI", "ORCC"),
    "hexagonal": ("HEX",),
    "trigonal": ("HEX",),
    "rhombohedral": ("RHL",),
    "monoclinic": ("MCL", "MCLC"),
    "triclinic": ("TRI",),
}

# Space group of a lattice with one atom per cell (its holohedry) -> Bravais key
_HOLOHEDRY_SPACEGROUPS: dict[int, str] = {
    2: "TRI", 10: "MCL", 12: "MCLC",
    47: "ORC", 65: "ORCC", 69: "ORCF", 71: "ORCI",
    123: "TET", 139: "BCT", 166: "RHL", 191: "HEX",
    221: "CUB", 225: "FCC", 229: "BCC",
}


def _target_family(target: str) -> tuple[str, ...]:
    """Bravais keys a start cell may have for a Bravais key or crystal system target."""
    key = str(target).upper()
    if key in _BRAVAIS_LATTICE_SYSTEMS:
        return next(f for f in _HOLOHEDRY_FAMILIES.values() if key in f)
    return _HOLOHEDRY_FAMILIES[str(target).lower()]


def _detect_bravais(
    atoms,
    supercell: list[int] | tuple[int, int, int] = (1, 1, 1),
    bravais_tol: float = DEFAULT_BRAVAIS_TOL,
) -> tuple[str, np.ndarray]:
    """Bravais key and lattice point group of the (primitive) cell of ``atoms``.

    spglib analyses the bare lattice (one atom per primitive cell) with
    ``symprec = bravais_tol * V**(1/3)``, i.e. a relative tolerance. The rotations
    are integer matrices in the fractional basis of the primitive cell, so the
    label and the constraint built from them always agree.
    """
    try:
        import spglib
    except ImportError as exc:
        raise ImportError(
            "spglib is required for Bravais-lattice-preserving relaxation. "
            "Install it with: pip install spglib"
        ) from exc

    n = np.asarray(supercell, dtype=float)
    prim_cell = np.asarray(atoms.cell.array) / n[:, None]
    symprec = bravais_tol * np.cbrt(abs(np.linalg.det(prim_cell)))
    dataset = spglib.get_symmetry_dataset((prim_cell, [[0.0, 0.0, 0.0]], [1]), symprec=symprec)
    if dataset is None:
        raise RuntimeError(
            "spglib could not determine the lattice symmetry for the Bravais-lattice "
            f"constraint (bravais_tol={bravais_tol:g})."
        )
    return _HOLOHEDRY_SPACEGROUPS[int(dataset.number)], np.asarray(dataset.rotations, dtype=int)


class FixBravaisCell:
    """Constrain the cell metric and (optionally) atomic positions to the target Bravais type.

    Cell / stress constraint
    ------------------------
    The lattice point group of the start cell is detected with
    :func:`_detect_bravais` (relative tolerance ``bravais_tol``), as integer
    rotations in the fractional basis of the primitive cell. A start cell
    within ``bravais_tol`` of a higher-symmetry lattice gets the full point
    group of that lattice, so it is projected onto the exact lattice. For an
    explicit target (Bravais key or crystal system) the detected type must
    belong to the target's holohedry family, otherwise a ValueError is raised.

    For a supercell input with sizes [n1, n2, n3] the metric is transformed to
    the primitive space, projected, then reconstructed:

        G_prim     = P⁻¹ G_super P⁻ᵀ
        G_prim_sym = mean_R( Rᵀ G_prim R )   # R expressed in C_prim fractional coords
        G_super    = P G_prim_sym Pᵀ

    For a primitive-cell input (supercell=[1,1,1], P=I) this reduces to the
    standard single-cell formula.

    Frozen positions (fix_positions=True; default False)
    ----------------------------------------------------
    After every optimizer step each atom is snapped back to its starting
    fractional coordinate (stored at construction time), expressed in the current
    Bravais-projected cell.  Atomic forces are zeroed so that convergence is
    determined entirely by the cell stress; the ML-calculator stress already
    encodes the full virial contribution from all atomic forces.

    This freezes the internal coordinates completely; it does not project them
    onto a symmetry-allowed subspace.  It is exact only if every atom sits at a
    position without positional freedom.  In a decorated (substitutional)
    cluster the site symmetry is lowered, so atoms can generally relax locally,
    and fix_positions=True neglects that relaxation.  With fix_positions=False
    (default) only the cell metric / stress is constrained.
    """

    def __init__(
        self,
        atoms,
        target: bool | str = True,
        supercell: list[int] | tuple[int, int, int] = (1, 1, 1),
        fix_positions: bool = False,
        bravais_tol: float = DEFAULT_BRAVAIS_TOL,
    ):
        if not bravais_tol > 0:
            raise ValueError(f"bravais_tol must be positive, got {bravais_tol!r}")
        self.atoms = atoms.copy()
        self.target = target
        self.bravais_tol = float(bravais_tol)
        self.fix_positions = fix_positions
        self.supercell = list(supercell)
        n = np.asarray(supercell, dtype=float)
        self._P = np.diag(np.asarray(supercell, dtype=int))
        self._P_inv = np.diag(1.0 / n)

        self.bravais_lattice, rotations = _detect_bravais(atoms, supercell, self.bravais_tol)
        if target is not True and self.bravais_lattice not in _target_family(target):
            raise ValueError(_start_lattice_error(self.bravais_lattice, target, supercell, self.bravais_tol))
        self._rotations_prim = list(rotations)

        if fix_positions:
            self._setup_position_constraint(atoms)

    def _setup_position_constraint(self, atoms) -> None:
        """Store the ideal (initial) fractional positions for adjust_positions."""
        # Fractional coordinates are invariant under scale_atoms=True, so this
        # survive the cell changes during the relaxation.
        self._f_ideal = atoms.get_scaled_positions().copy()

    def adjust_cell(self, atoms, cell) -> None:
        # 1. Lift the supercell metric to the primitive-cell metric.
        # 2. Project onto the Bravais-invariant subspace using the primitive
        #    rotation group (rotations are in primitive fractional coords).
        # 3. Reconstruct the supercell metric.
        # For a primitive input (P = I) this reduces to the original formula.
        # The metric symmetrization is orientation-independent, so the cached
        # primitive-basis holohedry rotations are used directly.
        rotations = self._rotations_prim

        # G = metric tensor, transform from N1xN2xN3 supercell to 1x1x1 cell
        G_super = cell @ cell.T
        G_prim = self._P_inv @ G_super @ self._P_inv.T
        G_prim_sym = sum(R.T @ G_prim @ R for R in rotations) / len(rotations)
        cell[:] = np.linalg.cholesky(self._P.astype(float) @ G_prim_sym @ self._P.astype(float).T)
        

    def adjust_stress(self, atoms, stress) -> None:
        # Stress from  is a Cartesian rank-2 tensor, so the holohedry must be applied
        # as Cartesian rotations g = H R H^-1, with R the primitive fractional
        # rotation and H the primitive cell (columns = lattice vectors). H is
        # taken from the *current* cell so the rotations share the frame of the
        # stress tensor (adjust_cell re-orients the cell via Cholesky each step).
        # Primitive vs. supercell H give identical g (the diagonal supercell
        # factor cancels), so the primitive cell is used here.
        n = np.asarray(self.supercell, dtype=float)
        H = (atoms.cell.array / n[:, None]).T
        H_inv = np.linalg.inv(H)

        raw_stress = voigt_6_to_full_3x3_stress(stress)
        M_sym = np.zeros((3, 3))
        for r in self._rotations_prim:
            g = H @ r @ H_inv          # fractional -> current-frame Cartesian
            M_sym += g @ raw_stress @ g.T
        M_sym /= len(self._rotations_prim)
        stress[:] = full_3x3_to_voigt_6_stress(M_sym)


    def adjust_positions(self, atoms, new) -> None:
        if not self.fix_positions:
            return
        # Freeze all atoms at their starting fractional coordinates, expressed
        # in the current (Bravais-projected) Cartesian cell.  No internal
        # relaxation at all, see the class docstring.
        new[:] = self._f_ideal @ atoms.cell.array

    def adjust_forces(self, atoms, forces) -> None:
        if not self.fix_positions:
            return
        # The positions are frozen, so the forces are zeroed and convergence is
        # determined solely by the cell stress.
        # The ML-calculator stress (used to drive the cell DOFs) already encodes
        # the full virial, so cell relaxation is unaffected.
        forces[:] = 0.0

    def todict(self) -> dict:
        return {
            "name": "FixBravaisCell",
            "kwargs": {
                "atoms": self.atoms,
                "target": self.target,
                "supercell": self.supercell,
                "fix_positions": self.fix_positions,
                "bravais_tol": self.bravais_tol,
            },
        }


def _bravais_cell_constraint(
    fix_bravais: bool | str,
    atoms: Atoms,
    supercell: list[int] | tuple[int, int, int] = (1, 1, 1),
    fix_positions: bool = False,
    bravais_tol: float = DEFAULT_BRAVAIS_TOL,
) -> FixBravaisCell:
    """Return a Bravais-lattice constraint for the requested setting.

    fix_bravais=True          — Bravais type detected from atoms.cell
    fix_bravais="FCC"         — Bravais key; the start cell must be of its holohedry family
    fix_bravais="cubic"       — crystal system; the start cell must be of that family
    supercell                 — diagonal supercell sizes [n1, n2, n3]; lattice vectors
                                are divided by these before the symmetry is detected
    fix_positions             — also freeze the atomic positions at their starting
                                fractional coordinates (no internal relaxation); default
                                False constrains only the cell metric / stress
    bravais_tol               — relative tolerance of the lattice detection; a start cell
                                within it of a higher-symmetry lattice is projected onto it

    For a Bravais key or crystal system, the starting cell must already be of
    that type (within bravais_tol); otherwise a ValueError is raised.
    """
    if fix_bravais is not True and (
            str(fix_bravais).upper() not in _BRAVAIS_LATTICE_SYSTEMS
            and str(fix_bravais).lower() not in _CRYSTAL_SYSTEMS):
        raise ValueError(
            f"Unknown Bravais key or crystal system: {fix_bravais!r}. "
            f"Bravais keys: {sorted(_BRAVAIS_LATTICE_SYSTEMS)}. "
            f"Crystal systems: {sorted(_CRYSTAL_SYSTEMS)}"
        )
    return FixBravaisCell(atoms, target=fix_bravais, supercell=supercell,
                          fix_positions=fix_positions, bravais_tol=bravais_tol)


def _start_lattice_error(lattice: str, target: str, supercell=(1, 1, 1),
                         bravais_tol: float = DEFAULT_BRAVAIS_TOL) -> str:
    sc = "x".join(str(int(n)) for n in supercell)
    return (
        f"The requested Bravais constraint {target!r} does not match the starting cell, which is "
        f"{_bravais_full_name(lattice)} ({lattice}) for bravais_supercell={sc} and "
        f"bravais_tol={bravais_tol:g}. "
        "The constraint only keeps the symmetry of the starting cell, it cannot impose "
        "a new one. Start from a structure (S0) with the target symmetry: the supercell "
        "program builds the clusters from the symmetry of S0, so a wrong S0 also gives "
        "the wrong set of clusters."
    )


def _bravais_lattice_name(
    atoms, eps: float = 0.02, supercell: tuple | list = (1, 1, 1)
) -> str:
    n = np.asarray(supercell, dtype=float)
    if np.all(n == 1.0):
        return atoms.cell.get_bravais_lattice(eps=eps, pbc=atoms.pbc).name
    from ase.cell import Cell
    prim_cell = Cell(atoms.cell.array / n[:, None])
    return prim_cell.get_bravais_lattice(eps=eps, pbc=atoms.pbc).name


def _parse_bravais_supercell(value: str) -> list[int]:
    """Parse '2x2x1' into [2, 2, 1]."""
    parts = value.strip().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"bravais-supercell must be three integers separated by 'x', e.g. '2x2x1', got {value!r}"
        )
    try:
        result = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bravais-supercell values must be integers, got {value!r}"
        )
    if any(n < 1 for n in result):
        raise argparse.ArgumentTypeError(
            f"bravais-supercell values must be >= 1, got {result}"
        )
    return result


def _parse_phonon_supercell(value: str) -> int | list[int]:
    """Parse a phonon supercell command-line value into an int or [N1, N2, N3].

    Accepts either a single integer ``'N'`` (returned as ``int`` -> ``N x N x N``)
    or three comma-separated integers ``'N1,N2,N3'`` (returned as ``list[int]``),
    e.g. ``'2'`` or ``'2,2,1'``.
    """
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    if len(parts) not in (1, 3):
        raise argparse.ArgumentTypeError(
            "phonon supercell must be a single integer or three integers separated "
            f"by ',', e.g. '2' or '2,2,1', got {value!r}"
        )
    try:
        result = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"phonon supercell values must be integers, got {value!r}"
        )
    if any(n < 1 for n in result):
        raise argparse.ArgumentTypeError(
            f"phonon supercell values must be >= 1, got {result}"
        )
    return result[0] if len(result) == 1 else result


def _parse_optimizer_kwargs(value: str) -> dict:
    """argparse type function for --optimizer-kwargs: a JSON object, e.g. '{"maxstep": 0.1}'."""
    try:
        kwargs = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--optimizer-kwargs is not valid JSON: {exc}") from None
    if not isinstance(kwargs, dict):
        raise argparse.ArgumentTypeError(
            f"--optimizer-kwargs must be a JSON object such as '{{\"maxstep\": 0.1}}', got {value!r}"
        )
    return kwargs


def _parse_optimizer(value: str):
    """Resolve an ASE optimizer class from its name, e.g. 'FIRE', 'BFGS', 'LBFGS'.

    Returns the optimizer class (suitable to pass to ``geometry_relaxation``).
    The lookup is case-insensitive.
    """
    import ase.optimize as ase_opt

    name = value.strip()
    optimizer = getattr(ase_opt, name, None)
    if not isinstance(optimizer, type):
        # Exact attribute is missing or not a class (e.g. the lowercase
        # submodule ``ase.optimize.bfgs``); search classes case-insensitively.
        optimizer = None
        for attr in dir(ase_opt):
            candidate = getattr(ase_opt, attr)
            if isinstance(candidate, type) and attr.lower() == name.lower():
                optimizer = candidate
                break
    if not isinstance(optimizer, type):
        available = sorted(
            a for a in dir(ase_opt)
            if a[:1].isupper() and isinstance(getattr(ase_opt, a), type)
        )
        raise argparse.ArgumentTypeError(
            f"unknown ASE optimizer {value!r}. Available: {', '.join(available)}"
        )
    return optimizer


def _normalize_relaxation_constraint(value: bool | str | None) -> bool | str | None:
    """Canonical form of a ``relaxation_constraint``, the same from Python and the CLI.

    Strings are case-insensitive: ``"none"`` -> None, ``"auto"`` -> True,
    ``"fixsymmetry"`` -> ``"FixSymmetry"``, Bravais keys upper-case (``"FCC"``),
    crystal systems lower-case (``"cubic"``). None and True are returned as they
    are; anything else raises a ValueError.
    """
    if value is None or value is True:
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower == "none":
            return None
        if lower == "auto":
            return True
        if lower == "fixsymmetry":
            return "FixSymmetry"
        if value.upper() in _BRAVAIS_LATTICE_SYSTEMS:
            return value.upper()
        if lower in _CRYSTAL_SYSTEMS:
            return lower
    raise ValueError(
        f"Unknown relaxation_constraint {value!r}. Use None, True (auto), 'FixSymmetry', "
        f"a Bravais key {sorted(_BRAVAIS_LATTICE_SYSTEMS)} or a crystal system "
        f"{sorted(_CRYSTAL_SYSTEMS)} (case-insensitive)."
    )


def _fix_bravais_arg(value: str) -> str:
    """argparse type function for --fix-bravais.

    Accepts either a Bravais lattice key (e.g. "FCC") or a crystal system
    name (e.g. "cubic").  Keys are returned upper-case; system names lower-case.
    """
    normalized = None
    try:
        normalized = _normalize_relaxation_constraint(value)
    except ValueError:
        pass
    if isinstance(normalized, str) and normalized != "FixSymmetry":
        return normalized
    raise argparse.ArgumentTypeError(
        f"Unknown Bravais key or crystal system: {value!r}. "
        f"Bravais keys: {sorted(_BRAVAIS_LATTICE_SYSTEMS)}. "
        f"Crystal systems: {sorted(_CRYSTAL_SYSTEMS)}"
    )


def _relaxation_constraint_arg(value: str) -> bool | str | None:
    """argparse type function for --relaxation-constraint.

    Accepts one of four forms:

    - ``"none"`` — no ASE constraint; cell DOF are controlled by
      ``--cell-filter-mask`` only.
    - ``"FixSymmetry"`` — preserve the space group via ASE FixSymmetry with a
      fully free cell.
    - ``"auto"`` — FixBravaisCell with the Bravais type auto-detected from
      the structure's own (pre-relaxation) cell, rather than specified
      explicitly.
    - A Bravais lattice key (e.g. ``"RHL"``) or crystal system name
      (e.g. ``"rhombohedral"``) — keep the cell metric consistent with the
      specified Bravais type via FixBravaisCell with a fully free cell.

    To restrict individual strain components use ``--cell-filter-mask`` instead.
    """
    try:
        return _normalize_relaxation_constraint(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None

def _bravais(atoms: Atoms, eps: float = 2e-4) -> str:
    return atoms.cell.get_bravais_lattice(eps=eps).name

def _print_cell(
    label: str,
    atoms: Atoms,
    supercell: tuple | list = (1, 1, 1),
) -> None:
    n = np.asarray(supercell, dtype=float)
    if np.all(n == 1.0):
        cell = atoms.cell
    else:
        from ase.cell import Cell
        cell = Cell(atoms.cell.array / n[:, None])
    L = cell.lengths()
    A = cell.angles()
    print(f"{label:10s} "
          f"a={L[0]:.4f}  b={L[1]:.4f}  c={L[2]:.4f}  "
          f"α={A[0]:.2f}  β={A[1]:.2f}  γ={A[2]:.2f}")
