import contextlib
import importlib
import json
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from functools import lru_cache
from math import gcd
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import re
import shutil
import tempfile
import os
import traceback
from typing import Any

from ._banner import entry_point

import numpy as np
from ase import Atoms, atoms, units
from ase import __version__ as ase_version
from ase.calculators.calculator import BaseCalculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.io import read, write

from ase.phonons import Phonons
from ase.spectrum.dosdata import RawDOSData

from .constants import Ry_to_mev
from .workflow import (
    get_directories,
    _write_ml_clusters_header,
    _write_ml_volume_header,
    _compound_from_cif,
    _compound_from_cif_pair,
    _header_permutable_sites,
    _resolve_permutable_sites,
    _resolve_cluster_setup,
    _infer_cluster_setup,
    _parse_clusters_template,
    _cluster_ids_from_template,
)
from ._ml_utils import (
    _Tee,
    _safe_ml_name,
    _ml_model_display_name,
    _original_cluster_cif,
    _cluster_info_from_id,
    _extract_snumber_from_name,
    DEFAULT_RELAXATION_CONSTRAINT,
    DEFAULT_BRAVAIS_TOL,
    _bravais_cell_constraint,
    _normalize_relaxation_constraint,
    _bravais_lattice_name,
    _bravais_full_name,
    _BRAVAIS_LATTICE_SYSTEMS,
    _print_cell,
    _runtime_line,
    _model_identity,
    _check_model_identity,
    _run_settings,
    _phonon_settings,
    _check_run_settings,
    _load_run_settings,
    _parse_optimizer,
    _atomic_write,
)

# Layout of a saved phonon force cache; see :func:`phonons`.
PHONON_FORCE_CACHE_NAME = "forces"
PHONON_EQUILIBRIUM_NAME = "equilibrium.xyz"
PHONON_INFO_NAME = "phonon_info.json"
# Arguments of Phonons.read(); stored with the cache so the force constants can
# be rebuilt exactly as they were.
PHONON_READ_DEFAULTS = {
    "method": "Frederiksen",
    "symmetrize": 3,
    "acoustic": True,
    "cutoff": None,
}


def _validate_cell_mask(mask: list[bool]) -> list[bool]:
    if not isinstance(mask, list):
        raise TypeError(
            "relaxation_constraint cell mask must be a list of six bools for the strain components [xx, yy, zz, yz, xz, xy]."
        )
    if len(mask) != 6:
        raise ValueError(
            "relaxation_constraint cell mask must be a list of six bools for the strain components [xx, yy, zz, yz, xz, xy]."
        )
    if not all(isinstance(value, bool) for value in mask):
        raise TypeError(
            "relaxation_constraint cell mask entries must be bool values for the "
            "strain components [xx, yy, zz, yz, xz, xy]."
        )
    return list(mask)


def _optimizer_name(optimizer: Any) -> str:
    return getattr(optimizer, "__name__", optimizer.__class__.__name__)


# ---------------------- ML calculators ---------------------- #
def _matter_sim_calculator() -> type:
    try:
        forcefield = importlib.import_module("mattersim.forcefield")
    except ImportError as exc:
        raise ImportError(
            "MatterSim is required for ML_model='mattersim-*'. Install it with "
            "`pip install egqca[mattersim]` in an environment that does not also "
            "install conflicting ML backends."
        ) from exc
    return forcefield.MatterSimCalculator


def _mace_calculator() -> Any:
    try:
        calculators = importlib.import_module("mace.calculators")
    except ImportError as exc:
        raise ImportError(
            "MACE is required for ML_model='mace*'. Install it with "
            "`pip install egqca[mace]` or `pip install mace-torch`."
        ) from exc
    return calculators


def _mace_mp_calculator(
    calculators: Any,
    *,
    model: str,
    device: str,
    head: str | None = None,
) -> Any:
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    calculator_kwargs = {
        "model": model,
        "dispersion": False,
        "default_dtype": "float64",
        "device": device,
    }
    if head is not None:
        calculator_kwargs["head"] = head
    try:
        calculator = calculators.mace_mp(**calculator_kwargs)
    finally:
        # float64 must not stay the default of the whole process (other models
        # loaded later in the same process would inherit it)
        torch.set_default_dtype(previous_dtype)

    # MACE builds its input tensors with the default dtype at calculation time, so
    # float64 is set around every calculation of this calculator only
    calculate = calculator.calculate

    def calculate_in_float64(*args, **kwargs):
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return calculate(*args, **kwargs)
        finally:
            torch.set_default_dtype(dtype)

    calculator.calculate = calculate_in_float64
    return calculator


# Maps user-facing ML_model strings to (mace_mp key, head).
# Keys in mace_mp_urls (from mace.calculators.foundations_models) are used verbatim.
# All entries use the mace_mp calculator; molecular models (mace_off, mace_anicc) are excluded.
_MACE_MODEL_MAP: dict[str, tuple[str, str | None]] = {
    # MACE-MP-0 — MIT license
    "mace-small":              ("small",                  None),
    "mace-medium":             ("medium",                 None),
    "mace-large":              ("large",                  None),
    # MACE-MP-0b — MIT license
    "mace-small-0b":           ("small-0b",               None),
    "mace-medium-0b":          ("medium-0b",              None),
    # MACE-MP-0b2 — MIT license
    "mace-small-0b2":          ("small-0b2",              None),
    "mace-medium-0b2":         ("medium-0b2",             None),
    "mace-large-0b2":          ("large-0b2",              None),
    # MACE-MP-0b3 — MIT license
    "mace-medium-0b3":         ("medium-0b3",             None),
    # MACE-MPA-0 — MIT license; default since mace v0.3.10
    "mace-medium-mpa-0":       ("medium-mpa-0",           None),
    # MACE-OMAT-0 — ASL license
    "mace-small-omat-0":       ("small-omat-0",           None),
    "mace-medium-omat-0":      ("medium-omat-0",          None),
    # MACE-MATPES-0 — ASL license
    "mace-matpes-pbe-0":       ("mace-matpes-pbe-0",      None),
    "mace-matpes-r2scan-0":    ("mace-matpes-r2scan-0",   None),
    # MACE-MH (multihead) — ASL license
    "mace-mh-0":               ("mh-0",                   "matpes_r2scan"),
    "mace-mh-1":               ("mh-1",                   "matpes_r2scan"),
    # Shorthands and backwards-compatible aliases
    "mace":                    ("medium-mpa-0",            None),
    "mace-mp":                 ("medium-mpa-0",            None),
    "mace-omat-0":             ("medium-omat-0",           None),
}


def _grace_foundation_models() -> Any:
    try:
        # tensorpotential must be imported before tensorflow, it configures Keras on import
        return importlib.import_module("tensorpotential.calculator.foundation_models")
    except ImportError as exc:
        raise ImportError(
            "GRACE (tensorpotential) is required for ML_model='GRACE-*'. Install it with "
            "`pip install egqca[grace]` or `pip install tensorpotential` in an environment "
            "that does not also install conflicting ML backends."
        ) from exc


def _resolve_grace_model(foundation_models: Any, ML_model: str) -> str:
    """Registry name of the GRACE foundation model ``ML_model``, matched case-insensitively.

    Backward-compatible aliases (e.g. ``"GRACE-2L-OAM-L"``) map to their full name.
    """
    names = {name.lower(): name for name in foundation_models.MODELS_NAME_LIST}
    names.update(
        (alias.lower(), name) for alias, name in foundation_models.MODELS_ALIASES_DICT.items()
    )
    try:
        return names[ML_model.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported GRACE model: {ML_model!r}. "
            f"Valid model strings: {sorted(foundation_models.MODELS_NAME_LIST)}. "
            "Run `grace_models list` to see the models of the installed tensorpotential."
        ) from None


def _select_tensorflow_device(device: str) -> None:
    """Make TensorFlow run on ``device`` ("cpu", "cuda" or "cuda:<index>").

    TensorFlow has no per-calculator device argument and places the GRACE model on
    the first GPU it finds, so ``device="cpu"`` is honoured by hiding the GPUs from
    TensorFlow. That is only possible before TensorFlow has initialised them in this
    process; afterwards the current placement is kept and a warning is issued.
    """
    tf = importlib.import_module("tensorflow")
    gpus = tf.config.list_physical_devices("GPU")
    kind, _, index = device.lower().partition(":")
    if kind == "cpu":
        target = []
    elif kind in ("cuda", "gpu"):
        if not gpus:
            warnings.warn(f"device={device!r} requested, but TensorFlow finds no GPU; GRACE runs on the CPU.")
            return
        target = [gpus[int(index)]] if index else gpus
    else:
        raise ValueError(f"Unsupported device for GRACE: {device!r}. Use 'cpu', 'cuda' or 'cuda:<index>'.")
    if tf.config.get_visible_devices("GPU") == target:
        return
    try:
        tf.config.set_visible_devices(target, "GPU")
    except RuntimeError:
        warnings.warn(
            f"TensorFlow already initialised its devices in this process, so device={device!r} "
            "cannot be applied to GRACE. Start a new process (or set CUDA_VISIBLE_DEVICES) "
            "to change the device."
        )


def _grace_calculator(ML_model: str, device: str) -> Any:
    foundation_models = _grace_foundation_models()
    model = _resolve_grace_model(foundation_models, ML_model)
    _select_tensorflow_device(device)
    # mode="uniform": dense engine, ~2x faster for relaxation trajectories and phonon
    # displacements, which all share one shape. The per-atom uncertainty (UQ) the
    # foundation models carry is not used by egqca and only costs time and memory.
    return foundation_models.grace_fm(
        model,
        mode="uniform",
        enable_uq_if_available=False,
    )


def _build_ml_calculator(ML_model: str, device: str = "cpu") -> Any:
    if ML_model == "mattersim-1m":
        MatterSimCalculator = _matter_sim_calculator()
        return MatterSimCalculator(device=device)
    if ML_model == "mattersim-5m":
        MatterSimCalculator = _matter_sim_calculator()
        return MatterSimCalculator(
            load_path="MatterSim-v1.0.0-5M.pth",
            device=device,
        )
    if ML_model.startswith("mace"):
        calculators = _mace_calculator()
        if ML_model in _MACE_MODEL_MAP:
            model_name, head = _MACE_MODEL_MAP[ML_model]
        elif ML_model.startswith("mace-mp-"):
            # pass-through for custom URLs or local checkpoint paths
            model_name = ML_model.removeprefix("mace-mp-")
            head = None
        else:
            raise ValueError(
                f"Unsupported MACE model: {ML_model!r}. "
                f"Valid model strings: {sorted(_MACE_MODEL_MAP)}. "
                "For custom checkpoints pass 'mace-mp-<url-or-path>'."
            )
        return _mace_mp_calculator(calculators, model=model_name, device=device, head=head)
    if ML_model.startswith("pet"):
        from upet.calculator import UPETCalculator
        return UPETCalculator(
            model=ML_model,
            dtype="float64",
            device=device,
        )
    if ML_model.upper().startswith("GRACE-"):
        return _grace_calculator(ML_model, device)
    if ML_model.startswith("eddp:"):
        from .eddp_calculator import EddpCalculator
        model_path = ML_model.removeprefix("eddp:")
        return EddpCalculator(model_path)
    raise ValueError(
        f"Unsupported ML model: {ML_model!r}.\n"
        "The model string must start with the name of a supported model family:\n"
        "  'mattersim' : 'mattersim-1m', 'mattersim-5m'\n"
        "  'mace'      : "
        + ", ".join(repr(name) for name in sorted(_MACE_MODEL_MAP))
        + ";\n"
        "                custom checkpoints via 'mace-mp-<url-or-path>'\n"
        "  'pet'       : UPET models, e.g. 'pet-mad-s', 'pet-omat-m', 'pet-oam-xl'\n"
        "                (the string is forwarded to UPETCalculator)\n"
        "  'GRACE'     : GRACE foundation models, e.g. 'GRACE-3L-OMAT-large-ft-AM'\n"
        "                (see `grace_models list`)\n"
        "  'eddp'      : 'eddp:<path-to-model>'\n"
        "See docs/ml-workflow.md ('Available Models') for the full list."
    )


#: What ``ML_model`` accepts: a model string, an ASE calculator, or a function
#: (or calculator class) that returns an ASE calculator when called without arguments.
MLModel = str | BaseCalculator | Callable[[], BaseCalculator]


def _is_calculator_instance(ML_model: Any) -> bool:
    # duck-typed, so calculators that do not derive from ASE's base classes work too
    return not isinstance(ML_model, type) and hasattr(ML_model, "get_potential_energy")


@lru_cache(maxsize=None)
def _cached_ml_calculator(ML_model: str, device: str) -> Any:
    return _build_ml_calculator(ML_model, device)


@lru_cache(maxsize=None)
def _cached_factory_calculator(factory: Callable[[], BaseCalculator]) -> Any:
    return factory()


def _ml_calculator(ML_model: MLModel, device: str = "cpu", reuse: bool = True) -> Any:
    """ASE calculator for ``ML_model``, loaded once per (model, device).

    The checkpoint is kept in memory and reused, so a cluster does not load the
    model twice, once for the relaxation and once for the phonons. The ML
    calculators only hold the model plus stateless settings, the per-structure
    results are handled by the ASE base class, so one instance can serve several
    ``Atoms`` objects. Pass ``reuse=False`` to force a freshly built calculator.

    A user-supplied calculator is returned as is, and a calculator factory is
    called without arguments, once per factory unless ``reuse=False``. ``device``
    only applies to the built-in model strings.
    """
    if not isinstance(ML_model, str):
        if _is_calculator_instance(ML_model):
            return ML_model
        if not callable(ML_model):
            raise TypeError(
                "ML_model must be a model string, an ASE calculator, or a function "
                f"returning an ASE calculator, got {type(ML_model).__name__}."
            )
        return _cached_factory_calculator(ML_model) if reuse else ML_model()
    if not reuse:
        return _build_ml_calculator(ML_model, device)
    # always called positionally: lru_cache keys on the arguments as passed
    return _cached_ml_calculator(ML_model, device)


def clear_ml_calculator_cache() -> None:
    """Drop the cached ML calculators, e.g. to free GPU memory."""
    _cached_ml_calculator.cache_clear()
    _cached_factory_calculator.cache_clear()


def _ml_model_name(ML_model: MLModel, ML_model_name: str | None = None) -> str:
    """Name of the model in output file/folder names, which also identify a run on resume.

    ``ML_model_name`` if given, else the model string. A user-supplied calculator
    or factory has no reliable name of its own (two different checkpoints share
    one calculator class), so for those ``ML_model_name`` is required.
    """
    if ML_model_name is not None:
        if not ML_model_name.strip():
            raise ValueError("ML_model_name must not be empty.")
        return ML_model_name
    if isinstance(ML_model, str):
        return ML_model
    raise ValueError(
        "ML_model_name is required when ML_model is a calculator or a calculator "
        "factory: it names the output files and identifies the run when it is resumed."
    )


def _ml_model_description(ML_model: MLModel, ML_model_name: str | None = None) -> str:
    """Human-readable model description for headers and logs."""
    if isinstance(ML_model, str):
        return ML_model
    if _is_calculator_instance(ML_model):
        source = type(ML_model).__name__
    else:
        source = f"factory {getattr(ML_model, '__qualname__', type(ML_model).__name__)}"
    return f"{ML_model_name or '<unnamed>'} (user-supplied calculator: {source})"


def _ml_model_source(ML_model: MLModel) -> str:
    """The model behind a run, recorded in run_settings.json next to its name.

    The model string itself, or the class / factory name of a user-supplied
    calculator. Checkpoints of one calculator class cannot be told apart.
    """
    if isinstance(ML_model, str):
        return ML_model
    if _is_calculator_instance(ML_model):
        return f"calculator {type(ML_model).__name__}"
    return f"factory {getattr(ML_model, '__qualname__', type(ML_model).__name__)}"


# ---------------------- ML relaxations and phonons ---------------------- #
def phonons(
    atoms: Atoms,
    ML_model: MLModel = "mattersim-5m",
    N: int | list[int] = 2,
    device: str = "cpu",
    delta: float = 0.01,
    kpts: tuple[int, int, int] = (20, 20, 20),
    npts: int = 500,
    width: float = 1e-4,
    cache_dir: Path | str | None = None,
    ML_model_name: str | None = None,
) -> Any:
    ''' Run a phonon calculation with ASE's finite-displacement ``Phonons`` class,
        using the given ML model as calculator.

        The DOS is sampled on the ``kpts`` q-point grid with ``npts`` points and
        ``width`` smearing, and is written by :func:`run_ml_relaxation` as
        ``<run_folder>/phonons/<cluster_id>.dos`` with the mode energy hbar*omega
        in eV in the first column. It integrates to 3 modes per atom, including
        any imaginary (negative-energy) modes of a dynamically unstable cluster;
        those are reported by the stability summary at the end of the run and are
        excluded again when :func:`egqca.io.read_phonons` reads the file back.

        param atoms: ASE Atoms object with the relaxed geometry
        param ML_model: ML model to use for the phonon calculation: a model string,
            an ASE calculator, or a function returning one (see
            :func:`run_ml_relaxation`)
        param N: supercell size for the phonon calculation. Either a single int
            (interpreted as an N x N x N supercell) or a list of three ints
            [N1, N2, N3] for an N1 x N2 x N3 supercell.
        param device: device to use for the calculation
        param delta: finite-displacement amplitude in Angstrom
        param kpts: Monkhorst-Pack grid the phonon frequencies are sampled on
        param npts: number of energy points of the sampled DOS
        param width: Gaussian smearing width in eV of the sampled DOS
        param cache_dir: folder to keep the finite-displacement forces and their
            metadata in, so that the DOS can be re-sampled later by
            :func:`phonon_dos_from_cache` without repeating the force
            evaluations. An existing cache of the same structure, supercell,
            displacement and model is reused: ASE skips the displacements it
            already finds there, which also makes an interrupted run
            restartable. A rigidly rotated copy of the structure, e.g. the
            relaxed structure read back from its CIF, counts as the same; the
            calculation then continues with the cached structure, in the frame
            the cached forces belong to. A cache of anything else is discarded.
            ``None`` (default) uses a temporary folder that is deleted again.
        param ML_model_name: name of a user-supplied calculator, recorded in the
            phonon cache metadata
        return: ASE DOS object with the phonon density of states
    '''
    supercell = _phonon_supercell(N)
    # Suppress warnings about matrix logarithm accuracy issued by scipy during geometry
    # optimization to avoid cluttering the output
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="logm result may be inaccurate, approximate err",
    )

    calc = _ml_calculator(ML_model, device=device)
    atoms.calc = calc

    # Remove constraints before phonon calculation (incompatible with supercell creation)
    atoms.set_constraint()

    with contextlib.ExitStack() as stack:
        if cache_dir is None:
            # Temporary directory, discarded together with the forces
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
            force_dir = os.path.join(tmpdir, 'phonons')
        else:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            force_dir = str(cache_dir / PHONON_FORCE_CACHE_NAME)
            model_description = _ml_model_description(ML_model, ML_model_name)
            # ASE reuses every force it finds in the cache, so forces of another
            # structure (e.g. before a re-relaxation), supercell or model must go
            if Path(force_dir).exists():
                if _phonon_cache_matches(cache_dir, atoms, supercell, delta, model_description):
                    # the cached forces are Cartesian in the frame of the cached
                    # structure, which may be rotated against atoms: continue with it
                    atoms = read(cache_dir / PHONON_EQUILIBRIUM_NAME, format="extxyz")
                    atoms.calc = calc
                else:
                    print(f"Discarding the phonon force cache {force_dir}: it belongs to another "
                          "structure, supercell, displacement or model.")
                    shutil.rmtree(force_dir)
            # written before the forces, so that an interrupted run can be checked as well
            _write_phonon_cache_metadata(
                cache_dir, atoms, supercell, delta, model_description, kpts, npts, width,
            )

        # Phonon calculator
        ph = Phonons(atoms, calc, supercell=supercell, delta=delta, name=force_dir)
        # a job killed during a force evaluation leaves an empty entry, which
        # run() would take as done
        ph.cache.strip_empties()
        ph.run()

        # Read forces and assemble the dynamical matrix
        ph.read(**PHONON_READ_DEFAULTS)
        t_dos0 = time.monotonic()
        dos = _sampled_phonon_dos(ph, len(atoms), kpts, npts, width)
        print(f"DOS runtime: {time.monotonic() - t_dos0:.2f} s")
        if cache_dir is None:
            ph.clean()

    return dos


def _sampled_phonon_dos(ph: Phonons, n_atoms: int, kpts, npts: int, width: float) -> Any:
    """Total phonon DOS of ``ph``, normalized to 3 modes per atom, on an energy grid.

    ASE <= 3.24 gives every mode at every q-point a weight of 1, so its DOS
    integrates to 3 * n_atoms * n_qpoints; newer versions divide by the number
    of q-points. Rescaling the raw weights makes the result independent of the
    ASE version.
    """
    raw = ph.get_dos(kpts=kpts)
    weights = raw.get_weights()
    raw = RawDOSData(raw.get_energies(), weights * (3 * n_atoms / weights.sum()), info=raw.info)
    return raw.sample_grid(npts=npts, width=width)


def _phonon_supercell(N: int | list[int]) -> tuple[int, int, int]:
    """Normalize a phonon supercell specification to a tuple of three ints."""
    if isinstance(N, int):
        return (N, N, N)
    supercell = tuple(int(n) for n in N)
    if len(supercell) != 3:
        raise ValueError(
            f"phonon supercell must be a single int or a list of three ints, got {N!r}"
        )
    return supercell


def _phonon_cache_matches(
    cache_dir: Path,
    atoms: Atoms,
    supercell: tuple[int, int, int],
    delta: float,
    ML_model: str,
    tol: float = 1e-4,
) -> bool:
    """Whether the forces in ``cache_dir`` were computed for this structure and setup.

    Cell and positions are compared within ``tol`` Angstrom, independent of the
    orientation: a relaxed structure read back from its CIF on a phonons-only
    resume is in ASE's standard orientation, while the relaxed cell may be rotated.
    Mirror images do not match. A cache without metadata cannot be checked and does
    not match.
    """
    info_path = cache_dir / PHONON_INFO_NAME
    equilibrium_path = cache_dir / PHONON_EQUILIBRIUM_NAME
    if not (info_path.exists() and equilibrium_path.exists()):
        return False
    try:
        info = json.loads(info_path.read_text())
        cached = read(equilibrium_path, format="extxyz")
    except Exception:
        # e.g. truncated by a job killed before the metadata was written atomically
        return False
    if info.get("supercell") != list(supercell) or info.get("delta") != delta or info.get("ML_model") != ML_model:
        return False
    if len(cached) != len(atoms) or not np.array_equal(cached.numbers, atoms.numbers):
        return False
    # standard form: lower-triangular cell with the rigid rotation removed
    cached_cell, _ = cached.cell.standard_form()
    cell, _ = atoms.cell.standard_form()
    if not np.allclose(cached_cell, cell, rtol=0.0, atol=tol):
        return False
    # minimum-image displacement, in case one of the structures was wrapped
    d_scaled = atoms.get_scaled_positions(wrap=False) - cached.get_scaled_positions(wrap=False)
    d_scaled -= np.round(d_scaled)
    return bool(np.max(np.linalg.norm(d_scaled @ atoms.cell.array, axis=1), initial=0.0) <= tol)


def _write_phonon_cache_metadata(
    cache_dir: Path,
    atoms: Atoms,
    supercell: tuple[int, int, int],
    delta: float,
    ML_model: str,
    kpts: tuple[int, int, int],
    npts: int,
    width: float,
) -> None:
    """Record everything besides the forces that the DOS is derived from.

    The equilibrium structure goes next to the ASE force cache in extended XYZ
    (exact cell, positions and species order as handed to ``Phonons``), so that
    :func:`phonon_dos_from_cache` can rebuild the dynamical matrix without a
    calculator. The ``dos`` block documents the sampling that produced the
    ``.dos`` file of this run and serves as the default when re-sampling.
    """
    # atoms.copy() drops the attached calculator, whose results are those of the
    # last displaced supercell and do not match the equilibrium cell.
    with _atomic_write(cache_dir / PHONON_EQUILIBRIUM_NAME) as f:
        write(f, atoms.copy(), format="extxyz")
    info = {
        "supercell": list(supercell),
        "delta": delta,
        "ML_model": ML_model,
        "ase_version": ase_version,
        "force_cache": PHONON_FORCE_CACHE_NAME,
        "equilibrium": PHONON_EQUILIBRIUM_NAME,
        "read": dict(PHONON_READ_DEFAULTS),
        "dos": {"kpts": list(kpts), "npts": npts, "width": width},
    }
    with _atomic_write(cache_dir / PHONON_INFO_NAME) as f:
        json.dump(info, f, indent=2)


def phonon_dos_from_cache(
    cache_dir: Path | str,
    kpts: tuple[int, int, int] | None = None,
    npts: int | None = None,
    width: float | None = None,
    method: str | None = None,
    symmetrize: int | None = None,
    acoustic: bool | None = None,
    cutoff: float | None = None,
    save_dos: Path | str | None = None,
) -> Any:
    """Re-sample a phonon DOS from a saved force cache, without any ML calls.

    ``cache_dir`` is a folder written by :func:`phonons` (as
    ``<run_folder>/phonons/<cluster_id>_cache`` when it comes from
    :func:`run_ml_relaxation`) holding the finite-displacement forces, the
    equilibrium structure and ``phonon_info.json``. Every parameter left as
    ``None`` falls back to the value recorded there, so

        phonon_dos_from_cache(cache_dir, npts=2000, width=5e-4)

    reproduces the original DOS at a finer sampling. ``kpts`` repeats the
    Fourier interpolation and the diagonalizations, the ``read`` parameters
    (``method``, ``symmetrize``, ``acoustic``, ``cutoff``) re-derive the force
    constants from the stored forces — neither re-runs the ML model.

    param cache_dir: folder holding the phonon force cache
    param save_dos: optional path to write the DOS to, in the two-column
        ``Energy(eV) DOS`` format read by :func:`egqca.io.read_phonons`
    return: ASE DOS object with the phonon density of states
    """
    cache_dir = Path(cache_dir)
    info_path = cache_dir / PHONON_INFO_NAME
    if not info_path.is_file():
        raise FileNotFoundError(f"No {PHONON_INFO_NAME} in {cache_dir}")
    with open(info_path) as f:
        info = json.load(f)

    atoms = read(cache_dir / info.get("equilibrium", PHONON_EQUILIBRIUM_NAME))
    force_dir = cache_dir / info.get("force_cache", PHONON_FORCE_CACHE_NAME)
    if not force_dir.is_dir():
        raise FileNotFoundError(f"No force cache in {force_dir}")

    read_kwargs = {**PHONON_READ_DEFAULTS, **info.get("read", {})}
    for key, value in (
        ("method", method),
        ("symmetrize", symmetrize),
        ("acoustic", acoustic),
        ("cutoff", cutoff),
    ):
        if value is not None:
            read_kwargs[key] = value

    dos_info = info.get("dos", {})
    kpts = tuple(kpts if kpts is not None else dos_info.get("kpts", (20, 20, 20)))
    npts = npts if npts is not None else dos_info.get("npts", 500)
    width = width if width is not None else dos_info.get("width", 1e-4)

    # calc=None: the forces come from the cache, the ML model is never loaded.
    ph = Phonons(
        atoms,
        calc=None,
        supercell=_phonon_supercell(info["supercell"]),
        delta=info["delta"],
        name=str(force_dir),
    )
    ph.read(**read_kwargs)
    dos = _sampled_phonon_dos(ph, len(atoms), kpts, npts, width)

    if save_dos is not None:
        save_dos = Path(save_dos)
        save_dos.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_write(save_dos) as f:
            np.savetxt(
                f,
                np.column_stack((dos.get_energies(), dos.get_weights())),
                fmt="%.6e",
                header="Energy(eV) DOS",
            )

    return dos


def geometry_relaxation(
    atoms: Atoms,
    ML_model: str,
    pressure_GPa: float | None = None,
    relaxation_constraint: bool | str | None = DEFAULT_RELAXATION_CONSTRAINT,
    CellFilter_mask: list[bool] | None = None,
    bravais_supercell: list[int] | tuple[int, int, int] = (1, 1, 1),
    fix_bravais_positions: bool = False,
    bravais_tol: float = DEFAULT_BRAVAIS_TOL,
    steps: int = 5000,
    device: str = "cpu",
    fmax: float = 1e-4,
    optimizer: Any = FIRE,
    optimizer_kwargs: dict | None = None,
    save_cif: Path | str | None = None,
    ML_model_name: str | None = None,
) -> Atoms:
    """Run geometry relaxation with an ML potential.

    Parameters
    ----------
    atoms : Atoms
        Input structure.
    ML_model : str, ASE calculator, or callable
        ML potential: a model string (e.g. ``"mattersim-5m"``), an ASE
        calculator, or a function returning one (see :func:`run_ml_relaxation`).
    pressure_GPa : float, optional
        External scalar pressure in GPa added to the stress.
    relaxation_constraint : bool, str, or None, optional
        Controls the space-group / Bravais-lattice ASE constraint applied
        during relaxation.
        Strings are case-insensitive (``"fcc"`` is ``"FCC"``); ``"auto"`` and
        ``"none"`` are accepted for ``True`` and ``None``, as on the command line.

        - ``None`` — no ASE constraint; cell degrees of freedom are
          controlled solely by ``CellFilter_mask``.
        - ``"FixSymmetry"`` (default) — ASE FixSymmetry constraint that
          preserves the space group of each cluster throughout the relaxation.
        - ``True`` — FixBravaisCell constraint with the Bravais type
          auto-detected from ``atoms``' own cell (within ``bravais_tol``)
          rather than specified explicitly. Note
          this only looks at the cell geometry, not the atomic decoration:
          for a set of clusters generated from one template (e.g. EGQCA's
          own substitutional-alloy workflow, where every cluster shares the
          template's pre-relaxation cell shape by construction), this
          detects the *same* Bravais type for every cluster -- including
          intermediate compositions whose true space group is lower once
          the atomic decoration breaks the end-members' symmetry. Use this
          when constraining every cluster to a common cell shape is the
          intended behaviour (e.g. so their excess energies are comparable
          on a consistent cell family); use an explicit Bravais key/crystal
          system (below) or ``"FixSymmetry"`` when it isn't.
        - ``str`` (Bravais lattice key or crystal system name, e.g.
          ``"RHL"``, ``"rhombohedral"``) — FixBravaisCell constraint that
          keeps the cell metric consistent with the specified Bravais type.
          Accepted keys: ``_BRAVAIS_LATTICE_SYSTEMS``; accepted crystal
          system names: ``_CRYSTAL_SYSTEMS``.
          Supercell inputs are handled automatically via ``bravais_supercell``.
    CellFilter_mask : list[bool] or None, optional
        Strain mask of six bools passed to ``FrechetCellFilter``, in ASE's Voigt
        order of the Cartesian strain components ``[xx, yy, zz, yz, xz, xy]``.
        ``True`` = component free to relax, ``False`` = fixed.  ``None``
        (default) leaves all six components free.

        .. note::
            These are not the cell parameters ``[a, b, c, alpha, beta, gamma]``.
            The two only coincide for an orthogonal cell aligned with the
            Cartesian axes; for a hexagonal, monoclinic or triclinic cell, e.g.,
            fixing ``yz, xz, xy`` does not keep the cell angles. To keep a cell
            shape, use ``relaxation_constraint`` instead.

        .. warning::
            Setting both ``relaxation_constraint`` and ``CellFilter_mask``
            applies an ASE constraint (FixSymmetry / FixBravaisCell) *and* a
            FrechetCellFilter mask simultaneously, which may cause unexpected
            behaviour.
    fix_bravais_positions : bool, optional
        Only used when ``relaxation_constraint`` is ``True``, a Bravais key or a
        crystal system string.  If ``True`` the atomic positions are frozen at
        their starting fractional coordinates (forces are zeroed), so only the
        cell relaxes: no internal relaxation at all, not a projection onto the
        symmetry-allowed positions.  Atoms of a decorated cluster generally have
        positional freedom, so this neglects their local relaxation.  ``False``
        (default) constrains only the cell metric / stress and leaves the
        internal coordinates free to relax.
    bravais_tol : float, optional
        Relative tolerance of the Bravais lattice detection used by the
        FixBravaisCell constraint (default 0.01, i.e. about 1 % in cell lengths
        and angles). A start cell within this tolerance of a higher-symmetry
        lattice is projected onto that exact lattice; the detected lattice is
        also the one reported as initial and target lattice. Only used when
        ``relaxation_constraint`` is ``True``, a Bravais key or a crystal system.
    steps : int
        Maximum number of optimizer steps.
    device : str
        Calculator device (``"cpu"`` or ``"cuda"``). Only used for model
        strings; a user-supplied calculator runs where it was set up.
    fmax : float
        Force convergence threshold in eV/Å.
    optimizer : optimizer class
        ASE optimizer class (default: FIRE).
    optimizer_kwargs : dict, optional
        Extra keyword arguments forwarded to the optimizer constructor.
    ML_model_name : str, optional
        Name of a user-supplied calculator, only used in the printed output.
    """
    # Suppress warnings about matrix logarithm accuracy issued by scipy during geometry
    # optimization to avoid cluttering the output
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="logm result may be inaccurate, approximate err",
    )

    # We suggest using double precison (``dtype="float64"``) for geometry optimization, as
    # it allows smaller values of the ``fmax`` convergence threshold and more reliable
    # symmetry detection.

    # case-insensitive, as on the command line ("fcc" -> "FCC")
    relaxation_constraint = _normalize_relaxation_constraint(relaxation_constraint)

    calc = _ml_calculator(ML_model, device=device)

    # Starting message
    bravais_constraint = None
    if relaxation_constraint is not None and relaxation_constraint != "FixSymmetry":
        bravais_constraint = _bravais_cell_constraint(
            relaxation_constraint, atoms, supercell=bravais_supercell,
            fix_positions=fix_bravais_positions, bravais_tol=bravais_tol,
        )
        # the lattice the constraint was built from
        initial_bravais_lattice = bravais_constraint.bravais_lattice
    else:
        initial_bravais_lattice = _bravais_lattice_name(atoms, supercell=bravais_supercell)
    print(f"\n{'='*62}")
    print(f"  ML model       : {_ml_model_description(ML_model, ML_model_name)}")
    print(f"  steps / fmax   : {steps} / {fmax} eV/Å")
    print(f"  Initial lattice: {_bravais_full_name(initial_bravais_lattice)} ({initial_bravais_lattice})")
    _print_cell(" "*18, atoms, supercell=bravais_supercell)

    if relaxation_constraint is None:
        print(f"  Constraint     : none")
    elif relaxation_constraint == "FixSymmetry":
        atoms.set_constraint(FixSymmetry(atoms))
        print(f"  Constraint     : FixSymmetry")
        report_symmetry(atoms, "  Space group    :")
    else:
        atoms.info["egqca_ml_initial_bravais_lattice"] = initial_bravais_lattice
        atoms.set_constraint(bravais_constraint)
        sc_str = "x".join(str(n) for n in bravais_supercell)
        pos_str = "cell+positions" if fix_bravais_positions else "cell only"
        if relaxation_constraint is True:
            print(
                f"  Target lattice : "
                f"{_bravais_full_name(initial_bravais_lattice)} ({initial_bravais_lattice})"
            )
        else:
            print(f"  Target lattice : {_bravais_full_name(relaxation_constraint)} ({relaxation_constraint})")
        print(f"  Supercell      : {sc_str}  [{pos_str}]")
        print(f"  Bravais tol    : {bravais_tol:g} (relative)")

    if CellFilter_mask is not None:
        CellFilter_mask = _validate_cell_mask(CellFilter_mask)
        print(f"  Cell mask      : {CellFilter_mask}")

    if relaxation_constraint is not None and CellFilter_mask is not None:
        warnings.warn(
            "Both relaxation_constraint and CellFilter_mask are set. "
            "Using an ASE constraint (FixSymmetry/FixBravaisCell) together with a "
            "FrechetCellFilter mask may cause unexpected behavior.",
            UserWarning,
            stacklevel=2,
        )

    cell_mask = CellFilter_mask if CellFilter_mask is not None else [True] * 6

    print(f"{'='*62}")
    print()


    atoms.calc = calc

    # ----------------------- optimization ----------------------- #
    cell_filter_kwargs = {}
    cell_filter_kwargs["mask"] = cell_mask
    if pressure_GPa is not None:
        cell_filter_kwargs["scalar_pressure"] = pressure_GPa * units.GPa

    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    cell_filter = FrechetCellFilter(atoms, **cell_filter_kwargs)
    opt_al = optimizer(
        cell_filter,
        **optimizer_kwargs,
    )
    converged = opt_al.run(fmax=fmax, steps=steps)
    atoms.info["egqca_ml_converged"] = converged
    final_forces = np.asarray(cell_filter.get_forces())
    atoms.info["egqca_ml_final_fmax"] = float(
        np.sqrt((final_forces**2).sum(axis=1)).max()
    )

    # ------------------------ print results ----------------------- #
    if not converged:
        print("Warning: Geometry optimization did not converge within the maximum number of steps.")

    # Lattice geometry
    print()
    print(f"{'='*13}"+" After relaxation "+f"{'='*13}")
    report_symmetry(atoms, "  Space group:    ")
    final_bravais_lattice = _bravais_lattice_name(atoms, supercell=bravais_supercell)
    atoms.info["egqca_ml_final_bravais_lattice"] = final_bravais_lattice
    print(f"  Bravais lattice: {_bravais_full_name(final_bravais_lattice)} ({final_bravais_lattice})")
    _print_cell(" "*18, atoms, supercell=bravais_supercell)

    # Additional information if fix bravais constraint
    if relaxation_constraint is True or (
        isinstance(relaxation_constraint, str) and relaxation_constraint != "FixSymmetry"
    ):
        # initial_bravais_lattice is already a resolved Bravais key (from
        # atoms.cell.get_bravais_lattice()), so the auto-detect case always
        # takes this branch rather than the crystal-system-name one below.
        key = initial_bravais_lattice if relaxation_constraint is True else str(relaxation_constraint).upper()
        requested_label = "auto-detected" if relaxation_constraint is True else "requested"
        if key in _BRAVAIS_LATTICE_SYSTEMS:
            # Accept any Bravais type in the same crystal system.
            # Cell.get_bravais_lattice() uses geometry only, so conventional cells
            # (e.g. BCC or FCC conventional) are reported as CUB; a strict equality
            # check would therefore always fail for conventional-cell inputs.
            target_systems = _BRAVAIS_LATTICE_SYSTEMS[key]
            final_systems = _BRAVAIS_LATTICE_SYSTEMS.get(final_bravais_lattice, set())
            if not target_systems & final_systems:
                _bravais_mismatch_warning = (
                    f"WARNING: Final Bravais lattice {final_bravais_lattice!r} (crystal system "
                    f"{sorted(final_systems)}) is not compatible with the {requested_label} "
                    f"Bravais type {key!r} (crystal system {sorted(target_systems)})."
                )
                print(_bravais_mismatch_warning)
                atoms.info["egqca_ml_bravais_mismatch"] = _bravais_mismatch_warning
        else:
            crystal_system = str(relaxation_constraint).lower()
            valid = {bl for bl, systems in _BRAVAIS_LATTICE_SYSTEMS.items()
                     if crystal_system in systems}
            if final_bravais_lattice not in valid:
                _bravais_mismatch_warning = (
                    f"WARNING: Final Bravais lattice {final_bravais_lattice!r} is not compatible "
                    f"with the requested crystal system {crystal_system!r}. "
                    f"Expected one of: {sorted(valid)}."
                )
                print(_bravais_mismatch_warning)
                atoms.info["egqca_ml_bravais_mismatch"] = _bravais_mismatch_warning

    if save_cif is not None:
        save_cif = Path(save_cif)
        save_cif.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_write(save_cif, "wb") as f:
            write(f, atoms, format="cif")

    return atoms


def _ml_model_label(ML_model, ML_model_name):
    if ML_model_name is not None:
        return ML_model_name
    return ML_model if isinstance(ML_model, str) else f"custom ({type(ML_model).__name__})"


@entry_point("ML relaxation", lambda b: [
    ("Clusters folder", b.arguments["clusters_folder"]),
    ("ML model", _ml_model_label(b.arguments["ML_model"], b.arguments["ML_model_name"])),
    ("Output folder", b.arguments["out_folder_name"] or "model name (default)"),
    ("Pressure (GPa)", b.arguments["pressure_GPa"]),
    ("Relaxation constraint", b.arguments["relaxation_constraint"]
     if b.arguments["relaxation_constraint"] in (None, "FixSymmetry")
     else f"{b.arguments['relaxation_constraint']} (bravais_tol {b.arguments['bravais_tol']:g})"),
    ("Cell filter mask", b.arguments["CellFilter_mask"]),
    ("fmax (eV/A) / max steps", f"{b.arguments['fmax']:g} / {b.arguments['steps']}"),
    ("Device", b.arguments["device"]),
    ("Phonons", f"yes (supercell {b.arguments['N_phonon_supercell']})" if b.arguments["calc_phonons"] else "no"),
])
def run_ml_relaxation(
    clusters_folder: str | Path,
    ML_model: MLModel = "mattersim-5m",
    ml_clusters_filename: str | None = None,
    ml_volume_filename: str | None = None,
    out_folder_name: str | None = None,
    save_relaxed_structure: bool = True,
    start_from_relaxed: bool = False,
    relaxation_constraint: bool | str | None = DEFAULT_RELAXATION_CONSTRAINT,
    CellFilter_mask: list[bool] | None = None,
    bravais_supercell: list[int] | tuple[int, int, int] = (1, 1, 1),
    fix_bravais_positions: bool = False,
    bravais_tol: float = DEFAULT_BRAVAIS_TOL,
    calc_phonons: bool = False,
    save_phonon_cache: bool = True,
    pressure_GPa: float | None = None,
    N_phonon_supercell: int | list[int] = 2,    # phonon supercell: int N (N x N x N) or [N1, N2, N3]
    steps: int = 5000,   # max optimization steps
    device: str = "cpu",
    fmax: float = 1e-4,   # eV/Å, force convergence threshold
    optimizer: Any = FIRE,
    optimizer_kwargs: dict | None = None,
    ML_model_name: str | None = None,
    permutable_sites: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Run ML relaxations for all cluster CIF files and write ML-specific cluster and volume files.

    Results are written into a run folder inside ``clusters_folder``. If
    ``out_folder_name`` is ``None``, the model name (``ML_model_name``, else
    the model string) is used, except for
    ``ML_model="eddp:<path>"`` where the folder is named
    ``"eddp-<file name without .eddp>"`` (the full model path would make for
    an unwieldy folder name).
    Slashes in the run folder name are replaced by underscores. The same name
    always means the same folder (as in the parallel script and
    :func:`rerun_nonconverged_ml`): a run folder that is already complete is not
    replaced; its results are returned. Use another ``out_folder_name`` for a new
    run with the same model.

    The cluster setup (n, sites, Nat, compound) is read from
    ``<clusters_folder>/clusters_template.dat``, else from ``DFT/clusters.dat`` of
    an older tree, else from a scan of the cluster CIFs; the template is written
    in the latter two cases.

    An incomplete run folder is continued, but only with the settings it was
    started with: the first run records them in ``<run_folder>/run_settings.json``
    (phonon supercell in ``phonons/phonon_settings.json``), and a run with
    different settings stops with an error listing them. ``steps`` and
    ``device`` may differ, see :func:`egqca._ml_utils._run_settings`.

    If ``save_relaxed_structure`` is ``True``, each relaxed structure is written
    into ``<run_folder>/cif_relaxed/<cluster_id>_<model name>.cif``.
    If it is ``False``, relaxed structures are not written.
    With ``start_from_relaxed=True``, a cluster that is relaxed (again) starts
    from its structure in ``cif_relaxed/`` if there is one, e.g. the last
    structure of a non-converged relaxation, instead of from its original CIF.
    Each relaxation writes its optimizer output to
    ``<run_folder>/ML_logs/clusters/<cluster_id>.log``.
    A global ``<run_folder>/ML_logs/<model name>.log`` records resume
    decisions, missing rows, non-converged relaxations, and final completeness.

    Parameters
    ----------
    ML_model : str, ASE calculator, or callable
        The ML potential. Either

        - a model string of one of the built-in model families (e.g.
          ``"mattersim-5m"``, ``"mace-medium-mpa-0"``, ``"pet-oam-xl"``,
          ``"GRACE-3L-OMAT-large-ft-AM"``, ``"eddp:<path>"``),
        - any ASE calculator providing energy, forces and stress, e.g. a
          fine-tuned model ``MACECalculator(model_paths="my.model")``, or
        - a function (or calculator class) that returns such a calculator when
          called without arguments, e.g. ``functools.partial(MyCalculator,
          model="my.ckpt")``. It is only called once a cluster actually needs
          relaxing, so a run with nothing left to do never loads the model.

        One calculator instance serves all clusters and their phonons. The
        ``device`` argument only applies to model strings; set the device of
        your own calculator when you build it.
    ML_model_name : str, optional
        Name of the model in the run folder, output file names and logs.
        Required if ``ML_model`` is a calculator or a function, defaults to
        the model string otherwise. A resumed run is matched by this name and
        by the model behind it: a model string, or the class of a user-supplied
        calculator. Checkpoints of one calculator class cannot be told apart, so
        give each its own name.
    relaxation_constraint : bool, str, or None, optional
        Space-group / Bravais-lattice ASE constraint forwarded to
        :func:`geometry_relaxation`. See that function for accepted values,
        including ``True`` for auto-detecting the Bravais type from each
        cluster's own (pre-relaxation) cell -- since EGQCA's substitutional
        clusters share one template's cell shape before relaxation, this
        constrains every cluster to the same cell family regardless of
        composition, which is the common case for a set of clusters whose
        excess energies should be compared on a consistent basis.
    CellFilter_mask : list[bool] or None, optional
        Strain mask of six bools forwarded to
        :func:`geometry_relaxation` and passed on to ``FrechetCellFilter``, in ASE's Voigt
        order of the Cartesian strain components ``[xx, yy, zz, yz, xz, xy]``.
        ``True`` = component free to relax, ``False`` = fixed.  ``None``
        (default) leaves all six components free.

        .. note::
            These are not the cell parameters ``[a, b, c, alpha, beta, gamma]``.
            The two only coincide for an orthogonal cell aligned with the
            Cartesian axes; for a hexagonal, monoclinic or triclinic cell, e.g.,
            fixing ``yz, xz, xy`` does not keep the cell angles. To keep a cell
            shape, use ``relaxation_constraint`` instead.

        .. warning::
            Setting both ``relaxation_constraint`` and ``CellFilter_mask``
            applies an ASE constraint (FixSymmetry / FixBravaisCell) *and* a
            FrechetCellFilter mask simultaneously, which may cause unexpected
            behaviour.
    bravais_supercell : list[int] or tuple[int, int, int], optional
        Diagonal supercell sizes ``[n1, n2, n3]`` describing how the cluster
        cells relate to the underlying primitive cell.  The lattice vectors
        are divided by these factors before the Bravais lattice is detected
        and before the FixBravaisCell symmetry operations are built, so
        supercell inputs are constrained to the correct primitive lattice
        type.  Default ``(1, 1, 1)`` (cells are treated as-is).  Only
        relevant when ``relaxation_constraint`` is a Bravais key or crystal
        system string; it additionally controls the lattice reported in the
        per-cluster logs.
    fix_bravais_positions : bool, optional
        Only used when ``relaxation_constraint`` is ``True``, a Bravais key or a
        crystal system string.  If ``True`` the atomic positions are frozen at
        their starting fractional coordinates (forces are zeroed), so only the
        cell relaxes: no internal relaxation at all, not a projection onto the
        symmetry-allowed positions.  Atoms of a decorated cluster generally have
        positional freedom, so this neglects their local relaxation.  ``False``
        (default) constrains only the cell metric / stress and leaves the
        internal coordinates of the cluster free to relax.
    bravais_tol : float, optional
        Relative tolerance of the Bravais lattice detection used by the
        FixBravaisCell constraint (default 0.01, i.e. about 1 % in cell lengths
        and angles). A start cell within this tolerance of a higher-symmetry
        lattice is projected onto that exact lattice; the detected lattice is
        also the one reported as initial and target lattice. Only used when
        ``relaxation_constraint`` is ``True``, a Bravais key or a crystal system.
    save_phonon_cache : bool, optional
        Only used when ``calc_phonons`` is ``True``.  If ``True`` (default) the
        finite-displacement forces of each cluster are kept in
        ``<run_folder>/phonons/<cluster_id>_cache/`` together with the
        equilibrium structure and ``phonon_info.json``.  The DOS can then be
        re-sampled with different ``kpts``/``npts``/``width`` (or different
        force-constant settings) via :func:`phonon_dos_from_cache` without
        re-running the ML model, and an interrupted phonon run resumes from the
        displacements already in the cache.
    permutable_sites : int, optional
        Number of permutable sites n, used for x = n_j / n. By default read from
        the clusters template (see above), else taken as the highest ``S*``
        folder number. Give it only when neither is right, e.g. for a tree
        without template whose clusters cover a sub-range of compositions; it is
        then written into the template. A value that differs from the template
        raises a ValueError.
    """

    t0 = time.monotonic()
    clusters_folder = Path(clusters_folder)
    # case-insensitive, as on the command line; recorded and compared in this form
    relaxation_constraint = _normalize_relaxation_constraint(relaxation_constraint)
    model_name = _ml_model_name(ML_model, ML_model_name)
    model_description = _ml_model_description(ML_model, ML_model_name)
    safe_ML_model = _ml_model_display_name(model_name)
    if ml_clusters_filename is None:
        ml_clusters_filename = f"clusters_{safe_ML_model}.dat"
    if ml_volume_filename is None:
        ml_volume_filename = f"volume_{safe_ML_model}.dat"

    permutable_sites, sites, Nat, compound, _ = _resolve_cluster_setup(
        clusters_folder, permutable_sites=permutable_sites,
    )
    target_cifs = _ml_target_cifs(clusters_folder)
    target_cluster_ids = {sub_dir.name for sub_dir, _ in target_cifs}
    run_folder = _prepare_ml_run_folder(
        clusters_folder,
        out_folder_name,
        model_name,
        ml_clusters_filename,
        ml_volume_filename,
        target_cluster_ids=target_cluster_ids,
        calc_phonons=calc_phonons,
    )
    # file-based models (eddp): refuse to resume a run folder made with another model file
    model_identity = _model_identity(ML_model)
    _check_model_identity(run_folder, model_identity)
    # refuse to continue a run folder with settings other than those it was started with
    _check_run_settings(
        run_folder,
        _run_settings(
            model=model_name,
            pressure_GPa=pressure_GPa,
            relaxation_constraint=relaxation_constraint,
            CellFilter_mask=CellFilter_mask,
            bravais_supercell=bravais_supercell,
            fix_bravais_positions=fix_bravais_positions,
            fmax=fmax,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            bravais_tol=bravais_tol,
            model_source=_ml_model_source(ML_model),
        ),
        _phonon_settings(N_phonon_supercell) if calc_phonons else None,
    )

    (
        ml_clusters_filename,
        ml_volume_filename,
        resume_existing,
    ) = _prepare_ml_output_files(
        run_folder,
        ml_clusters_filename,
        ml_volume_filename,
        target_cluster_ids=target_cluster_ids,
        permutable_sites=permutable_sites,
        sites=sites,
        compound=compound,
        ML_model=model_description,
    )
    clusters_file = run_folder / ml_clusters_filename
    volume_file = run_folder / ml_volume_filename
    # results of parallel jobs (ml_relaxation_parallel.py) in this run folder; the
    # shared run_settings.json guarantees they were made with the same settings
    per_cluster_files = sorted((run_folder / "per_cluster").glob(f"*_{safe_ML_model}_ml_result.dat"))
    per_cluster_rows, _ = _parse_result_files(per_cluster_files, safe_ML_model)
    imported_ids, imported_non_converged = _merge_per_cluster_results(
        [row for row in per_cluster_rows if row[0] in target_cluster_ids],
        clusters_file, volume_file,
        permutable_sites=permutable_sites, Nat_default=Nat,
    )
    completed_clusters = _read_completed_ids(clusters_file)
    completed_volumes = _read_completed_ids(volume_file)
    completed_ids = completed_clusters & completed_volumes
    pending_ids = target_cluster_ids - completed_ids

    global_log_file = _global_ml_log_file(run_folder, model_name)
    _write_global_ml_run_settings(
        global_log_file,
        clusters_folder=clusters_folder,
        run_folder=run_folder,
        ML_model=model_name,
        model_description=model_description,
        model_identity=model_identity,
        clusters_file=clusters_file,
        volume_file=volume_file,
        save_relaxed_structure=save_relaxed_structure,
        relaxation_constraint=relaxation_constraint,
        CellFilter_mask=CellFilter_mask,
        bravais_supercell=bravais_supercell,
        fix_bravais_positions=fix_bravais_positions,
        bravais_tol=bravais_tol,
        calc_phonons=calc_phonons,
        N_phonon_supercell=N_phonon_supercell,
        pressure_GPa=pressure_GPa,
        steps=steps,
        fmax=fmax,
        device=device if isinstance(ML_model, str) else "set by the user-supplied calculator",
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )
    _write_global_ml_log(
        global_log_file,
        f"Starting ML relaxation run for model {model_description}. \n"
        f"Total clusters: {len(target_cluster_ids)}. \n"
        f"Already complete: {len(completed_ids)}. \n"
        f"Pending: {len(pending_ids)}.\n"
    )
    if resume_existing:
        _write_global_ml_log(
            global_log_file,
            f"Continuing existing run folder: {run_folder.name}. "
            f"Output files: {clusters_file.name}, {volume_file.name}."
        )
    if imported_ids:
        _write_global_ml_log(
            global_log_file,
            f"Imported {len(imported_ids)} results from per_cluster/ (parallel runs)."
        )
    for cluster_id, final_fmax in imported_non_converged:
        final_fmax_text = "unavailable" if np.isnan(final_fmax) else f"{final_fmax:.6g}"
        _write_global_ml_log(
            global_log_file,
            f"Imported relaxation did not converge: {cluster_id}. Final fmax: {final_fmax_text}."
        )

    ML_energy = {}
    ML_latpar = {}
    n_converged = 0
    n_not_converged = 0
    n_skipped = len(completed_ids)
    n_phonons_resumed = 0
    bravais_mismatch_ids: list[str] = []
    phonon_dir = run_folder / "phonons"

    for sub_dir, cif_file in target_cifs:
        relaxed_cif_path = run_folder / "cif_relaxed" / f"{sub_dir.name}_{safe_ML_model}.cif"
        phonon_dos_path = phonon_dir / f"{sub_dir.name}.dos"
        # A cluster already in completed_ids has an energy/lattice row, but if
        # calc_phonons was turned on after that row was written (or the phonon
        # step previously raised and stopped the run before this cluster's turn
        # came around again), its phonon DOS may still be missing. Don't treat
        # it as complete in that case.
        phonons_pending = calc_phonons and not phonon_dos_path.exists()

        if sub_dir.name in completed_ids and not phonons_pending:
            print("Already complete: "+sub_dir.name)
            continue

        # Reuse the already-relaxed structure instead of paying for a full
        # relaxation again when only the phonon DOS is missing.
        resume_phonons_only = (
            sub_dir.name in completed_ids
            and phonons_pending
            and relaxed_cif_path.exists()
        )

        try:
            ml_output_file = _cluster_ml_log_file(run_folder, sub_dir.name)
            # a phonons-only pass keeps the relaxation's log (runtime, Bravais check)
            with open(ml_output_file, "a" if resume_phonons_only else "w") as f:
                if resume_phonons_only:
                    f.write(f"\n{_PHONONS_ONLY_SEPARATOR} (reusing {relaxed_cif_path.name}) =====\n")
                tee_stdout = _Tee(sys.stdout, f)
                tee_stderr = _Tee(sys.stderr, f)
                with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                    t_total0 = time.monotonic()
                    if resume_phonons_only:
                        print(
                            "\n----- Reusing relaxed structure for "+sub_dir.name+
                            " (phonons pending) -----"
                        )
                        atoms_relaxed = read(relaxed_cif_path)
                        t_relax = 0.0
                        n_phonons_resumed += 1
                    else:
                        if sub_dir.name in completed_ids:
                            print(
                                f"\n{sub_dir.name}: relaxation already recorded but phonon "
                                f"DOS is missing, and no relaxed CIF ({relaxed_cif_path}) is "
                                f"available to resume from; relaxing again from scratch."
                            )
                        if start_from_relaxed and relaxed_cif_path.exists():
                            print(f"\nStarting from the relaxed structure {relaxed_cif_path}")
                            atoms = read(relaxed_cif_path)
                        else:
                            atoms = read(cif_file)
                        print("\n----- Geometry relaxation for "+sub_dir.name+" -----")
                        _save_cif = (
                            relaxed_cif_path if save_relaxed_structure else None
                        )
                        atoms_relaxed = geometry_relaxation(
                            atoms.copy(),
                            ML_model=ML_model,
                            pressure_GPa=pressure_GPa,
                            relaxation_constraint=relaxation_constraint,
                            CellFilter_mask=CellFilter_mask,
                            bravais_supercell=bravais_supercell,
                            fix_bravais_positions=fix_bravais_positions,
                            bravais_tol=bravais_tol,
                            steps=steps,
                            device=device,
                            fmax=fmax,
                            optimizer=optimizer,
                            optimizer_kwargs=optimizer_kwargs,
                            save_cif=_save_cif,
                            ML_model_name=ML_model_name,
                        )
                        t_relax = time.monotonic() - t_total0
                        energy = atoms_relaxed.get_potential_energy()/(Ry_to_mev/1000)
                        latpar = np.hstack((atoms_relaxed.cell.lengths(), atoms_relaxed.cell.angles()))
                        ML_energy[sub_dir.name] = energy
                        ML_latpar[sub_dir.name] = latpar
                        if sub_dir.name not in completed_clusters:
                            _append_ml_cluster_result(
                                clusters_file,
                                sub_dir.name,
                                energy,
                                permutable_sites=permutable_sites,
                                Nat=Nat,
                                converged=atoms_relaxed.info.get("egqca_ml_converged", True),
                            )
                            completed_clusters.add(sub_dir.name)
                        if not atoms_relaxed.info.get("egqca_ml_converged", True):
                            n_not_converged += 1
                            final_fmax = atoms_relaxed.info.get("egqca_ml_final_fmax")
                            final_fmax_text = (
                                "unavailable" if final_fmax is None else f"{final_fmax:.6g}"
                            )
                            _write_global_ml_log(
                                global_log_file,
                                f"Relaxation did not converge within {steps} steps: {sub_dir.name}. "
                                f"Final fmax: {final_fmax_text}."
                            )
                        else:
                            n_converged += 1
                        if sub_dir.name not in completed_volumes:
                            _append_ml_volume_result(volume_file, sub_dir.name, latpar)
                            completed_volumes.add(sub_dir.name)
                        if atoms_relaxed.info.get("egqca_ml_bravais_mismatch"):
                            bravais_mismatch_ids.append(sub_dir.name)
                    if calc_phonons:
                        print("\n----- Phonon calculation for "+sub_dir.name+" -----")
                        t_phonon0 = time.monotonic()
                        # Keeping the forces makes the DOS re-samplable with
                        # egqca.ml.phonon_dos_from_cache, and lets an
                        # interrupted phonon run resume where it stopped.
                        cache_dir = (
                            phonon_dir / f"{sub_dir.name}_cache"
                            if save_phonon_cache else None
                        )
                        dos = phonons(
                            atoms_relaxed,
                            ML_model,
                            N_phonon_supercell,
                            device=device,
                            cache_dir=cache_dir,
                            ML_model_name=ML_model_name,
                        )
                        # Save DOS data: energy in first column, density of states in second column
                        dos_data = np.column_stack((dos.get_energies(), dos.get_weights()))
                        phonon_dir.mkdir(parents=True, exist_ok=True)
                        with _atomic_write(phonon_dos_path) as f_dos:
                            np.savetxt(f_dos, dos_data, fmt="%.6e", header="Energy(eV) DOS")
                        t_phonon = time.monotonic() - t_phonon0
                        print(f"Relaxation runtime: {t_relax:.2f} s")
                        print(f"Phonon runtime: {t_phonon:.2f} s")
                    print(_runtime_line(time.monotonic() - t_total0))
            #_write_global_ml_log(global_log_file, f"Completed {sub_dir.name}.")
        except Exception as exc:
            _write_global_ml_error(global_log_file, sub_dir.name, exc)
            raise

    _update_convergence_warning(clusters_file, volume_file)

    final_clusters = _read_completed_ids(clusters_file)
    final_volumes = _read_completed_ids(volume_file)
    still_missing = target_cluster_ids - (final_clusters & final_volumes)
    n_total_run = n_converged + n_not_converged
    summary = (
        f"Summary:\n"
        f"Converged:                    {n_converged}\n"
        f"Not converged:                {n_not_converged}\n"
        f"skipped (already complete):   {n_skipped}\n"
    )
    if calc_phonons and n_phonons_resumed:
        summary += (
            f"phonons computed for already-relaxed clusters (reused "
            f"cif_relaxed/, not re-relaxed): {n_phonons_resumed}\n"
        )
    _write_global_ml_log(global_log_file, summary)
    if bravais_mismatch_ids:
        _write_global_ml_log(
            global_log_file,
            f"Bravais lattice changed (incompatible with constraint): "
            f"{len(bravais_mismatch_ids)}/{n_total_run} clusters."
        )
    if still_missing:
        _write_global_ml_log(
            global_log_file,
            "Run ended with missing entries: " + ", ".join(sorted(still_missing))
        )
    else:
        _write_global_ml_log(global_log_file, "Run ended with complete output files.")
    if calc_phonons:
        for line in _phonon_stability_report_lines(run_folder / "phonons"):
            _write_global_ml_log(global_log_file, line)
    _write_global_ml_log(global_log_file, f"\nTotal runtime: {time.monotonic() - t0:.2f} s")

    # clusters completed by earlier runs or parallel jobs, from the files
    energy_rows, volume_rows = _read_ml_rows(clusters_file), _read_ml_rows(volume_file)
    for cid in sorted(target_cluster_ids - set(ML_energy)):
        if cid in energy_rows and cid in volume_rows:
            ML_energy[cid] = float(energy_rows[cid][5])
            ML_latpar[cid] = np.array([float(v) for v in volume_rows[cid][1:7]])

    return ML_energy, ML_latpar


def _ml_target_cifs(clusters_folder: str | Path) -> list[tuple[Path, Path]]:
    target_cifs = []
    for sub_dir in get_directories(clusters_folder):
        cif_file = _original_cluster_cif(sub_dir)
        if cif_file is not None:
            target_cifs.append((sub_dir, cif_file))
    return target_cifs


def _cluster_ml_log_file(run_folder: str | Path, cluster_id: str) -> Path:
    ml_logs_dir = Path(run_folder) / "ML_logs" / "clusters"
    ml_logs_dir.mkdir(parents=True, exist_ok=True)
    return ml_logs_dir / f"{cluster_id}.log"


def _global_ml_log_file(run_folder: str | Path, ML_model: str) -> Path:
    ml_logs_dir = Path(run_folder) / "ML_logs"
    ml_logs_dir.mkdir(exist_ok=True)
    ml_log_name = _ml_model_display_name(ML_model)
    return ml_logs_dir / f"{ml_log_name}.log"


def _write_global_ml_log(log_file: str | Path, message: str) -> None:
    with open(log_file, "a") as f:
        f.write(message + "\n")


def _write_global_ml_error(log_file: str | Path, cluster_id: str, exc: Exception) -> None:
    error_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    with open(log_file, "a") as f:
        f.write(f"\nERROR during ML relaxation for {cluster_id}.\n")
        f.write("Traceback:\n")
        f.write(error_text)
        if not error_text.endswith("\n"):
            f.write("\n")
        f.write("Stopping ML relaxation run.\n")


def _write_global_ml_run_settings(
    log_file: str | Path,
    *,
    clusters_folder: str | Path,
    run_folder: str | Path,
    ML_model: str,
    model_description: str | None = None,
    model_identity: dict | None = None,
    clusters_file: str | Path,
    volume_file: str | Path,
    save_relaxed_structure: bool,
    relaxation_constraint: bool | str | None,
    CellFilter_mask: list[bool] | None = None,
    bravais_supercell: list[int] | tuple[int, int, int] = (1, 1, 1),
    fix_bravais_positions: bool = False,
    bravais_tol: float = DEFAULT_BRAVAIS_TOL,
    calc_phonons: bool,
    N_phonon_supercell: int | list[int],
    pressure_GPa: float | None,
    steps: int,
    fmax: float,
    device: str,
    optimizer: Any = FIRE,
    optimizer_kwargs: dict | None = None,
) -> None:
    safe_ML_model = _ml_model_display_name(ML_model)
    relaxed_structure_pattern = (
        f"cif_relaxed/<cluster_id>_{safe_ML_model}.cif" if save_relaxed_structure else "None"
    )
    optimizer_kwargs_log = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    sc_str = "x".join(str(n) for n in bravais_supercell)
    phonon_sc = (
        (N_phonon_supercell, N_phonon_supercell, N_phonon_supercell)
        if isinstance(N_phonon_supercell, int)
        else N_phonon_supercell
    )
    phonon_sc_str = "x".join(str(n) for n in phonon_sc)

    settings = [
        "ML relaxation settings:",
        f"  clusters_folder: {clusters_folder}",
        f"  run_folder: {run_folder}",
        f"  model: {model_description or ML_model}",
        *([f"  model file: {model_identity['path']}",
           f"  model sha256: {model_identity['sha256']}"] if model_identity else []),
        f"  clusters file: {Path(clusters_file).name}",
        f"  volume file: {Path(volume_file).name}",
        f"  save_relaxed_structure: {save_relaxed_structure}",
        f"  relaxed structure filename: {relaxed_structure_pattern}",
        f"  relaxation_constraint: {relaxation_constraint}",
        f"  CellFilter_mask: {CellFilter_mask}",
        f"  bravais_supercell: {sc_str}",
        f"  fix_bravais_positions: {fix_bravais_positions}",
        f"  bravais_tol: {bravais_tol}",
        f"  phonons calculated: {calc_phonons}",
        f"  phonon DOS files: {'phonons/<cluster_id>.dos' if calc_phonons else 'None'}",
        f"  phonon supercell size: {phonon_sc_str if calc_phonons else 'None'}",
        f"  pressure_GPa: {pressure_GPa}",
        f"  steps: {steps}",
        f"  fmax: {fmax}",
        f"  device: {device}",
        f"  optimizer: {_optimizer_name(optimizer)}",
        f"  optimizer_kwargs: {optimizer_kwargs_log}",
    ]
    with open(log_file, "a") as f:
        f.write("\n".join(settings) + "\n")


def _read_ml_rows(path: str | Path) -> dict[str, list[str]]:
    """Complete data rows of a clusters/volume file by cluster ID (fields without comment)."""
    path = Path(path)
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            if not line.endswith("\n"):
                break
            fields = line.partition("#")[0].split()
            if len(fields) >= 6:
                rows[fields[0]] = fields
    return rows


def _read_completed_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()

    completed_ids = set()
    with open(path) as f:
        for line in f:
            # rows are written with their newline in one call: a last line
            # without it was cut off by a killed job
            if not line.endswith("\n"):
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 6:
                continue
            completed_ids.add(fields[0])
    return completed_ids


def _drop_partial_last_line(path: Path) -> None:
    """Remove a last line without newline, left by an interrupted append."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
    text = path.read_text()
    with _atomic_write(path) as f:
        f.write(text[: text.rfind("\n") + 1])


def _read_non_converged_legacy_ids(path: str | Path) -> set[str]:
    """Return cluster IDs flagged non-converged in an existing clusters_<model>.dat.

    Used by :func:`collect_ml_results` to recover convergence status for rows
    that were already present before collection (e.g. written directly by
    :func:`run_ml_relaxation`) and are therefore not re-parsed from a
    per-cluster result file.
    """
    path = Path(path)
    if not path.exists():
        return set()

    non_converged_ids = set()
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "Relaxation did not converge" in stripped:
                non_converged_ids.add(stripped.split()[0])
    return non_converged_ids


def _ml_phonons_complete(run_folder: str | Path, target_cluster_ids: set[str]) -> bool:
    """Whether every target cluster already has a phonon DOS file in run_folder/phonons/."""
    if not target_cluster_ids:
        return False
    phonon_dir = Path(run_folder) / "phonons"
    return all((phonon_dir / f"{cid}.dos").exists() for cid in target_cluster_ids)


def _prepare_ml_output_files(
    clusters_folder: str | Path,
    ml_clusters_filename: str,
    ml_volume_filename: str,
    *,
    target_cluster_ids: set[str],
    permutable_sites: int | None = None,
    sites: int | None = None,
    compound: str | None = None,
    ML_model: str | None = None,
) -> tuple[str, str, bool]:
    clusters_folder = Path(clusters_folder)
    clusters_file = clusters_folder / ml_clusters_filename
    volume_file = clusters_folder / ml_volume_filename
    resume_existing = clusters_file.exists() or volume_file.exists()

    if not clusters_file.exists():
        _write_ml_clusters_header(
            clusters_file,
            permutable_sites=permutable_sites,
            sites=sites,
            compound=compound,
            ML_model=ML_model,
        )
    if not volume_file.exists():
        _write_ml_volume_header(volume_file, ML_model=ML_model)

    return ml_clusters_filename, ml_volume_filename, resume_existing


def _prepare_ml_run_folder(
    clusters_folder: str | Path,
    out_folder_name: str | None,
    ML_model: str,
    ml_clusters_filename: str,
    ml_volume_filename: str,
    *,
    target_cluster_ids: set[str],
    calc_phonons: bool = False,
) -> Path:
    clusters_folder = Path(clusters_folder)
    folder_name = (
        _safe_ml_name(out_folder_name)
        if out_folder_name is not None
        else _ml_model_display_name(ML_model)
    )
    if not folder_name:
        raise ValueError("out_folder_name must not be empty.")

    # One name, one folder, as in the parallel script and the rerun: a complete
    # folder is not replaced by a new one; its results are returned. A new run
    # with the same model needs another out_folder_name.
    run_folder = clusters_folder / folder_name
    if _ml_run_folder_is_complete(
        run_folder,
        ml_clusters_filename,
        ml_volume_filename,
        target_cluster_ids=target_cluster_ids,
        calc_phonons=calc_phonons,
    ):
        print(f"{run_folder} is already complete; returning its results. "
              "Use another out_folder_name for a new run.")
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def _ml_run_folder_is_complete(
    run_folder: str | Path,
    ml_clusters_filename: str,
    ml_volume_filename: str,
    *,
    target_cluster_ids: set[str],
    calc_phonons: bool = False,
) -> bool:
    run_folder = Path(run_folder)
    clusters_file = run_folder / ml_clusters_filename
    volume_file = run_folder / ml_volume_filename
    complete = (
        clusters_file.exists()
        and volume_file.exists()
        and bool(target_cluster_ids)
        and target_cluster_ids.issubset(_read_completed_ids(clusters_file))
        and target_cluster_ids.issubset(_read_completed_ids(volume_file))
    )
    # See _prepare_ml_output_files: a folder still missing phonon DOS files
    # must not be treated as complete when phonons are requested, or a
    # phonons-only resume would fork off a brand new run folder instead.
    if complete and calc_phonons:
        complete = _ml_phonons_complete(run_folder, target_cluster_ids)
    return complete


_NOT_CONVERGED_WARNING = (
    "# WARNING: Relaxation did not converge for some of the clusters. "
    "For more information refer to the log-file."
)


def _update_convergence_warning(clusters_path: str | Path, volume_path: str | Path) -> None:
    """Keep one header warning in both files exactly while a cluster row is flagged non-converged.

    Derived from the rows, so a resume does not add it again and a rerun that
    converged removes it. Each file is replaced atomically.
    """
    needed = bool(_read_non_converged_legacy_ids(clusters_path))
    for path in (Path(clusters_path), Path(volume_path)):
        if not path.exists():
            continue
        lines = [l for l in path.read_text().splitlines(keepends=True) if l.rstrip("\n") != _NOT_CONVERGED_WARNING]
        if needed:
            lines.insert(0, _NOT_CONVERGED_WARNING + "\n")
        with _atomic_write(path) as f:
            f.write("".join(lines))


def _append_ml_cluster_result(
    path: str | Path,
    cluster_id: str,
    energy: float,
    *,
    permutable_sites: int,
    Nat: int,
    converged: bool = True,
) -> None:
    n_j, x, g_j = _cluster_info_from_id(cluster_id, permutable_sites)
    _drop_partial_last_line(Path(path))
    with open(path, "a") as f:
        line = "\t".join(map(str, (cluster_id, n_j, Nat, x, g_j))) + f"\t{energy}"
        if not converged:
            line += "\t  # WARNING: Relaxation did not converge"
        line += "\n"
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _append_ml_volume_result(path: str | Path, cluster_id: str, latpar: np.ndarray | list[float]) -> None:
    _drop_partial_last_line(Path(path))
    with open(path, "a") as f:
        f.write(cluster_id+"\t"+"\t".join(map(str, latpar)) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---- Result file column layout (written by ml_relaxation_parallel.py) ---- #
# cluster_id  n_j  Nat  x  g_j  energy_Ry  a  b  c  alpha  beta  gamma  converged  final_fmax_eV_per_A
_RESULT_HEADER = (
    "cluster_id\tn_j\tNat\tx\tg_j\tenergy_Ry"
    "\ta\tb\tc\talpha\tbeta\tgamma\tconverged\tfinal_fmax_eV_per_A"
)


def _read_ml_result_file(
    path: str | Path,
) -> tuple[str, int | None, int | None, float | None, int | None, float, float, float, float, float, float, float, bool, float] | None:
    """Parse a single *_ml_result.dat file.

    Returns None if the file has no data row or its first data row is malformed,
    e.g. truncated by a killed job.
    """
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                return _parse_ml_result_row(stripped)
            except (IndexError, ValueError):
                return None
    return None


def _parse_ml_result_row(stripped: str):
    """Fields of one tab-separated *_ml_result.dat data row."""
    fields = stripped.split("\t")
    cluster_id = fields[0]
    n_j   = int(fields[1])   if fields[1]  != "None" else None
    Nat   = int(fields[2])   if fields[2]  != "None" else None
    x     = float(fields[3]) if fields[3]  != "None" else None
    g_j   = int(fields[4])   if fields[4]  != "None" else None
    energy_ry = float(fields[5])
    a, b, c, alpha, beta, gamma = map(float, fields[6:12])
    converged = True
    if len(fields) > 12:
        flag = fields[12].strip().lower()
        if flag not in ("true", "1", "false", "0"):
            raise ValueError(f"converged flag {fields[12]!r}")
        converged = flag in ("true", "1")
    final_fmax = float(fields[13]) if len(fields) > 13 and fields[13].strip() not in ("None", "nan", "") else float("nan")
    return cluster_id, n_j, Nat, x, g_j, energy_ry, a, b, c, alpha, beta, gamma, converged, final_fmax


_PHONONS_ONLY_SEPARATOR = "===== Phonons-only resume"


def _extract_log_runtime(log_file: Path) -> float | None:
    """Return the runtime in seconds of a cluster log: the sum of its :func:`_runtime_line` lines.

    A relaxation and a later phonons-only pass appended to the same log each
    write one. 'Total runtime:' lines are accepted too: the per-cluster script
    (then ml_relaxation_per_cluster.py) wrote those before it used _runtime_line.
    """
    runtimes = []
    try:
        with open(log_file) as f:
            for line in f:
                m = re.match(r'^(?:Total runtime|Runtime):\s+([\d.]+)\s+s', line)
                if m:
                    runtimes.append(float(m.group(1)))
    except OSError:
        pass
    return sum(runtimes) if runtimes else None


# Phonon frequencies (in the DOS file's energy units, eV for ML runs) with
# magnitude below this value are treated as numerical noise around zero rather
# than genuine imaginary modes.
_PHONON_NEGATIVE_FREQ_TOL = 1e-4


def _has_negative_phonon_frequencies(
    dos_file: str | Path,
    freq_tol: float = _PHONON_NEGATIVE_FREQ_TOL,
) -> bool | None:
    """Return whether a phonon DOS file contains negative frequencies.

    Negative frequencies (imaginary modes) indicate a dynamically unstable
    structure. Returns ``True`` if the DOS has appreciable weight at frequencies
    below ``-freq_tol``, ``False`` if not, and ``None`` if the file cannot be read.
    """
    try:
        data = np.loadtxt(dos_file)
    except (OSError, ValueError):
        return None
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] == 0:
        return None
    freqs = data[:, 0]
    weights = data[:, 1]
    wmax = float(weights.max())
    if wmax <= 0.0:
        return False
    # Ignore the smeared DOS tails by requiring non-negligible weight.
    significant = weights > 1e-3 * wmax
    return bool(np.any(freqs[significant] < -freq_tol))


def _phonon_stability_summary(
    phonon_dir: str | Path,
) -> tuple[int, list[str], list[str]]:
    """Scan a directory of ``<cluster_id>.dos`` files for dynamic stability.

    Returns ``(n_stable, unstable_ids, unreadable_ids)``.
    """
    phonon_dir = Path(phonon_dir)
    n_stable = 0
    unstable_ids: list[str] = []
    unreadable_ids: list[str] = []
    for dos_file in sorted(phonon_dir.glob("*.dos")):
        negative = _has_negative_phonon_frequencies(dos_file)
        if negative is None:
            unreadable_ids.append(dos_file.stem)
        elif negative:
            unstable_ids.append(dos_file.stem)
        else:
            n_stable += 1
    return n_stable, unstable_ids, unreadable_ids


def _phonon_stability_report_lines(
    phonon_dir: str | Path,
    max_listed: int = 50,
) -> list[str]:
    """Build human-readable overview lines of phonon dynamic stability.

    Returns an empty list when no phonon DOS files are present, so callers can
    skip the section entirely.
    """
    n_stable, unstable_ids, unreadable_ids = _phonon_stability_summary(phonon_dir)
    n_total = n_stable + len(unstable_ids) + len(unreadable_ids)
    if n_total == 0:
        return []
    lines = [
        f"Phonon dynamic stability ({n_total} clusters with phonon DOS):",
        f"  dynamically stable:   {n_stable}",
        f"  dynamically unstable: {len(unstable_ids)}",
    ]
    if unstable_ids:
        shown = ", ".join(unstable_ids[:max_listed])
        if len(unstable_ids) > max_listed:
            shown += ", ..."
        lines.append(f"  unstable clusters: {shown}")
    if unreadable_ids:
        lines.append(f"  unreadable DOS files: {len(unreadable_ids)}")
    return lines


def _extract_log_bravais_mismatch(log_file: Path) -> bool:
    """Return True if the per-cluster log file contains a Bravais lattice mismatch warning."""
    try:
        with open(log_file) as f:
            for line in f:
                if line.startswith("WARNING: Final Bravais lattice"):
                    return True
    except OSError:
        pass
    return False


def _extract_log_header(log_file: Path) -> str:
    """Return the header lines of an S*.log file (before the optimizer step table)."""
    lines: list[str] = []
    try:
        with open(log_file) as f:
            for line in f:
                if line.startswith("cluster_dir"):
                    continue
                if re.match(r'^[A-Z][A-Za-z0-9]+:\s+\d+\s', line):
                    break
                if line.startswith("-----"):
                    break
                lines.append(line)
    except OSError:
        return ""
    while lines and not lines[-1].strip():
        lines.pop()
    return "  " + "  ".join(lines) if lines else ""


def _parse_result_files(
    files: list[Path], safe_model: str, progress: bool = False,
) -> tuple[list[tuple], list[str]]:
    """Rows of the readable ``*_ml_result.dat`` files and the IDs of the unreadable ones.

    Empty, header-only or malformed files (e.g. a killed job) are unreadable.
    """
    rows, unreadable = [], []
    for i, rf in enumerate(files, 1):
        if progress:
            print(f"\rCollecting ML results: {i}/{len(files)} ({100*i//len(files)}%)", end="", flush=True)
        row = _read_ml_result_file(rf)
        if row is None:
            warnings.warn(f"No readable data row in {rf}; the cluster is treated as missing.")
            unreadable.append(rf.stem[: -len(f"_{safe_model}_ml_result")])
        else:
            rows.append(row)
    if progress and files:
        print()
    return rows, unreadable


def _merge_per_cluster_results(
    rows: list[tuple],
    clusters_path: Path,
    volume_path: Path,
    *,
    permutable_sites: int,
    Nat_default: int,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Append per-cluster result rows to the clusters and volume files.

    Only clusters without a row yet are added; existing rows are never replaced.
    Returns the IDs added to the clusters file and the non-converged ones among
    them with their final fmax.
    """
    existing_cluster_ids = _read_completed_ids(clusters_path)
    existing_volume_ids = _read_completed_ids(volume_path)
    added: list[str] = []
    non_converged: list[tuple[str, float]] = []
    for row in rows:
        cluster_id, n_j, Nat, x, g_j, energy_ry, a, b, c, alpha, beta, gamma, converged, final_fmax = row
        if cluster_id not in existing_cluster_ids:
            _append_ml_cluster_result(
                clusters_path,
                cluster_id,
                energy_ry,
                permutable_sites=permutable_sites,
                Nat=Nat if Nat is not None else Nat_default,
                converged=converged,
            )
            existing_cluster_ids.add(cluster_id)
            added.append(cluster_id)
            if not converged:
                non_converged.append((cluster_id, final_fmax))
        if cluster_id not in existing_volume_ids:
            _append_ml_volume_result(volume_path, cluster_id, [a, b, c, alpha, beta, gamma])
            existing_volume_ids.add(cluster_id)
    return added, non_converged


def _detect_ml_model(result_files: list[Path]) -> str:
    """The one model name in ``*_ml_result.dat`` file names."""
    detected: set[str] = set()
    for rf in result_files:
        m = re.match(r'^(S\d+(?:_d\d+)?_i\d+_w\d+)_(.+)_ml_result$', rf.stem)
        if m:
            detected.add(m.group(2))
    if not detected:
        raise ValueError("Could not detect ML model name from result filenames.")
    if len(detected) > 1:
        raise ValueError(
            f"Multiple ML models found: {sorted(detected)}. Specify model= explicitly."
        )
    return detected.pop()


def _remove_rows(path: Path, cluster_ids: set[str]) -> list[str]:
    """Remove the data rows of ``cluster_ids`` from a clusters/volume file, atomically."""
    if not path.exists():
        return []
    kept, removed = [], []
    for line in path.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.split()[0] in cluster_ids:
            removed.append(stripped.split()[0])
        else:
            kept.append(line)
    if removed:
        with _atomic_write(path) as f:
            f.write("".join(kept))
    return removed


def reset_ml_clusters(
    run_folder: str | Path,
    model: str | None = None,
    cluster_ids: Iterable[str] | None = None,
    *,
    clusters_folder: str | Path | None = None,
    ml_clusters_filename: str | None = None,
    ml_volume_filename: str | None = None,
) -> list[str]:
    """Undo the results of clusters so that they are relaxed again.

    Removes the clusters' rows from ``clusters_<model>.dat`` and
    ``volume_<model>.dat``, their ``per_cluster/`` result files and their phonon
    DOS; phonon force caches are discarded by :func:`phonons` once the structure
    has changed. Both :func:`run_ml_relaxation` and ``ml_relaxation_parallel.py``
    then relax them again, e.g. with more ``steps``.

    ``cluster_ids=None`` resets the non-converged clusters: those flagged in the
    clusters file and those whose ``per_cluster/`` result is not converged, also
    when it has not been collected yet. ``model`` is detected from the result
    file names when ``None``. The reset clusters are written to
    ``<clusters_folder>/rerun_clusters.txt`` in the ``<S folder>, <cluster id>``
    format of ``missing_clusters.txt``, ready to be queued; ``clusters_folder``
    defaults to the parent of ``run_folder``. Returns the reset cluster IDs.
    """
    run_folder = Path(run_folder)
    clusters_folder = Path(clusters_folder) if clusters_folder is not None else run_folder.parent
    per_cluster_dir = run_folder / "per_cluster"
    if model is None:
        safe_model = _detect_ml_model(sorted(per_cluster_dir.glob("*_ml_result.dat")))
    else:
        safe_model = _ml_model_display_name(model)
    clusters_path = run_folder / (ml_clusters_filename or f"clusters_{safe_model}.dat")
    volume_path = run_folder / (ml_volume_filename or f"volume_{safe_model}.dat")
    result_file = lambda cid: per_cluster_dir / f"{cid}_{safe_model}_ml_result.dat"

    if cluster_ids is None:
        targets = set(_read_non_converged_legacy_ids(clusters_path))
        for rf in per_cluster_dir.glob(f"*_{safe_model}_ml_result.dat"):
            row = _read_ml_result_file(rf)
            if row is not None and not row[12]:
                targets.add(row[0])
    else:
        targets = set(cluster_ids)
    # numeric order of S, d, i, w, as in the other cluster lists
    targets = sorted(targets, key=lambda cid: [int(n) for n in re.findall(r"\d+", cid)])

    _remove_rows(clusters_path, set(targets))
    _remove_rows(volume_path, set(targets))
    _update_convergence_warning(clusters_path, volume_path)
    for cid in targets:
        result_file(cid).unlink(missing_ok=True)
        (run_folder / "phonons" / f"{cid}.dos").unlink(missing_ok=True)

    rerun_path = clusters_folder / "rerun_clusters.txt"
    if targets:
        with open(rerun_path, "w") as f:
            for cid in targets:
                m = re.match(r"(S\d+)", cid)
                f.write(f"{m.group(1) if m else cid}, {cid}\n")
        print(f"Reset {len(targets)} clusters; queue them from {rerun_path}")
    else:
        # nothing to rerun: an old list would requeue clusters that are done
        rerun_path.unlink(missing_ok=True)
        print("No clusters to reset.")
    return targets


def rerun_nonconverged_ml(
    clusters_folder: str | Path,
    ML_model: MLModel,
    *,
    ML_model_name: str | None = None,
    out_folder_name: str | None = None,
    steps: int = 20000,
    device: str = "cpu",
    optimizer_kwargs: dict | None = None,
    start_from_relaxed: bool = True,
) -> list[str]:
    """Relax the non-converged clusters of an existing run folder again.

    The clusters are reset with :func:`reset_ml_clusters` and relaxed with
    :func:`run_ml_relaxation`, using the settings recorded in the run folder; only
    ``steps``, ``device`` and ``start_from_relaxed`` are taken from the arguments.
    ``ML_model`` and ``ML_model_name`` must be those of the original run (checked
    against the recorded settings). By default the relaxation continues from the
    last structure in ``cif_relaxed/``. ``device`` is not recorded and defaults to
    ``"cpu"``: pass it again for a GPU run. ``optimizer_kwargs`` is only needed when
    the original values were not plain JSON and could not be recorded.

    Returns the IDs of the clusters that were relaxed again.
    """
    clusters_folder = Path(clusters_folder)
    model_name = _ml_model_name(ML_model, ML_model_name)
    run_folder = clusters_folder / (out_folder_name or _ml_model_display_name(model_name))
    settings, phonon_settings = _load_run_settings(run_folder)
    if optimizer_kwargs is None:
        if settings.get("optimizer_kwargs_not_json"):
            raise ValueError(
                f"The optimizer_kwargs {settings['optimizer_kwargs_not_json']} of {run_folder} were "
                "not plain JSON values and are only recorded as text. Pass them as optimizer_kwargs."
            )
        optimizer_kwargs = settings["optimizer_kwargs"]

    # n as the run used it: its clusters file records it in the header (needed for a
    # tree without DFT/clusters.dat, where the run was started with permutable_sites)
    run_clusters_file = run_folder / f"clusters_{_ml_model_display_name(model_name)}.dat"
    permutable_sites = _header_permutable_sites(run_clusters_file) if run_clusters_file.exists() else None

    nonconverged = reset_ml_clusters(run_folder, model_name, clusters_folder=clusters_folder)
    if not nonconverged:
        return []
    run_ml_relaxation(
        clusters_folder,
        permutable_sites=permutable_sites,
        ML_model=ML_model,
        ML_model_name=ML_model_name,
        out_folder_name=run_folder.name,
        steps=steps,
        device=device,
        start_from_relaxed=start_from_relaxed,
        pressure_GPa=settings["pressure_GPa"],
        relaxation_constraint=settings["relaxation_constraint"],
        # an all-free mask is recorded for None; None avoids the constraint-and-mask warning
        CellFilter_mask=None if all(settings["CellFilter_mask"]) else settings["CellFilter_mask"],
        bravais_supercell=settings["bravais_supercell"],
        fix_bravais_positions=settings["fix_bravais_positions"],
        bravais_tol=settings.get("bravais_tol", DEFAULT_BRAVAIS_TOL),
        fmax=settings["fmax"],
        optimizer=_parse_optimizer(settings["optimizer"]),
        optimizer_kwargs=optimizer_kwargs,
        calc_phonons=phonon_settings is not None,
        N_phonon_supercell=phonon_settings["phonon_supercell"] if phonon_settings else 2,
        save_relaxed_structure=True,
    )
    return nonconverged


def collect_ml_results(
    run_folder: str | Path,
    *,
    clusters_folder: str | Path | None = None,
    clusters_template: str | Path | None = None,
    model: str | None = None,
    ml_clusters_filename: str | None = None,
    ml_volume_filename: str | None = None,
    log_file: str | Path | None = None,
    permutable_sites: int | None = None,
) -> tuple[Path, Path, Path]:
    """Aggregate per-cluster *_ml_result.dat files into clusters_*.dat and volume_*.dat.

    ``run_folder`` is the model run folder (e.g. ``<clusters_folder>/mattersim-5m/``).
    Result files are read from ``run_folder/per_cluster/``.
    Aggregated output (clusters_*.dat, volume_*.dat) and the convergence log are
    written into ``run_folder``.
    ``clusters_folder`` defaults to ``run_folder.parent`` and is used to infer
    system setup (permutable sites, atom count, compound name) from the S* structure.
    Header metadata (permutable_sites, sites, Nat, compound) and the list of all
    clusters are read from ``clusters_template`` if given, else from
    ``<clusters_folder>/clusters_template.dat``, else from ``DFT/clusters.dat`` of an
    older tree, else from a scan of the cluster CIFs with n = ``permutable_sites`` or
    the highest S* folder number; in the latter two cases clusters_template.dat is
    written. A ``permutable_sites`` that differs from the template raises a ValueError.
    If ``model`` is ``None`` the model name is inferred from the filenames; an error
    is raised when multiple models are found.
    Returns the paths to the written clusters, volume, and log files.
    """
    run_folder = Path(run_folder)
    per_cluster_dir = run_folder / "per_cluster"
    clusters_folder = Path(clusters_folder) if clusters_folder is not None else run_folder.parent

    result_files = sorted(per_cluster_dir.glob("*_ml_result.dat"))
    if not result_files:
        warnings.warn(f"No *_ml_result.dat files found under {per_cluster_dir}")

    safe_model = _detect_ml_model(result_files) if model is None else _ml_model_display_name(model)

    matching = [
        rf for rf in result_files
        if rf.stem.endswith(f"_{safe_model}_ml_result")
    ]
    if not matching:
        warnings.warn(f"No *_{safe_model}_ml_result.dat files found under {per_cluster_dir}")
    matching.sort(key=lambda p: tuple(
        int(x) for x in re.match(r"S(\d+)(?:_d(\d+))?_i(\d+)_w(\d+)", p.name).groups(default='0')
    ))

    permutable_sites, sites, Nat_default, compound, all_cluster_ids = _resolve_cluster_setup(
        clusters_folder, permutable_sites=permutable_sites, clusters_template=clusters_template,
    )

    if ml_clusters_filename is None:
        ml_clusters_filename = f"clusters_{safe_model}.dat"
    if ml_volume_filename is None:
        ml_volume_filename = f"volume_{safe_model}.dat"

    run_folder.mkdir(parents=True, exist_ok=True)
    clusters_path = run_folder / ml_clusters_filename
    volume_path = run_folder / ml_volume_filename
    log_path = Path(log_file) if log_file is not None else run_folder / f"ML_logs/convergence_{safe_model}.log"

    # clusters_path/volume_path may already hold rows from an earlier
    # run_ml_relaxation (sequential) run, or an earlier collect_ml_results call,
    # on this same run folder. Those rows have no corresponding per_cluster/
    # file, so they must be preserved rather than clobbered: the header is only
    # (re-)written when the file doesn't exist yet, and only cluster IDs not
    # already present are appended below.

    # parse first: only files with a data row count as found; unreadable files
    # (e.g. a killed job) are missing, so they get requeued
    parsed_rows, unreadable_ids = _parse_result_files(matching, safe_model, progress=True)
    matching_ids = {row[0] for row in parsed_rows}

    if not clusters_path.exists():
        _write_ml_clusters_header(
            clusters_path,
            permutable_sites=permutable_sites,
            sites=sites,
            compound=compound,
            ML_model=model or safe_model,
        )
    if not volume_path.exists():
        _write_ml_volume_header(volume_path, ML_model=model or safe_model)

    _merge_per_cluster_results(
        parsed_rows, clusters_path, volume_path,
        permutable_sites=permutable_sites, Nat_default=Nat_default,
    )
    _update_convergence_warning(clusters_path, volume_path)

    # counts from the state of the files after the merge, so that every call
    # reports the same for the same results
    cluster_order = lambda cid: [int(n) for n in re.findall(r"\d+", cid)]
    recorded_ids = _read_completed_ids(clusters_path) & _read_completed_ids(volume_path)
    flagged_ids = _read_non_converged_legacy_ids(clusters_path)
    fmax_by_id = {row[0]: row[13] for row in parsed_rows}
    not_converged_ids = sorted(recorded_ids & flagged_ids, key=cluster_order)
    missing_ids = [cid for cid in all_cluster_ids if cid not in recorded_ids]
    missing_ids += [cid for cid in unreadable_ids if cid not in recorded_ids and cid not in missing_ids]
    cluster_log_dir = run_folder / "ML_logs" / "clusters"
    bravais_mismatch_ids = [
        cid for cid in sorted(recorded_ids, key=cluster_order)
        if _extract_log_bravais_mismatch(cluster_log_dir / f"{cid}.log")
    ]

    n_total = len(matching)
    n_unreadable = len(unreadable_ids)
    n_non_converged = len(not_converged_ids)
    n_converged = len(recorded_ids) - n_non_converged
    n_missing = len(missing_ids)
    n_legacy_only = len(recorded_ids - matching_ids)

    # Write the missing clusters to a standalone, condor-ready file in the same
    # "<S folder>, <cluster id>" format as condor_jobs.txt, so they can be
    # requeued directly with
    #   queue S_DIR, CLUSTER_ID from missing_clusters.txt
    # It lives in clusters_folder, the directory condor_submit is run from, and
    # always lists what is missing now: without missing clusters it is removed.
    missing_clusters_path = clusters_folder / "missing_clusters.txt"
    if missing_ids:
        with open(missing_clusters_path, "w") as mf:
            for cid in missing_ids:
                m = re.match(r"(S\d+)", cid)
                s_dir = m.group(1) if m else cid
                mf.write(f"{s_dir}, {cid}\n")
        print(f"Wrote {n_missing} missing clusters to {missing_clusters_path}")
    else:
        missing_clusters_path.unlink(missing_ok=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        cluster_logs = sorted(cluster_log_dir.glob("S*.log"))
        if cluster_logs:
            header = _extract_log_header(cluster_logs[0])
            if header:
                f.write(f"ML-relaxation settings (from <cluster>.log):\n")
                f.write(header)
                f.write("\n\n")
        f.write(f"ML collect results summary\n")
        f.write(f"Model:           {model or safe_model}\n")
        f.write(f"Run folder:      {run_folder}\n")
        f.write(f"Clusters folder: {clusters_folder}\n")
        f.write(f"Clusters file:   {clusters_path}\n")
        f.write(f"Volume file:     {volume_path}\n")
        f.write(f"\n")
        f.write(f"Total clusters:     {len(all_cluster_ids)}\n")
        f.write(f"Total result files: {n_total}\n")
        if n_legacy_only:
            f.write(
                f"Recorded without per_cluster file (e.g. by run_ml_relaxation): "
                f"{n_legacy_only}\n"
            )
        f.write(f"Missing clusters:   {n_missing}"
                + (f" (incl. {n_unreadable} unreadable result files)" if n_unreadable else "") + "\n")
        f.write(f"Converged:          {n_converged}\n")
        f.write(f"Not converged:      {n_non_converged}\n")
        f.write(f"Unreadable files:   {n_unreadable}\n")
        if bravais_mismatch_ids:
            f.write(
                f"Bravais lattice violations: "
                f"{len(bravais_mismatch_ids)}/{len(recorded_ids)}\n"
            )
        cluster_log_files = sorted(cluster_log_dir.glob("S*.log"))
        if cluster_log_files:
            runtimes = [_extract_log_runtime(p) for p in cluster_log_files]
            runtimes = [r for r in runtimes if r is not None]
            if runtimes:
                f.write(
                    f"\nTotal runtime ({len(runtimes)} clusters): "
                    f"{sum(runtimes):.2f} s\n"
                )
                f.write(
                    f"Average runtime per cluster: "
                    f"{(sum(runtimes)/len(runtimes)):.2f} s\n"
                )
        phonon_report = _phonon_stability_report_lines(run_folder / "phonons")
        if phonon_report:
            f.write("\n")
            for line in phonon_report:
                f.write(line + "\n")
        if missing_ids:
            f.write(f"\nMissing clusters ({n_missing}) written to: {missing_clusters_path}\n")
        if not_converged_ids:
            f.write(f"\nNon-converged clusters: {n_non_converged}\n")
            for cid in not_converged_ids:
                fmax = fmax_by_id.get(cid, float("nan"))
                if cid not in fmax_by_id:
                    fmax_str = "unavailable (no per_cluster file)"
                elif np.isnan(fmax):
                    fmax_str = "unavailable"
                else:
                    fmax_str = f"{fmax:.6g} eV/Å"
                f.write(f"  {cid}  final_fmax: {fmax_str}\n")
        if unreadable_ids:
            f.write(
                f"\nUnreadable result files (empty, header only or malformed), "
                f"requeued via missing_clusters.txt: {n_unreadable}\n"
            )
            for cid in unreadable_ids:
                f.write(f"  {cid}\n")

    return clusters_path, volume_path, log_path


def report_symmetry(atoms, label="", loose_tol=0.02):
    import spglib
    """Detect and report space group using spglib."""
    spglib_cell = (
        atoms.get_cell(),
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    sg_loose = spglib.get_spacegroup(spglib_cell, symprec=loose_tol)
    sg_tight = spglib.get_spacegroup(spglib_cell, symprec=1e-6)
    print(
        f"{label:20s}  loose ({loose_tol}): {str(sg_loose):15s}"
        f"  tight (1e-6): {str(sg_tight)}"
    )
