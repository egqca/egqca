import importlib
from contextlib import nullcontext
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from egqca._ml_utils import (
    MODEL_IDENTITY_NAME,
    RUN_SETTINGS_NAME,
    _check_model_identity,
    _check_run_settings,
    _load_run_settings,
    _ml_model_display_name,
    _model_identity,
    _phonon_settings,
    _run_settings,
)

from egqca.workflow import set_up_clusters_dat

REPO = Path(__file__).resolve().parents[1]
ml = importlib.import_module("egqca.ml")


def _settings(**overrides) -> dict:
    kwargs = dict(
        model="mattersim-5m",
        pressure_GPa=None,
        relaxation_constraint="FixSymmetry",
        CellFilter_mask=None,
        bravais_supercell=(1, 1, 1),
        fix_bravais_positions=False,
        fmax=1e-4,
        optimizer=importlib.import_module("ase.optimize").FIRE,
        optimizer_kwargs=None,
    )
    kwargs.update(overrides)
    return _run_settings(**kwargs)


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


# ------------------------------ run settings ------------------------------ #

@pytest.mark.parametrize("value, expected", [
    (None, None), (True, True), ("none", None), ("Auto", True),
    ("fixsymmetry", "FixSymmetry"), ("FixSymmetry", "FixSymmetry"),
    ("fcc", "FCC"), ("Fcc", "FCC"), ("CUBIC", "cubic"), ("hexagonal", "hexagonal"),
])
def test_relaxation_constraint_is_normalized(value, expected):
    from egqca._ml_utils import _normalize_relaxation_constraint, _relaxation_constraint_arg
    assert _normalize_relaxation_constraint(value) is expected or _normalize_relaxation_constraint(value) == expected
    if isinstance(value, str):
        assert _relaxation_constraint_arg(value) == expected     # the CLI gives the same


@pytest.mark.parametrize("value", ["fcx", False, 1])
def test_relaxation_constraint_rejects_unknown_values(value):
    from egqca._ml_utils import _normalize_relaxation_constraint
    with pytest.raises(ValueError, match="Unknown relaxation_constraint"):
        _normalize_relaxation_constraint(value)


def test_run_settings_match_for_python_and_cli_spelling(tmp_path: Path):
    from egqca._ml_utils import _relaxation_constraint_arg
    _check_run_settings(tmp_path, _settings(relaxation_constraint="fcc"))           # Python API
    _check_run_settings(tmp_path, _settings(relaxation_constraint=_relaxation_constraint_arg("FCC")))  # CLI
    with pytest.raises(ValueError, match="relaxation_constraint: recorded 'FCC', this run 'cubic'"):
        _check_run_settings(tmp_path, _settings(relaxation_constraint="cubic"))


def test_run_settings_accept_an_old_unnormalized_record(tmp_path: Path):
    record = _settings(relaxation_constraint="FCC")
    record["relaxation_constraint"] = "fcc"     # as written before the normalisation
    (tmp_path / RUN_SETTINGS_NAME).write_text(json.dumps(record))
    _check_run_settings(tmp_path, _settings(relaxation_constraint="FCC"))


def test_run_settings_are_recorded_once(tmp_path: Path):
    _check_run_settings(tmp_path, _settings())
    assert json.loads((tmp_path / RUN_SETTINGS_NAME).read_text())["pressure_GPa"] == 0.0
    # bravais_tol only matters for Bravais constraints
    assert "bravais_tol" not in _settings() and "bravais_tol" not in _settings(relaxation_constraint=None)
    assert _settings(relaxation_constraint=True)["bravais_tol"] == 0.01

    # phonons can be added later, but only with one supercell; relaxation-only jobs stay allowed
    assert _load_run_settings(tmp_path)[1] is None
    _check_run_settings(tmp_path, _settings(), _phonon_settings(2))
    _check_run_settings(tmp_path, _settings(), _phonon_settings([2, 2, 2]))
    _check_run_settings(tmp_path, _settings())
    with pytest.raises(ValueError, match="phonon_supercell"):
        _check_run_settings(tmp_path, _settings(), _phonon_settings(1))
    assert _load_run_settings(tmp_path)[1] == {"phonon_supercell": [2, 2, 2]}


@pytest.mark.parametrize("recorded, new, errors", [
    ({}, {"pressure_GPa": 0.0}, []),                                     # None and 0 GPa are the same
    ({}, {"pressure_GPa": 2.0, "relaxation_constraint": None},
     ["pressure_GPa: recorded 0.0, this run 2.0", "relaxation_constraint: recorded 'FixSymmetry', this run None",
      "another out_folder_name / output_dir"]),
    # one model name cannot hide two models
    ({"model": "mace", "model_source": "mace-mp-small"}, {"model": "mace", "model_source": "mace-mp-large"},
     ["model_source: recorded 'mace-mp-small', this run 'mace-mp-large'"]),
    ({"model": "mine", "model_source": ml._ml_model_source(EMT())},
     {"model": "mine", "model_source": ml._ml_model_source(LennardJones())},
     ["recorded 'calculator EMT', this run 'calculator LennardJones'"]),
    ({"relaxation_constraint": "CUB"}, {"relaxation_constraint": "CUB", "bravais_tol": 0.02},
     ["bravais_tol: recorded 0.01, this run 0.02"]),
])
def test_run_settings_refuse_changes(tmp_path: Path, recorded, new, errors):
    _check_run_settings(tmp_path, _settings(**recorded))
    _check_run_settings(tmp_path, _settings(**recorded))          # the same settings continue
    if not errors:
        _check_run_settings(tmp_path, _settings(**new))
        return
    with pytest.raises(ValueError) as exc:
        _check_run_settings(tmp_path, _settings(**new))
    for error in errors:
        assert error in str(exc.value)


def test_old_run_folders_warn_instead_of_failing(tmp_path: Path):
    # results but no record at all
    (tmp_path / "clusters_mattersim-5m.dat").write_text("# header\nS0_i0_w1\t0\t2\t0.0\t1\t-1.0\n")
    with pytest.warns(UserWarning, match="no run_settings.json"):
        _check_run_settings(tmp_path, _settings())

    # a record from before model_source and bravais_tol were recorded
    old = _settings(relaxation_constraint="CUB")
    del old["model_source"], old["bravais_tol"]
    (tmp_path / RUN_SETTINGS_NAME).write_text(json.dumps(old))
    with pytest.warns(UserWarning, match="before (model_source|bravais_tol) was recorded") as record:
        _check_run_settings(tmp_path, _settings(relaxation_constraint="CUB", model_source="mace-mp-small"))
    messages = " ".join(str(w.message) for w in record)
    assert "before model_source was recorded" in messages and "before bravais_tol was recorded" in messages
    # other settings are still checked
    with pytest.raises(ValueError, match="fmax"), pytest.warns(UserWarning):
        _check_run_settings(tmp_path, _settings(relaxation_constraint="CUB", fmax=1e-3))


def test_per_cluster_script_defaults_record_the_same_settings_as_run_ml_relaxation():
    args = _script("ml_relaxation_parallel")._build_parser().parse_args(["cluster", "out"])
    defaults = {name: p.default for name, p in inspect.signature(ml.run_ml_relaxation).parameters.items()}

    from_script = _settings(
        pressure_GPa=args.pressure_gpa,
        relaxation_constraint=args.relaxation_constraint,
        CellFilter_mask=args.cell_filter_mask,
        bravais_supercell=args.bravais_supercell,
        fix_bravais_positions=args.fix_bravais_positions,
        bravais_tol=args.bravais_tol,
        fmax=args.fmax,
        optimizer=args.optimizer,
        optimizer_kwargs=args.optimizer_kwargs,
    )
    from_function = _settings(**{
        key: defaults[key]
        for key in ("pressure_GPa", "relaxation_constraint", "CellFilter_mask", "bravais_supercell",
                    "fix_bravais_positions", "bravais_tol", "fmax", "optimizer", "optimizer_kwargs")
    })
    assert from_script == from_function
    assert _phonon_settings(args.supercell) == _phonon_settings(defaults["N_phonon_supercell"])


@pytest.mark.parametrize("script", ["ml_relaxation_parallel", "ml_relaxation_one_file"])
def test_scripts_take_the_optimizer_and_its_kwargs(script):
    parser = _script(script)._build_parser()
    positional = ["cluster", "out"] if script == "ml_relaxation_parallel" else ["file.cif"]
    args = parser.parse_args([*positional, "--optimizer", "BFGS", "--optimizer-kwargs", '{"maxstep": 0.1}'])
    assert args.optimizer.__name__ == "BFGS"
    assert args.optimizer_kwargs == {"maxstep": 0.1}
    assert parser.parse_args(positional).optimizer.__name__ == "FIRE"     # default
    for bad in ("maxstep=0.1", "[0.1]"):
        with pytest.raises(SystemExit):
            parser.parse_args([*positional, "--optimizer-kwargs", bad])


def test_parallel_jobs_can_continue_a_sequential_run_with_optimizer_kwargs(tmp_path: Path):
    # the sequential run records the kwargs; a parallel job given the same JSON matches them
    _check_run_settings(tmp_path, _settings(optimizer_kwargs={"maxstep": 0.1}))
    args = _script("ml_relaxation_parallel")._build_parser().parse_args(
        ["cluster", "out", "--optimizer-kwargs", '{"maxstep": 0.1}'])
    _check_run_settings(tmp_path, _settings(optimizer_kwargs=args.optimizer_kwargs))
    with pytest.raises(ValueError, match="optimizer_kwargs"):
        _check_run_settings(tmp_path, _settings(optimizer_kwargs=None))


def test_eddp_model_identity(tmp_path: Path):
    def eddp_model(folder: Path, member_content: str) -> str:
        (folder / "ddps").mkdir(parents=True)
        for i in (1, 2):
            (folder / "ddps" / f"m{i}.ddp").write_text(f"{member_content}{i}")
        (folder / "NbV_v3.eddp").write_text("ddps/m1.ddp 0.5\nddps/m2.ddp 0.5\n")
        return f"eddp:{folder / 'NbV_v3.eddp'}"

    assert _ml_model_display_name("eddp:/models/NbV_v3.eddp") == "eddp-NbV_v3"
    assert _model_identity("mattersim-5m") is None
    # the hash covers the ensemble members, not only the manifest
    same, other = _model_identity(eddp_model(tmp_path / "a", "a")), _model_identity(eddp_model(tmp_path / "b", "b"))
    assert same["path"].endswith("NbV_v3.eddp") and same["sha256"] != other["sha256"]

    run = tmp_path / "run"
    run.mkdir()
    _check_model_identity(run, same)                        # the first run records the model
    _check_model_identity(run, same)                        # resume with the same model
    with pytest.raises(ValueError, match="was started with the model"):
        _check_model_identity(run, other)
    assert [p.name for p in run.iterdir()] == [MODEL_IDENTITY_NAME]   # no temporary files left


# ---------------------------- sequential workflow ---------------------------- #

def emt_factory():
    return EMT()


class _FakeRun:
    """Two-cluster tree; relaxations are recorded, not run, and converge only when told so."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.root = tmp_path
        for cid in ("S0_i0_w1", "S1_i0_w1"):
            (tmp_path / cid[:2] / cid).mkdir(parents=True)
            (tmp_path / cid[:2] / cid / f"{cid}.cif").write_text("fake cif")
        # the cluster setup comes from the template; the fake CIFs are never scanned
        set_up_clusters_dat(tmp_path, 1, 1, 1, "Cu")
        self.atoms = bulk("Cu", "fcc", a=3.6)
        self.atoms.calc = EMT()
        self.converged = False
        self.reads, self.relaxations = [], []
        monkeypatch.setattr(ml, "read", self._read)
        monkeypatch.setattr(ml, "geometry_relaxation", self._relax)

    def _read(self, path):
        self.reads.append(Path(path))
        return self.atoms

    def _relax(self, atoms_in, **kwargs):
        self.relaxations.append(kwargs)
        self.atoms.info["egqca_ml_converged"] = self.converged
        self.atoms.info["egqca_ml_final_fmax"] = 0.3
        if kwargs.get("save_cif") is not None:
            Path(kwargs["save_cif"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs["save_cif"]).write_text("relaxed")
        return self.atoms

    def header_warnings(self) -> tuple[int, int]:
        return tuple(
            sum(l.startswith("# WARNING: Relaxation did not converge for some")
                for l in (self.root / "emt" / name).read_text().splitlines())
            for name in ("clusters_emt.dat", "volume_emt.dat")
        )


def test_run_ml_relaxation_continues_only_with_its_settings(tmp_path: Path, monkeypatch):
    run = _FakeRun(tmp_path, monkeypatch)

    # the first run stops after S0, as if the job had been killed
    def relax_and_stop(atoms_in, **kwargs):
        if run.relaxations:
            raise KeyboardInterrupt
        return run._relax(atoms_in, **kwargs)

    monkeypatch.setattr(ml, "geometry_relaxation", relax_and_stop)
    with pytest.raises(KeyboardInterrupt):
        ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt")

    with pytest.raises(ValueError, match="pressure_GPa: recorded 0.0, this run 1.0"):
        ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt", pressure_GPa=1.0)

    # more steps is a continuation of the same run
    monkeypatch.setattr(ml, "geometry_relaxation", run._relax)
    ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt", steps=10)
    assert len(run.relaxations) == 2
    assert not (tmp_path / "emt_1").exists()


PARALLEL_S0 = "# h\nS0_i0_w1\t0\t1\t0.0\t1\t-1.0\t2.5\t2.5\t2.5\t60\t60\t60\t{converged}\t0.2\n"


@pytest.mark.parametrize("result, n_relaxed, expected", [
    (PARALLEL_S0.format(converged="True"), 1, "Imported 1 results from per_cluster/"),
    (PARALLEL_S0.format(converged="False"), 1, "Imported relaxation did not converge: S0_i0_w1. Final fmax: 0.2."),
    ("", 2, None),                         # unreadable: relaxed again
])
def test_sequential_run_imports_results_of_parallel_jobs(tmp_path: Path, monkeypatch, result, n_relaxed, expected):
    run = _FakeRun(tmp_path, monkeypatch)
    run.converged = True
    (tmp_path / "emt" / "per_cluster").mkdir(parents=True)
    # the parallel job records the settings of the run folder, then its result
    defaults = {k: v.default for k, v in inspect.signature(ml.run_ml_relaxation).parameters.items()}
    _check_run_settings(tmp_path / "emt", _settings(
        model="emt", model_source=ml._ml_model_source(EMT()),
        **{k: defaults[k] for k in ("relaxation_constraint", "fmax", "optimizer")},
    ))
    (tmp_path / "emt" / "per_cluster" / "S0_i0_w1_emt_ml_result.dat").write_text(result)

    with pytest.warns(UserWarning, match="treated as missing") if not result else nullcontext():
        ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt")

    assert len(run.relaxations) == n_relaxed
    log = (tmp_path / "emt" / "ML_logs" / "emt.log").read_text()
    s0 = next(l for l in (tmp_path / "emt" / "clusters_emt.dat").read_text().splitlines() if l.startswith("S0"))
    if expected is None:
        assert "Imported" not in log
    else:
        assert expected in log
        assert s0.split("\t")[5] == "-1.0"
        assert ("did not converge" in s0) == ("False" in result)


def test_sequential_rerun_of_non_converged_clusters(tmp_path: Path, monkeypatch):
    run = _FakeRun(tmp_path, monkeypatch)

    # not converged: one header warning per file, not added again on the next run
    ml.run_ml_relaxation(tmp_path, ML_model=emt_factory, ML_model_name="emt")
    assert run.header_warnings() == (1, 1)
    ml.reset_ml_clusters(tmp_path / "emt", "emt")
    ml.run_ml_relaxation(tmp_path, ML_model=emt_factory, ML_model_name="emt")
    assert run.header_warnings() == (1, 1)

    # the rerun continues from the relaxed structures, with the given device and steps
    run.reads.clear(), run.relaxations.clear()
    run.converged = True
    rerun = ml.rerun_nonconverged_ml(tmp_path, emt_factory, ML_model_name="emt", device="cuda", steps=123)
    assert rerun == ["S0_i0_w1", "S1_i0_w1"]
    assert [(r["device"], r["steps"]) for r in run.relaxations] == [("cuda", 123)] * 2
    # (the original CIFs are also read, to infer the cluster setup)
    assert {p.name for p in run.reads if p.parent.name == "cif_relaxed"} == {"S0_i0_w1_emt.cif", "S1_i0_w1_emt.cif"}
    assert run.header_warnings() == (0, 0)
    assert ml.rerun_nonconverged_ml(tmp_path, emt_factory, ML_model_name="emt") == []

    # another model under the same name is refused by the recorded settings
    run.converged = False
    ml.reset_ml_clusters(tmp_path / "emt", "emt", ["S0_i0_w1"])
    ml.run_ml_relaxation(tmp_path, ML_model=emt_factory, ML_model_name="emt")
    with pytest.raises(ValueError, match="model_source"):
        ml.rerun_nonconverged_ml(tmp_path, EMT(), ML_model_name="emt")


def test_rerun_needs_optimizer_kwargs_that_are_not_json(tmp_path: Path, monkeypatch):
    run = _FakeRun(tmp_path, monkeypatch)
    kwargs = {"logfile": tmp_path / "opt.log"}
    with pytest.warns(UserWarning, match="not plain JSON"):
        ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt", optimizer_kwargs=kwargs)

    with pytest.raises(ValueError, match=r"\['logfile'\].*Pass them as optimizer_kwargs"):
        ml.rerun_nonconverged_ml(tmp_path, EMT(), ML_model_name="emt")
    run.relaxations.clear()
    with pytest.warns(UserWarning, match="not plain JSON"):
        ml.rerun_nonconverged_ml(tmp_path, EMT(), ML_model_name="emt", optimizer_kwargs=kwargs)
    assert run.relaxations[0]["optimizer_kwargs"] == kwargs


def test_rerun_script_runs_only_as_main_and_passes_its_options(tmp_path: Path, monkeypatch):
    script = _script("ml_rerun_nonconverged")   # importing must not start a rerun

    calls = []
    monkeypatch.setattr(script, "rerun_nonconverged_ml", lambda *a, **kw: calls.append((a, kw)) or [])
    factory_file = tmp_path / "calcs.py"
    factory_file.write_text("from ase.calculators.emt import EMT\ndef emt():\n    return EMT()\n")
    monkeypatch.setattr(sys, "argv", ["rerun", str(tmp_path), "--model", f"py:{factory_file}:emt",
                                      "--model-name", "emt", "--device", "cuda", "--steps", "7"])
    script.main()

    (folder, model), kw = calls[0]
    assert folder == str(tmp_path) and callable(model) and model.__name__ == "emt"
    assert kw["ML_model_name"] == "emt" and kw["device"] == "cuda" and kw["steps"] == 7
    assert kw["start_from_relaxed"] is True


def test_sequential_phonons_resume(tmp_path: Path, monkeypatch):
    run = _FakeRun(tmp_path, monkeypatch)
    run.converged = True

    class FakeDos:
        def get_energies(self):
            return np.linspace(0.0, 0.05, 100)

        def get_weights(self):
            return np.full(100, 3.0)

    monkeypatch.setattr(ml, "phonons", lambda *a, **kw: FakeDos())
    log = tmp_path / "emt" / "ML_logs" / "clusters" / "S0_i0_w1.log"
    dos = tmp_path / "emt" / "phonons" / "S0_i0_w1.dos"
    ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt")

    # phonons added later; the job is killed while writing the first DOS: no partial file
    savetxt = np.savetxt

    def killed_savetxt(f, data, **kwargs):
        f.write("# Energy(eV) DOS\n0.0 3.0\n")
        raise KeyboardInterrupt

    monkeypatch.setattr(ml.np, "savetxt", killed_savetxt)
    with pytest.raises(KeyboardInterrupt):
        ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt", calc_phonons=True)
    assert not dos.exists() and not list(dos.parent.glob("*.tmp"))

    # the resume computes the phonons and keeps the relaxation's log
    monkeypatch.setattr(ml.np, "savetxt", savetxt)
    ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt", calc_phonons=True)
    assert len(np.loadtxt(dos)) == 100
    text = log.read_text()
    assert text.index("Geometry relaxation for S0_i0_w1") < text.index("Phonons-only resume") < text.index("Phonon calculation")

    # a new relaxation starts a new log
    ml.reset_ml_clusters(tmp_path / "emt", "emt", ["S0_i0_w1"])
    ml.run_ml_relaxation(tmp_path, ML_model=EMT(), ML_model_name="emt", calc_phonons=True)
    text = log.read_text()
    assert text.count("Geometry relaxation for S0_i0_w1") == 1 and "Phonons-only resume" not in text


# ------------------------------- phonon cache ------------------------------- #

def _dos(cache_dir: Path | None, a: float, N: int = 2, atoms=None, calc=None) -> np.ndarray:
    atoms = bulk("Cu", "fcc", a=a) if atoms is None else atoms
    dos = ml.phonons(atoms, calc or EMT(), N, kpts=(4, 4, 4), npts=200, width=2e-3,
                     cache_dir=cache_dir, ML_model_name="emt")
    return np.column_stack((dos.get_energies(), dos.get_weights()))


def _counting_emt() -> EMT:
    """EMT that counts its force evaluations; still an EMT, so the cache sees the same model."""
    calc = EMT()
    calc.calls = 0
    calculate = calc.calculate

    def counting(*args, **kwargs):
        calc.calls += 1
        return calculate(*args, **kwargs)

    calc.calculate = counting
    return calc


def test_phonon_cache_is_reused(tmp_path: Path):
    cache = tmp_path / "cache"
    reference = _dos(cache, a=3.6)

    # a phonons-only resume reads the relaxed CIF back in another orientation: no force evaluations
    rotated = bulk("Cu", "fcc", a=3.6)
    rotated.rotate(30, "z", rotate_cell=True)
    rotated.rotate(20, "x", rotate_cell=True)
    calc = _counting_emt()
    np.testing.assert_allclose(_dos(cache, a=3.6, atoms=rotated, calc=calc), reference)
    assert calc.calls == 0

    # a killed job leaves the entry of the displacement it was working on empty
    entry = cache / "forces" / "cache.0x-.json"
    assert entry.stat().st_size > 0
    entry.write_text("")
    np.testing.assert_allclose(_dos(cache, a=3.6), reference)


def test_phonon_cache_of_anything_else_is_discarded(tmp_path: Path):
    cache = tmp_path / "cache"
    _dos(cache, a=3.6)
    atoms = bulk("Cu", "fcc", a=3.6)
    matches = lambda a: ml._phonon_cache_matches(cache, a, (2, 2, 2), 0.01, ml._ml_model_description(EMT(), "emt"))
    assert matches(atoms)
    strained = atoms.copy()
    strained.set_cell(atoms.cell.array * [1.0, 1.0, 1.001], scale_atoms=True)
    mirrored = atoms.copy()
    mirrored.set_cell(atoms.cell.array @ np.diag([1.0, 1.0, -1.0]), scale_atoms=True)
    assert not matches(strained) and not matches(mirrored)

    # another structure or supercell, or metadata truncated by a killed job: the forces are recomputed
    np.testing.assert_allclose(_dos(cache, a=3.4), _dos(None, a=3.4))
    np.testing.assert_allclose(_dos(cache, a=3.4, N=3), _dos(None, a=3.4, N=3))
    info = cache / "phonon_info.json"
    info.write_text(info.read_text()[:20])
    np.testing.assert_allclose(_dos(cache, a=3.4, N=3), _dos(None, a=3.4, N=3))


# ----------------------------- parallel workflow ----------------------------- #

def test_per_cluster_script_errors(tmp_path: Path, monkeypatch, capsys):
    script = _script("ml_relaxation_parallel")
    seen = []

    def fake_run_one_cluster(args):
        seen.append(args.cluster_dir)
        if args.cluster_dir == "S1/S1_i1_w2":
            raise RuntimeError("relaxation failed")

    monkeypatch.setattr(script, "_run_one_cluster", fake_run_one_cluster)
    args = script._build_parser().parse_args(["S1/S1_i0_w1", "S1/S1_i1_w2", "S2/S2_i0_w1", "run"])
    # a failing cluster does not stop the others; the script exits with an error listing it
    assert script._run_clusters(args) == ["S1/S1_i1_w2"]
    assert seen == ["S1/S1_i0_w1", "S1/S1_i1_w2", "S2/S2_i0_w1"]
    monkeypatch.setattr(sys, "argv", ["ml_relaxation_parallel.py", *args.cluster_dirs, "run"])
    with pytest.raises(SystemExit) as exit_info:
        script.main()
    assert exit_info.value.code == 1
    assert "S1/S1_i1_w2" in capsys.readouterr().err
    # a single cluster raises its own error
    with pytest.raises(RuntimeError, match="relaxation failed"):
        script._run_clusters(script._build_parser().parse_args(["S1/S1_i1_w2", "run"]))

    # a calculator factory needs a model name, and a py: spec needs a function
    monkeypatch.undo()
    script = _script("ml_relaxation_parallel")
    factory_file = tmp_path / "my_calc.py"
    factory_file.write_text("from ase.calculators.emt import EMT\ndef make_calculator():\n    return EMT()\n")
    cluster = tmp_path / "S0" / "S0_i0_w1"
    cluster.mkdir(parents=True)
    bulk("Cu", "fcc", a=3.7, cubic=True).write(cluster / "S0_i0_w1.cif")
    args = script._build_parser().parse_args([str(cluster), str(tmp_path / "run"),
                                              "--model", f"py:{factory_file}:make_calculator"])
    with pytest.raises(ValueError, match="ML_model_name is required"):
        script._run_clusters(args)
    with pytest.raises(SystemExit, match="expected py:"):
        script._resolve_model("py:no_function_given")


def test_parallel_workflow_lifecycle(tmp_path: Path, capsys):
    from egqca.ml import clear_ml_calculator_cache, collect_ml_results, reset_ml_clusters

    clear_ml_calculator_cache()
    clusters = tmp_path / "clusters"
    for cluster_id in ("S0_i0_w1", "S1_i0_w1"):
        cluster = clusters / cluster_id.split("_")[0] / cluster_id
        cluster.mkdir(parents=True)
        atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
        if cluster_id.startswith("S1"):
            atoms.symbols[0] = "Au"
        atoms.write(cluster / f"{cluster_id}.cif")
    calls = tmp_path / "factory_calls.txt"
    factory_file = tmp_path / "my_calc.py"
    factory_file.write_text(
        "from ase.calculators.emt import EMT\n"
        "def make_calculator():\n"
        f"    with open({str(calls)!r}, 'a') as f:\n"
        "        f.write('call\\n')\n"
        "    return EMT()\n"
    )
    script = _script("ml_relaxation_parallel")
    run_folder = clusters / "emt-custom"
    common = [str(clusters / "S0" / "S0_i0_w1"), str(clusters / "S1" / "S1_i0_w1"), str(run_folder),
              "--model", f"py:{factory_file}:make_calculator", "--model-name", "emt-custom",
              "--relaxation-constraint", "none", "--fmax", "0.05"]
    relax = lambda *extra: script._run_clusters(script._build_parser().parse_args([*common, *extra]))

    # 1. one step: not converged. ML_model_name names the files and the run settings;
    #    the factory is called once for both clusters
    assert relax("--steps", "1") == []
    for cluster_id in ("S0_i0_w1", "S1_i0_w1"):
        assert (run_folder / "per_cluster" / f"{cluster_id}_emt-custom_ml_result.dat").is_file()
    assert json.loads((run_folder / "run_settings.json").read_text())["model"] == "emt-custom"
    assert calls.read_text().count("call") == 1

    # 2. reset and continue from the last structure
    assert "S1_i0_w1" in reset_ml_clusters(run_folder, "emt-custom")
    capsys.readouterr()
    assert relax("--steps", "200", "--start-from-relaxed") == []
    assert "starting from the relaxed structure" in capsys.readouterr().out
    assert (run_folder / "per_cluster" / "S1_i0_w1_emt-custom_ml_result.dat").is_file()

    # 3. phonons added later: the relaxation's log is kept
    log = run_folder / "ML_logs" / "clusters" / "S1_i0_w1.log"
    with open(log, "a") as f:   # stands in for a Bravais check that failed during the relaxation
        f.write("WARNING: Final Bravais lattice 'ORC' is not compatible with the requested Bravais type 'TET'.\n")
    assert relax("--steps", "200", "--phonons", "--supercell", "1") == []
    text = log.read_text()
    assert "After relaxation" in text
    assert text.index("WARNING: Final Bravais lattice") < text.index("Phonons-only resume") < text.index("Phonon calculation")

    # 4. collect counts the Bravais violation and the runtime of both passes
    runtimes = [float(l.split()[1]) for p in log.parent.glob("S*.log")
                for l in p.read_text().splitlines() if l.startswith("Runtime:")]
    assert len(runtimes) == 4
    _, _, summary = collect_ml_results(run_folder, clusters_folder=clusters, model="emt-custom")
    summary = summary.read_text()
    assert "Bravais lattice violations: 1/2" in summary
    assert f"Total runtime (2 clusters): {sum(runtimes):.2f} s" in summary
    clear_ml_calculator_cache()
