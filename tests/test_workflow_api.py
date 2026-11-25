import importlib
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

import egqca
from egqca.constants import Ry_to_mev
from egqca.workflow import (
    _continuous_cluster_names,
    _supercell_runs,
    get_directories,
    get_elements,
    run_supercell,
)

ml = importlib.import_module("egqca.ml")
ml_utils = importlib.import_module("egqca._ml_utils")


def test_public_api_has_no_mutable_defaults():
    mutable = [
        f"{name}({parameter.name}=...)"
        for name in egqca.__all__
        if callable(getattr(egqca, name)) and not isinstance(getattr(egqca, name), type(egqca))
        for parameter in inspect.signature(getattr(egqca, name)).parameters.values()
        if isinstance(parameter.default, (list, dict, set))
    ]
    assert mutable == []


# ----------------------------- Bravais constraint ----------------------------- #

def _cu(cellpar) -> Atoms:
    return Atoms("Cu", scaled_positions=[(0.0, 0.0, 0.0)], cell=cellpar, pbc=True)


def _projected_cell(constraint, atoms):
    cell = np.array(atoms.cell.array, dtype=float)
    constraint.adjust_cell(atoms, cell)
    return cell


@pytest.mark.parametrize("cellpar, supercell, requested, bravais_tol, expected, n_ops", [
    (np.diag([4.0, 4.01, 3.995]), (1, 1, 1), [True], 0.01, "CUB", 48),       # nearly cubic: snapped
    (np.diag([4.0, 4.06, 4.0]), (1, 1, 1), [True], 0.01, "TET", 16),         # 1.5 % off: not at 1 %
    (np.diag([4.0, 4.06, 4.0]), (1, 1, 1), [True, "CUB"], 0.02, "CUB", 48),  # ... but at 2 %
    (np.diag([7.2, 7.25, 3.6]), (2, 2, 1), [True], 0.01, "CUB", 48),         # primitive lattice of a supercell
    # the unique axis may lie along c, b or a
    ([4.0, 4.0, 5.0, 90, 90, 90], (1, 1, 1), [True, "TET"], 0.01, "TET", 16),
    ([4.0, 5.0, 4.0, 90, 90, 90], (1, 1, 1), [True, "TET"], 0.01, "TET", 16),
    ([5.0, 4.0, 4.0, 90, 90, 90], (1, 1, 1), [True, "tetragonal"], 0.01, "TET", 16),
    ([5.0, 3.0, 3.0, 120, 90, 90], (1, 1, 1), [True, "HEX"], 0.01, "HEX", 24),
    ([3.0, 5.0, 3.0, 90, 120, 90], (1, 1, 1), [True, "hexagonal"], 0.01, "HEX", 24),
    ([5.0, 3.0, 4.0, 90, 90, 90], (1, 1, 1), [True, "ORC"], 0.01, "ORC", 8),
])
def test_bravais_constraint_detects_the_lattice(cellpar, supercell, requested, bravais_tol, expected, n_ops):
    atoms = _cu(cellpar)
    for target in requested:
        constraint = ml_utils._bravais_cell_constraint(target, atoms, supercell=supercell, bravais_tol=bravais_tol)
        assert constraint.bravais_lattice == expected
        assert len(constraint._rotations_prim) == n_ops
        # the projected cell is exactly of the detected lattice
        projected = Atoms(cell=_projected_cell(constraint, atoms), pbc=True)
        assert ml_utils._bravais_lattice_name(projected, eps=1e-8, supercell=supercell) == expected


@pytest.mark.parametrize("cellpar, supercell, target", [
    ([3.6, 3.6, 3.6, 90, 90, 90], (1, 1, 1), "CUB"),
    ([3.6, 3.6, 3.6, 90, 90, 90], (1, 1, 1), "FCC"),   # conventional FCC is detected as CUB
    ([2.5, 2.5, 2.5, 60, 60, 60], (1, 1, 1), "cubic"),  # primitive FCC
    ([7.2, 7.2, 3.6, 90, 90, 90], (2, 2, 1), "CUB"),   # 2x2x1 supercell of a cubic cell
    ([3.0, 3.0, 5.0, 90, 90, 120], (1, 1, 1), "HEX"),
    ([3.0, 3.0, 3.0, 70, 70, 70], (1, 1, 1), "RHL"),
    ([3.6, 3.61, 3.6, 90, 90, 90], (1, 1, 1), "CUB"),  # nearly ideal: projected onto CUB
    ([1.0, 1.0, 2.0, 90, 90, 90], (1, 1, 1), True),    # auto
])
def test_bravais_constraint_keeps_the_start_lattice(cellpar, supercell, target):
    atoms = _cu(cellpar)
    start = ml_utils._bravais_lattice_name(atoms, supercell=supercell)

    atoms.set_constraint(ml_utils._bravais_cell_constraint(target, atoms, supercell=supercell))
    atoms.set_cell(atoms.cell.array @ np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 1.03]]),
                   scale_atoms=True)

    assert ml_utils._bravais_lattice_name(atoms, supercell=supercell) == start


@pytest.mark.parametrize("cellpar, target, kwargs, error", [
    ([3.5, 3.6, 3.75, 90, 90, 90], "CUB", {}, "does not match the starting cell"),   # distorted cubic
    ([3.5, 3.6, 3.75, 90, 90, 90], "cubic", {}, "does not match the starting cell"),
    ([3.0, 3.0, 5.0, 90, 90, 120], "RHL", {}, "does not match the starting cell"),   # same family, other lattice
    ([3.0, 3.0, 3.0, 70, 70, 70], "HEX", {}, "does not match the starting cell"),
    (np.diag([4.0, 4.06, 4.0]), "CUB", {}, "bravais_tol=0.01"),                      # the error names the tolerance
    (np.diag([4.0, 4.06, 4.0]), True, {"bravais_tol": 0.0}, "bravais_tol must be positive"),
])
def test_bravais_constraint_rejects_a_start_cell_without_the_target_symmetry(cellpar, target, kwargs, error):
    with pytest.raises(ValueError, match=error):
        ml_utils._bravais_cell_constraint(target, _cu(cellpar), **kwargs)


def test_fix_bravais_constraint_does_not_symmetrize_atomic_forces():
    forces = np.array([[1.0, 2.0, 3.0]])
    atoms = _cu([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=forces, stress=np.zeros(6))

    # fix_positions=False constrains only the cell metric, never the atomic forces
    atoms.set_constraint(ml_utils._bravais_cell_constraint(True, atoms, fix_positions=False))

    np.testing.assert_allclose(atoms.get_forces(), forces)


@pytest.mark.parametrize("kwargs, frozen", [({}, False), ({"fix_positions": True}, True)])
def test_fix_bravais_positions_freezes_positions_only_on_request(kwargs, frozen):
    atoms = Atoms("Cu2", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=np.diag([3.0, 3.0, 3.0]), pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.ones((2, 3)), stress=np.zeros(6))
    atoms.set_constraint(ml_utils._bravais_cell_constraint(True, atoms, **kwargs))

    # frozen: no force drives the atoms and a move is undone; otherwise both pass through
    np.testing.assert_allclose(atoms.get_forces(), np.zeros((2, 3)) if frozen else np.ones((2, 3)))
    start = atoms.positions.copy()
    atoms.set_positions(start + [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(atoms.positions[0], [0.0, 0.0, 0.0] if frozen else [0.1, 0.0, 0.0], atol=1e-12)


def test_geometry_relaxation_auto_constraint_keeps_a_nearly_cubic_alloy_cubic():
    # L1_0 CuAu prefers a tetragonal cell; the 0.3 % strained start cell is cubic within bravais_tol
    a = 3.9
    atoms = Atoms(
        "Cu2Au2",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=np.diag([a, a * 1.003, a * 0.9985]),
        pbc=True,
    )

    relaxed = ml.geometry_relaxation(atoms, ML_model=EMT(), relaxation_constraint=True, fmax=1e-3, steps=400)

    assert relaxed.info["egqca_ml_initial_bravais_lattice"] == "CUB"
    assert relaxed.info["egqca_ml_final_bravais_lattice"] == "CUB"
    assert "egqca_ml_bravais_mismatch" not in relaxed.info
    np.testing.assert_allclose(relaxed.cell.lengths(), relaxed.cell.lengths()[0], rtol=1e-8)


def test_geometry_relaxation_reports_when_the_bravais_lattice_changes(monkeypatch):
    class FakeCellFilter:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms

        def get_forces(self):
            return [[0.0, 0.0, 0.0]]

    class ChangeBravaisOptimizer:
        def __init__(self, cell_filter, **kwargs):
            self.atoms = cell_filter.atoms

        def run(self, *, fmax, steps):
            self.atoms.set_constraint()
            self.atoms.set_cell([1.0, 1.2, 2.0, 90.0, 90.0, 90.0], scale_atoms=True)
            return True

    monkeypatch.setattr(ml, "_ml_calculator", lambda *args, **kwargs: object())
    monkeypatch.setattr(ml, "FrechetCellFilter", FakeCellFilter)

    relaxed = ml.geometry_relaxation(
        _cu([1.0, 1.0, 2.0, 90.0, 90.0, 90.0]), ML_model="fake", relaxation_constraint="TET",
        optimizer=ChangeBravaisOptimizer,
    )

    # the run is not aborted, the mismatch is recorded and collected afterwards
    assert relaxed.info["egqca_ml_final_bravais_lattice"] == "ORC"
    assert "TET" in relaxed.info["egqca_ml_bravais_mismatch"]
    assert "ORC" in relaxed.info["egqca_ml_bravais_mismatch"]


# ------------------------------ ML calculators ------------------------------ #

@pytest.mark.parametrize("model, expected_model, expected_head", [
    ("mace", "medium-mpa-0", None),
    ("mace-small", "small", None),
    ("mace-mh-1", "mh-1", "matpes_r2scan"),
])
def test_mace_calculator(monkeypatch, model, expected_model, expected_head):
    seen = {}

    class FakeTorch:
        float32, float64 = "float32", "float64"
        default_dtype = "float32"

        def get_default_dtype(self):
            return self.default_dtype

        def set_default_dtype(self, dtype):
            self.default_dtype = dtype

    fake_torch = FakeTorch()

    class FakeCalculator:
        def calculate(self, *args, **kwargs):
            seen["dtype_during_calculation"] = fake_torch.get_default_dtype()
            return "result"

    def fake_mace_mp(**kwargs):
        seen.update(kwargs, dtype_during_construction=fake_torch.get_default_dtype())
        return FakeCalculator()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(ml, "_mace_calculator", lambda: types.SimpleNamespace(mace_mp=fake_mace_mp))

    calculator = ml._ml_calculator(model, device="cpu", reuse=False)
    assert (seen["model"], seen.get("head")) == (expected_model, expected_head)
    # float64 while the model is built and while it calculates, but not for the
    # rest of the process: other models loaded later keep the default dtype
    assert seen["dtype_during_construction"] == seen["default_dtype"] == "float64"
    assert fake_torch.get_default_dtype() == "float32"
    assert calculator.calculate() == "result"
    assert seen["dtype_during_calculation"] == "float64"
    assert fake_torch.get_default_dtype() == "float32"


class _FakeTensorflowConfig:
    def __init__(self, gpus):
        self.gpus = list(gpus)
        self.visible_gpus = list(gpus)

    def list_physical_devices(self, kind):
        return list(self.gpus)

    def get_visible_devices(self, kind):
        return list(self.visible_gpus)

    def set_visible_devices(self, devices, kind):
        self.visible_gpus = list(devices)


@pytest.mark.parametrize("model, device, expected_model, visible", [
    ("GRACE-3L-OMAT-large-ft-AM", "cpu", "GRACE-3L-OMAT-large-ft-AM", []),       # cpu hides the GPUs
    ("grace-3l-omat-large-ft-am", "cuda:1", "GRACE-3L-OMAT-large-ft-AM", ["GPU:1"]),   # any case
    ("GRACE-2L-OAM-L", "cuda:1", "GRACE-2L-OMAT-large-ft-AM", ["GPU:1"]),          # alias
])
def test_grace_calculator(monkeypatch, model, device, expected_model, visible):
    seen = {}

    def fake_grace_fm(model, **kwargs):
        seen.update(model=model, kwargs=kwargs)
        return "calculator"

    foundation_models = types.SimpleNamespace(
        MODELS_NAME_LIST=["GRACE-2L-OMAT-large-ft-AM", "GRACE-3L-OMAT-large-ft-AM"],
        MODELS_ALIASES_DICT={"GRACE-2L-OAM-L": "GRACE-2L-OMAT-large-ft-AM"},
        grace_fm=fake_grace_fm,
    )
    tf_config = _FakeTensorflowConfig(["GPU:0", "GPU:1"])
    monkeypatch.setitem(sys.modules, "tensorpotential.calculator.foundation_models", foundation_models)
    monkeypatch.setitem(sys.modules, "tensorflow", types.SimpleNamespace(config=tf_config))

    assert ml._ml_calculator(model, device=device, reuse=False) == "calculator"
    assert seen["model"] == expected_model
    assert seen["kwargs"] == {"mode": "uniform", "enable_uq_if_available": False}
    assert tf_config.visible_gpus == visible


# ----------------------------- run_ml_relaxation ----------------------------- #

def test_run_ml_relaxation_with_a_user_calculator(tmp_path: Path, monkeypatch):
    for cluster_id in ("S0_i0_w1", "S1_i0_w1"):
        (tmp_path / cluster_id[:2] / cluster_id).mkdir(parents=True)
        (tmp_path / cluster_id[:2] / cluster_id / f"{cluster_id}.cif").write_text("fake cif")
    # the cluster setup comes from the template; the fake CIFs are never scanned
    egqca.workflow.set_up_clusters_dat(tmp_path, 1, 2, 2, "MgB")
    calculator = EMT()
    seen = []

    class FakeCell:
        def lengths(self):
            return [1.0, 2.0, 3.0]

        def angles(self):
            return [90.0, 90.0, 90.0]

    class FakeAtoms:
        cell = FakeCell()
        info = {}

        def __len__(self):
            return 2

        def copy(self):
            return self

        def get_chemical_symbols(self):
            return ["Mg", "B"]

        def get_potential_energy(self):
            return 13.6

    def fake_geometry_relaxation(atoms, **kwargs):
        seen.append(kwargs)
        print("current attempt")
        return atoms

    monkeypatch.setattr(ml, "read", lambda path: FakeAtoms())
    monkeypatch.setattr(ml, "geometry_relaxation", fake_geometry_relaxation)
    run_folder = tmp_path / "my_model"
    # the log of an earlier failed attempt is replaced
    stale_log = run_folder / "ML_logs" / "clusters" / "S1_i0_w1.log"
    stale_log.parent.mkdir(parents=True)
    stale_log.write_text("previous failed attempt\n")

    ml.run_ml_relaxation(tmp_path, ML_model=calculator, ML_model_name="my/model")

    # names come from ML_model_name, made safe for the file system
    assert seen[0]["ML_model"] is calculator
    assert seen[1]["save_cif"] == run_folder / "cif_relaxed" / "S1_i0_w1_my_model.cif"
    clusters = (run_folder / "clusters_my_model.dat").read_text()
    assert "my/model (user-supplied calculator: EMT)" in clusters
    row = next(l for l in clusters.splitlines() if l.startswith("S1_i0_w1")).split("\t")
    assert (row[1], row[2]) == ("1", "2")
    assert float(row[5]) == pytest.approx(13.6 / (Ry_to_mev / 1000))
    assert "S1_i0_w1\t1.0\t2.0\t3.0\t90.0\t90.0\t90.0" in (run_folder / "volume_my_model.dat").read_text()
    global_log = run_folder / "ML_logs" / "my_model.log"
    for line in ("model: my/model (user-supplied calculator: EMT)", "device: set by the user-supplied calculator",
                 "relaxed structure filename: cif_relaxed/<cluster_id>_my_model.cif"):
        assert line in global_log.read_text()
    log_text = stale_log.read_text()
    assert "previous failed attempt" not in log_text
    assert log_text.count("current attempt") == 1 and "Runtime:" in log_text

    # a new cluster: the incomplete folder is resumed and its global log appended to
    (tmp_path / "S2" / "S2_i0_w1").mkdir(parents=True)
    (tmp_path / "S2" / "S2_i0_w1" / "S2_i0_w1.cif").write_text("fake cif")
    ml.run_ml_relaxation(tmp_path, ML_model=calculator, ML_model_name="my/model")
    assert len(seen) == 3
    assert global_log.read_text().count("Starting ML relaxation run") == 2

    # a complete folder is kept: no new folder, nothing relaxed again, its results returned
    energies, latpar = ml.run_ml_relaxation(tmp_path, ML_model=calculator, ML_model_name="my/model")
    assert not (tmp_path / "my_model_1").exists()
    assert len(seen) == 3
    assert sorted(energies) == ["S0_i0_w1", "S1_i0_w1", "S2_i0_w1"]
    assert energies["S1_i0_w1"] == pytest.approx(13.6 / (Ry_to_mev / 1000))
    np.testing.assert_allclose(latpar["S2_i0_w1"], [1.0, 2.0, 3.0, 90.0, 90.0, 90.0])


def test_ml_output_files_resume_when_either_file_is_incomplete(tmp_path: Path):
    # killed between the clusters and the volume row
    clusters_file = tmp_path / "clusters_model.dat"
    clusters_file.write_text("# header\nS0_i0_w1\t0\t2\t0.0\t1\t-1.0\nS1_i0_w2\t1\t2\t0.5\t2\t-2.0\n")
    (tmp_path / "volume_model.dat").write_text("# header\nS0_i0_w1\t1\t1\t1\t90\t90\t90\n")

    names = ml._prepare_ml_output_files(
        tmp_path, "clusters_model.dat", "volume_model.dat", target_cluster_ids={"S0_i0_w1", "S1_i0_w2"},
        permutable_sites=1, sites=2, compound="A", ML_model="model",
    )

    assert names == ("clusters_model.dat", "volume_model.dat", True)
    assert clusters_file.read_text().count("S1_i0_w2") == 1


def test_global_ml_error_log_records_exception_and_traceback(tmp_path: Path):
    log_file = tmp_path / "ML_logs" / "mace.log"
    log_file.parent.mkdir()

    try:
        raise RuntimeError("model download failed")
    except RuntimeError as exc:
        ml._write_global_ml_error(log_file, "S2_i03_w4", exc)

    log_text = log_file.read_text()
    assert "ERROR during ML relaxation for S2_i03_w4." in log_text
    assert log_text.count("RuntimeError: model download failed") == 1
    assert "Traceback:" in log_text
    assert "Stopping ML relaxation run." in log_text


def test_original_cluster_cif_is_never_a_relaxed_structure(tmp_path: Path):
    cluster_dir = tmp_path / "S2_i03_w4"
    cluster_dir.mkdir()
    (cluster_dir / "S2_i03_w4_mace.cif").write_text("relaxed")
    assert ml._original_cluster_cif(cluster_dir) is None
    (cluster_dir / "S2_other_name.cif").write_text("original")
    assert ml._original_cluster_cif(cluster_dir) == cluster_dir / "S2_other_name.cif"
    (cluster_dir / "S2_i03_w4.cif").write_text("original")
    assert ml._original_cluster_cif(cluster_dir) == cluster_dir / "S2_i03_w4.cif"


def _p1_cif(path: Path, symbols: list[str]) -> None:
    rows = [
        f"  {symbol}  {symbol}{index}  {((index + 1) % 13) / 13:.8f}  "
        f"{((index + 3) % 13) / 13:.8f}  {((index + 5) % 13) / 13:.8f}  1"
        for index, symbol in enumerate(symbols)
    ]
    path.write_text("\n".join([
        "data_cluster", "_cell_length_a 10", "_cell_length_b 10", "_cell_length_c 10",
        "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'P 1'", "loop_", "_space_group_symop_operation_xyz", "'x, y, z'",
        "loop_", " _atom_site_type_symbol", " _atom_site_label", " _atom_site_fract_x", " _atom_site_fract_y",
        " _atom_site_fract_z", " _atom_site_occupancy", *rows,
    ]))


def test_infer_cluster_setup_extracts_substitution_compound_from_cifs(tmp_path: Path):
    for cluster_id, symbols in (("S0_i0_w1", ["Mg"] * 4 + ["B"] * 8), ("S8_i0_w1", ["Mg"] * 4 + ["C"] * 8)):
        (tmp_path / cluster_id[:2] / cluster_id).mkdir(parents=True)
        _p1_cif(tmp_path / cluster_id[:2] / cluster_id / f"{cluster_id}.cif", symbols)

    permutable_sites, sites, Nat, compound, cluster_ids = ml._infer_cluster_setup(tmp_path, 8)

    assert (permutable_sites, sites, Nat) == (8, 12, 12)
    assert compound == "(B$_{1-x}$C$_{x}$)$_{2}$Mg"     # x in [0, 1], permutable group first
    assert sorted(cluster_ids) == ["S0_i0_w1", "S8_i0_w1"]


def _header_only_tree(tmp_path: Path, n: int | None, where: str = "DFT/clusters.dat") -> Path:
    """Cluster tree with S1 and S2 folders (a sub-range) and, for n, a clusters file with that n."""
    for cluster_id in ("S1_i0_w4", "S2_i0_w6"):
        (tmp_path / cluster_id[:2] / cluster_id).mkdir(parents=True)
    if n is not None:
        (tmp_path / where).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / where).write_text(
            "# EGQCA input for compound=A$_{1-x}$B$_x$\n# total number of sites=8\n"
            f"# permutable sites n={n}\n#ID\t\tn_j\tNat\tx\tg_j\tE_j (Ry)\n"
            "S1_i0_w4\t1\t8\t0.125\t4\t-1.5    # WARNING: DFT run did not finish\n"
        )
    return tmp_path


def test_cluster_setup_comes_from_the_template_not_the_folders(tmp_path: Path):
    # only S1 and S2 exist: the folders would suggest n=2, the template says 8
    tree = _header_only_tree(tmp_path, 8, "clusters_template.dat")
    assert ml._resolve_permutable_sites(tree) == (8, tree / "clusters_template.dat")
    assert ml._resolve_permutable_sites(tree, 8)[0] == 8     # a matching explicit value is fine
    n, sites, Nat, compound, ids = ml._resolve_cluster_setup(tree)
    assert (n, sites, Nat, ids) == (8, 8, 8, ["S1_i0_w4"])


def test_old_tree_writes_the_template_from_dft_clusters_dat(tmp_path: Path):
    tree = _header_only_tree(tmp_path, 8)
    dft_before = (tree / "DFT" / "clusters.dat").read_text()

    assert ml._resolve_permutable_sites(tree) == (8, tree / "clusters_template.dat")

    template = (tree / "clusters_template.dat").read_text()
    assert "# permutable sites n=8" in template
    assert "S1_i0_w4\t1\t8\t0.125\t4\tmissing\n" in template     # energy and warning dropped
    assert "WARNING" not in template
    assert (tree / "DFT" / "clusters.dat").read_text() == dft_before   # never changed


@pytest.mark.parametrize("name, value", [("permutable_sites", 2), ("sites", 9), ("Nat", 7), ("compound", "AB")])
def test_explicit_values_conflicting_with_the_template_raise(tmp_path: Path, name, value):
    tree = _header_only_tree(tmp_path, 8, "clusters_template.dat")
    with pytest.raises(ValueError, match=f"{name}={value!r} differs from"):
        ml._resolve_cluster_setup(tree, **{name: value})


def test_without_template_n_comes_from_the_folders_and_the_template_is_written(tmp_path: Path, monkeypatch):
    tree = _header_only_tree(tmp_path, None)
    assert ml._resolve_permutable_sites(tree) == (2, None)       # highest S* folder
    assert ml._resolve_permutable_sites(tree, 8) == (8, None)    # an explicit value wins

    import egqca.workflow as workflow
    monkeypatch.setattr(workflow, "_infer_cluster_setup",
                        lambda folder, n: (n, 8, 8, "A$_{1-x}$B$_x$", ["S1_i0_w4"]))
    assert ml._resolve_cluster_setup(tree, permutable_sites=8)[:4] == (8, 8, 8, "A$_{1-x}$B$_x$")
    assert "# permutable sites n=8" in (tree / "clusters_template.dat").read_text()


def test_clusters_template(alloy, tmp_path):
    # the synthetic clusters.dat separates its columns by spaces, not tabs
    assert ml._parse_clusters_template(alloy[0]) == (6, 9, 9, "A$_{2-x}$B$_x$")
    ids = ml._cluster_ids_from_template(alloy[0])
    assert ids[:3] == ["S0_i0_w1", "S1_i0_w6", "S2_i0_w6"]
    assert all(" " not in cid for cid in ids)

    # a space after the equal sign
    template = tmp_path / "clusters.dat"
    template.write_text(
        "# compound= A$_{1-x}$B$_x$C\n# total number of sites= 12\n# permutable sites n= 8\n"
        "S0_i0_w1\t0\t12\t0.0\t1\tmissing\n"
    )
    assert ml._parse_clusters_template(template) == (8, 12, 12, "A$_{1-x}$B$_x$C")
    assert ml._cluster_ids_from_template(template) == ["S0_i0_w1"]


# --------------------------- structure and supercell --------------------------- #

GAN_ROWS = [
    "  Ga3+  Ga0  {m}  0.00000000  0.00000000  0.00000000  1",
    "  N3-   N1   {m}  0.25000000  0.25000000  0.75000000  1",
    "  Al3+  Al0  {m}  0.00000000  0.00000000  0.00000000  0.",   # not next to Ga0
]
BN_ROWS = [
    "  B   B0  1  0.00000000  0.00000000  0.00000000  1",
    "  C   C0  1  0.00000000  0.00000000  0.00000000  0",
    "  N   N0  1  0.33333333  0.66666667  0.00000000  1",
    "  C   C1  1  0.33333333  0.66666667  0.00000000  0",
]
GAN = (["Ga", "N", "Al"], ["Ga0", "Al0"], 8, 4, "Ga$_{1-x}$Al$_{x}$N")      # Ga4N4 reduced


@pytest.mark.parametrize("rows, supercell, expected", [
    ([r.format(m=4) for r in GAN_ROWS], "1x1x1", GAN),          # non-consecutive permutable rows
    ([r.format(m="4.0") for r in GAN_ROWS], "1x1x1", GAN),      # float-like multiplicity
    (BN_ROWS, "4x4x1", (["B", "C", "N", "C"],
                        {"mode": "coupled", "original": ["B0", "N0"], "substitution": ["C0", "C1"],
                         "sites_per_label": [16, 16]},
                        32, 32, "(BN)$_{1-x}$C$_{2x}$")),        # two coupled permutable sites
])
def test_get_elements(tmp_path: Path, rows, supercell, expected):
    cif_file = tmp_path / "structure.cif"
    cif_file.write_text("\n".join([
        "data_test", "loop_", " _atom_site_type_symbol", " _atom_site_label", " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy", *rows,
    ]))
    assert get_elements(cif_file, supercell) == expected


def test_get_elements_takes_the_element_from_the_type_symbol(tmp_path: Path):
    # labels that do not name the element: 'B1a' would read as Ba, 'N1x' as nothing
    cif_file = tmp_path / "structure.cif"
    cif_file.write_text("\n".join([
        "data_test", "loop_", " _atom_site_type_symbol", " _atom_site_label", " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy",
        "  B   B1a  1  0.00000000  0.00000000  0.00000000  1",
        "  C   C1b  1  0.00000000  0.00000000  0.00000000  0",
        "  N3-  N1x  1  0.33333333  0.66666667  0.00000000  1",
    ]))
    elements, labels, sites, n, _ = get_elements(cif_file, "1x1x1")
    assert elements == ["B", "C", "N"]
    assert labels == ["B1a", "C1b"]
    assert (sites, n) == (2, 1)


def _cif_with_rows(tmp_path: Path, *rows: str) -> Path:
    cif_file = tmp_path / "structure.cif"
    cif_file.write_text("\n".join([
        "data_test", "loop_", " _atom_site_type_symbol", " _atom_site_label", " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy", *rows,
    ]) + "\n")
    return cif_file


@pytest.mark.parametrize("rows, compound", [
    # Bi2Se3: two Se sites merged, x in [0, 1] with the multiplicity outside
    (["  Bi  Bi0  2  0.4  0.4  0.4  1", "  Sb  Sb0  2  0.4  0.4  0.4  0",
      "  Se  Se1  2  0.2  0.2  0.2  1", "  Se  Se2  1  0.0  0.0  0.0  1"], "(Bi$_{1-x}$Sb$_{x}$)$_{2}$Se$_{3}$"),
    # bcc Nb, one site of multiplicity 2 and no other element: reduced to m = 1
    (["  Nb  Nb0  2  0.0  0.0  0.0  1", "  Ti  Ti0  2  0.0  0.0  0.0  0"], "Nb$_{1-x}$Ti$_{x}$"),
    # multiplicity 1: no outer subscript
    (["  Ga  Ga0  1  0.0  0.0  0.0  1", "  In  In0  1  0.0  0.0  0.0  0",
      "  N  N1  1  0.3  0.3  0.3  1"], "Ga$_{1-x}$In$_{x}$N"),
])
def test_compound_label_uses_x_in_0_1_and_merges_sites(tmp_path: Path, rows, compound):
    assert get_elements(_cif_with_rows(tmp_path, *rows), "1x1x1")[4] == compound
    # the label carries no factor for the plots: x is plotted in [0, 1]
    assert egqca.EGQCA.extract_composition_number(None, compound) == 1


def test_get_elements_rejects_unknown_type_symbols(tmp_path: Path):
    cif_file = _cif_with_rows(tmp_path, "  B  B0  1  0.0  0.0  0.0  1", "  C  C0  1  0.0  0.0  0.0  0",
                              "  D  D1  1  0.5  0.5  0.5  1")
    with pytest.raises(ValueError, match="Unknown element 'D' .* site 'D1'"):
        get_elements(cif_file, "1x1x1")


def test_get_elements_stops_at_the_end_of_the_atom_site_loop(tmp_path: Path):
    # a data item or a new data block after the loop is not read as a site
    for trailer in ("_geom_special_details  ?", "data_second_block"):
        cif_file = _cif_with_rows(tmp_path, "  B  B0  1  0.0  0.0  0.0  1", "  C  C0  1  0.0  0.0  0.0  0",
                                  "  N  N0  1  0.5  0.5  0.5  1", trailer, "  Xx  X0  1  0.2  0.2  0.2  1")
        assert get_elements(cif_file, "1x1x1")[0] == ["B", "C", "N"]


def test_get_elements_needs_the_type_symbol_column(tmp_path: Path):
    cif_file = tmp_path / "structure.cif"
    cif_file.write_text("\n".join([
        "data_test", "loop_", " _atom_site_label", " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy",
        "  B0  1  0.00000000  0.00000000  0.00000000  1",
    ]))
    with pytest.raises(ValueError, match="no '_atom_site_type_symbol' column"):
        get_elements(cif_file, "1x1x1")


@pytest.mark.parametrize("permutable_sites, atoms_label, n_runs, last_run", [
    (32, {"mode": "coupled", "original": ["B0", "N0"], "substitution": ["C0", "C1"], "sites_per_label": [16, 16]},
     17, (32, "S32", ["-p", "B0:p=0", "-p", "N0:p=0", "-p", "C0:p=16", "-p", "C1:p=16"])),
    (4, {"mode": "coupled", "original": ["Mg0"], "substitution": ["Al0"], "sites_per_label": [4]},
     5, (4, "S4", ["-p", "Mg0:p=0", "-p", "Al0:p=4"])),
])
def test_supercell_runs(permutable_sites, atoms_label, n_runs, last_run):
    runs = list(_supercell_runs(permutable_sites, atoms_label))
    assert len(runs) == n_runs
    assert runs[-1] == last_run


def test_continuous_cluster_names():
    # 11 distributions, the supercell program numbers each from 0 with its own width
    stems = [f"S10_d{d}_i{i:0{2 if d == 10 else 1}d}_w{d + 1}" for d in range(11) for i in range(2)]
    mapping = _continuous_cluster_names(reversed(stems))

    new = [mapping[stem] for stem in stems]
    # numeric order of (d, i): d10 last, not after d1 as in string order
    assert new[:3] == ["S10_i00_w1", "S10_i01_w1", "S10_i02_w2"]
    assert new[-2:] == ["S10_i20_w11", "S10_i21_w11"]
    assert len(set(new)) == len(stems)
    assert new == sorted(new)                      # one width per S*, so string order is numeric

    # names without a distribution index are left alone; one S* at a time
    assert _continuous_cluster_names(["S2_i0_w4", "S2_i1_w2"]) == {}
    with pytest.raises(ValueError):
        _continuous_cluster_names(["S2_d0_i0_w4", "S3_d0_i0_w4"])


def _script(tmp_path: Path, name: str, body: str) -> str:
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\nimport sys\n{body}")
    script.chmod(0o755)
    return str(script)


def _fake_supercell(tmp_path: Path) -> str:
    """Executable writing two CIFs per call, named like the supercell program does."""
    return _script(tmp_path, "fake_supercell", (
        "args = sys.argv[1:]\n"
        "prefix = args[args.index('-o') + 1]\n"
        "print('fake supercell run', prefix)\n"
        "for i, w in enumerate((1, 2)):\n"
        "    open(f'{prefix}_i{i}_w{w}.cif', 'w').write('data_test\\n')\n"
    ))


DISTRIBUTION_LABEL = {
    "mode": "independent",
    "original": ["A0", "A1"],
    "substitution": ["B0", "B1"],
    "sites_per_label": [1, 2],
}


@pytest.mark.parametrize("permutable_sites, atoms_label, cif_only, expected", [
    # S1 has two distributions (1,0) and (0,1), i.e. two calls with two clusters each
    (3, DISTRIBUTION_LABEL, False, ["S1_i0_w1", "S1_i1_w2", "S1_i2_w1", "S1_i3_w2"]),
    (3, DISTRIBUTION_LABEL, True, ["S1_i0_w1", "S1_i1_w2", "S1_i2_w1", "S1_i3_w2"]),
    (2, ["A0", "B0"], True, ["S1_i0_w1", "S1_i1_w2"]),     # without distributions: names kept
])
def test_run_supercell_names_clusters_without_the_distribution_index(tmp_path, permutable_sites, atoms_label,
                                                                     cif_only, expected):
    out = tmp_path / "out"
    run_supercell(_fake_supercell(tmp_path), "x.cif", "1x1x1", permutable_sites, atoms_label, out, cif_only=cif_only)

    names = sorted(p.stem for p in out.glob("S*.cif")) if cif_only else [d.name for d in get_directories(out)]
    assert [n for n in names if n.startswith("S1_")] == expected
    assert not any("_d" in n for n in names)
    if not cif_only:
        for name in names:
            cluster = out / name.split("_")[0] / name
            assert (cluster / f"{name}.cif").is_file()
            assert (cluster / "vc-relax").is_dir()
        # the log of S1 keeps the output of both distributions
        log = (out / "S1" / "log.out").read_text()
        assert "S1_d0" in log and "S1_d1" in log


@pytest.mark.parametrize("exit_code, error, message", [
    (1, RuntimeError, "exit code 1"),
    (0, RuntimeError, "wrote no CIF files"),
    (None, FileNotFoundError, "supercell_program"),    # no such executable
])
def test_run_supercell_stops_when_the_supercell_program_fails(tmp_path, exit_code, error, message):
    if exit_code is None:
        program = str(tmp_path / "no_such_program")
    else:
        program = _script(tmp_path, "failing_supercell",
                          f"print('ERROR: occupancy of site A0 is invalid')\nsys.exit({exit_code})\n")
    out = tmp_path / "clusters"

    with pytest.raises(error, match=message) as excinfo:
        run_supercell(program, "x.cif", "1x1x1", 2, ["A0", "B0"], out)

    # the program's own output is part of the error, and no cluster folders were organised
    if exit_code is None:
        assert not out.exists()
    else:
        assert "occupancy of site A0 is invalid" in str(excinfo.value)
        assert get_directories(out) == []
