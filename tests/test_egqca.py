from pathlib import Path

import numpy as np
import pytest

from conftest import N_PERMUTABLE, SITES, cluster_energy, ring_clusters, write_alloy
from egqca import EGQCA
from egqca.constants import Ry_to_mev, kB

X_GRID = np.array([0.12, 0.48, 0.67, 0.89])
TLIST = [600.0, 1000.0]
APPROXIMATIONS = ["overlapping", "non-overlapping"]


def _model(alloy, out_dir, entropy_approximation: str = "overlapping") -> EGQCA:
    clusters_file, volume_file = alloy
    return EGQCA(
        datafile=clusters_file,
        volume_file=volume_file,
        out_folder_name=out_dir,
        entropy_approximation=entropy_approximation,
    )


def _write_properties(path: Path, cluster_ids, name: str = "test_prop") -> None:
    with open(path, "w") as f:
        f.write(f"# ID {name}\n")
        for idx, cluster_id in enumerate(cluster_ids):
            f.write(f"{cluster_id} {float(idx):.6f}\n")


def _write_surrogate_model(path: Path, cluster_ids) -> None:
    with open(path, "w") as f:
        f.write(f"# Selected clusters: {len(cluster_ids)}\n")
        f.write("# Cluster ID\t Probability weight\n")
        for cluster_id in cluster_ids:
            f.write(f"{cluster_id}\t 1.0\n")


def _one_cluster_per_nj(model: EGQCA) -> list[int]:
    return [int(np.where(model.nj == nj)[0][0]) for nj in np.unique(model.nj)]


@pytest.mark.parametrize("entropy_approximation", APPROXIMATIONS)
def test_thermodynamic_assessment(alloy, tmp_path, capsys, entropy_approximation):
    m = _model(alloy, tmp_path, entropy_approximation)
    m.thermodynamic_assessment(X_GRID, TLIST, run_miscibility_gap=False, plot_flag=False)

    # the overview of the last run is written next to its results
    overview = tmp_path / "thermodynamic_assessment" / EGQCA.OVERVIEW_FILE
    assert f"Entropy approximation : {entropy_approximation}" in overview.read_text()

    results_dir = tmp_path / "thermodynamic_assessment"
    for name in ("Delta_G", "Delta_H", "Delta_S", "excess_energy"):
        assert (results_dir / f"{name}.dat").is_file()
        assert (results_dir / f"{name}.pdf").is_file()

    # excess energies: A_{2-x}B_x plots x_j scaled by 2; values in meV per site
    excess = np.loadtxt(results_dir / "excess_energy.dat")
    configs = [config for config, _ in ring_clusters()]
    linear = np.array([cluster_energy(c, 0.0) for c in configs])
    expected = (np.array([cluster_energy(c, -0.01) for c in configs]) - linear) * Ry_to_mev / SITES
    np.testing.assert_allclose(excess[:, 0], 2 * m.xj)
    np.testing.assert_allclose(excess[:, 1], expected, atol=1e-6)

    # p_j are normalized, give the composition x and minimize F under these constraints
    x = m.x
    pj0 = m.calc_pj0(x)
    constraints = np.vstack([np.ones_like(m.nj, dtype=float), m.nj.astype(float)])
    directions = np.linalg.svd(constraints)[2][constraints.shape[0]:]   # null space
    rng = np.random.default_rng(0)
    for T in TLIST:
        p = m.pj[T]
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-10)
        np.testing.assert_allclose(p @ m.nj, N_PERMUTABLE * x, rtol=1e-8)
        F0 = m.calc_F(T, p, m.deltaFvib_j[T], x=x, pj0=pj0)
        assert np.all(F0 <= m.calc_F(T, pj0, m.deltaFvib_j[T], x=x, pj0=pj0) + 1e-9)
        for _ in range(5):
            d = rng.normal(size=directions.shape[0]) @ directions
            eps = 1e-3 * np.min(p / (np.abs(d) + 1e-300), axis=1, keepdims=True)
            for sign in (1.0, -1.0):
                F = m.calc_F(T, p + sign * eps * d, m.deltaFvib_j[T], x=x, pj0=pj0)
                assert np.all(F - F0 >= -1e-9 * np.abs(F0))

    m.thermodynamic_assessment(X_GRID, TLIST[:1], run_miscibility_gap=False, plot_flag=False)
    assert overview.read_text().count("Thermodynamic assessment") == 1     # overwritten, not appended


def test_ideal_solution_has_the_ideal_mixing_free_energy(tmp_path):
    # without interactions both entropy approximations give the random solid solution
    clusters_file, volume_file = write_alloy(tmp_path / "data", J=0.0)
    x = np.array([0.1, 0.3, 0.5, 0.8])
    ideal = (N_PERMUTABLE / SITES) * (x * np.log(x) + (1 - x) * np.log(1 - x))
    for entropy_approximation in APPROXIMATIONS:
        m = EGQCA(datafile=clusters_file, volume_file=volume_file,
                  out_folder_name=tmp_path / entropy_approximation, entropy_approximation=entropy_approximation)
        x_prepared = m._prepare_main(x, TLIST)
        m.init_thermo()
        m._ensure_probabilities(x_prepared, TLIST)
        for T in TLIST:
            m.thermo(T)
            np.testing.assert_allclose(m.pj[T], m.calc_pj0(x_prepared), atol=1e-10)
            np.testing.assert_allclose(m.F[T], kB * T * ideal, atol=1e-6)


def test_entropy_approximation_is_validated_and_resets_cached_results(alloy, tmp_path):
    with pytest.raises(ValueError, match="entropy_approximation"):
        _model(alloy, tmp_path, "pairwise")

    m = _model(alloy, tmp_path)
    m.thermodynamic_assessment(X_GRID, TLIST[:1], run_miscibility_gap=False, plot_flag=False)
    assert m.D_weight == m.n
    m.entropy_approximation = "non_overlapping"      # "_" is accepted as well
    assert m.entropy_approximation == "non-overlapping"
    assert m.D_weight == 1.0
    assert m.pj == {}
    assert m.F == {}


def test_pressure_without_volume_warns_and_is_set_to_zero(alloy, tmp_path):
    with pytest.warns(UserWarning, match="requires cell volumes") as record:
        m = EGQCA(datafile=alloy[0], out_folder_name=tmp_path, pressure_GPa=5.0)
    assert m.pressure_GPa == 0.0
    assert Path(record[0].filename) == Path(__file__)


def test_cluster_importance(alloy, tmp_path, capsys):
    m = _model(alloy, tmp_path)

    m.cluster_importance(X_GRID, [600.0], top_percentage=[0.1], plot_flag=False)

    expected_dir = tmp_path / "cluster_reduction"
    assert m.cluster_reduction_dir == m.current_output_dir == expected_dir
    assert list(expected_dir.glob("surrogate_T600.0_*clusters.dat"))
    assert not (tmp_path / "thermodynamic_assessment").exists()
    assert not (tmp_path / "results").exists()      # no property averaging
    assert "Cluster importance" in (expected_dir / EGQCA.OVERVIEW_FILE).read_text()


def test_average_properties(alloy, tmp_path):
    m = _model(alloy, tmp_path)
    results_dir = tmp_path / "results"

    # all clusters and two surrogate models; a property equal to the cluster index
    _write_properties(tmp_path / "index.dat", m.cluster_id, name="cluster_index")
    small = _one_cluster_per_nj(m)
    large = small + [next(i for i in range(len(m.cluster_id)) if i not in small)]
    surrogates = []
    for name, indices in (("surrogate_small.dat", small), ("surrogate_large.dat", large)):
        _write_surrogate_model(tmp_path / name, m.cluster_id[indices])
        surrogates.append(tmp_path / name)
    m.average_properties(tmp_path / "index.dat", Tlist=[600.0], x=X_GRID, surrogate_model=surrogates)

    output = results_dir / "cluster_index_T600.0.dat"
    assert (results_dir / "cluster_index_T600.0.pdf").is_file()
    header = output.read_text().splitlines()[0]
    n = len(m.cluster_id)
    for column in (f"avg_{n}_clusters", f"std_{n}_clusters", f"avg_{len(small)}_clusters", f"avg_{len(large)}_clusters"):
        assert column in header
    # the full average is sum_j p_j * j (the file keeps 8 digits)
    np.testing.assert_allclose(np.loadtxt(output)[:, 1], m.pj[600.0] @ np.arange(n, dtype=float), rtol=1e-8)

    # only a subset of the clusters has the property: no standard deviation
    _write_properties(tmp_path / "subset.dat", m.cluster_id[small])
    m.average_properties(tmp_path / "subset.dat", Tlist=[600.0], x=X_GRID, surrogate_model=surrogates[0])
    header = (results_dir / "test_prop_T600.0.dat").read_text().splitlines()[0]
    assert f"avg_{len(small)}_clusters" in header and "std_" not in header


@pytest.mark.parametrize("entropy_approximation", APPROXIMATIONS)
def test_entropy_is_finite_at_low_temperature(alloy, tmp_path, entropy_approximation):
    # probabilities underflow to exactly 0 at low T; 0 ln 0 must count as 0, not NaN
    m = _model(alloy, tmp_path, entropy_approximation)
    m.delta_j = m.excess_energy()
    x = np.arange(1e-6, 1.0, 1e-3)
    m.x = x
    m.apriori(x)
    pj = m.calc_pj(1.0, x)

    assert (pj == 0).any()
    assert np.isfinite(m.Ds(pj)).all()
    assert np.isfinite(m.Ss(pj)).all()


def test_deltaFvib_j_needs_one_value_per_cluster(alloy, tmp_path):
    m = _model(alloy, tmp_path)
    m.delta_j = m.excess_energy()
    x = np.array([0.25, 0.5])
    m.x = x
    m.apriori(x)
    pj = m.calc_pj(600.0, x)

    for wrong in (1.0, np.zeros(len(m.delta_j) - 1)):
        with pytest.raises(ValueError, match="one value per cluster"):
            m.calc_pj(600.0, x, wrong)
        with pytest.raises(ValueError, match="one value per cluster"):
            m.calc_F(600.0, pj, wrong)
    # None and an explicit zero array give the same probabilities
    np.testing.assert_allclose(m.calc_pj(600.0, x, np.zeros(len(m.delta_j))), pj)


def test_thermodynamic_assessment_reuses_cached_temperatures(alloy, tmp_path, monkeypatch):
    m = _model(alloy, tmp_path)
    m.thermodynamic_assessment(X_GRID, TLIST, run_miscibility_gap=False, plot_flag=False)
    calls = []
    equilibrate = m.equilibrate
    monkeypatch.setattr(m, "equilibrate", lambda T, x: (calls.append(T), equilibrate(T, x)))

    m.thermodynamic_assessment(X_GRID, TLIST, run_miscibility_gap=False, plot_flag=False)
    assert calls == []          # same grid and temperatures: nothing recomputed
    assert set(m.F) == set(TLIST)
