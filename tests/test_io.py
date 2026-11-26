from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egqca.constants import GPA_TO_PA, J_to_meV, Ry_to_mev, angstrom_to_meter
from egqca.io import (
    convert_volume_file,
    read_clusters_dat,
    read_phonons,
    read_properties,
    read_volume_dat,
)
from egqca.workflow import extract_total_energies, determine_dft_solver


def test_read_clusters_dat_reads_metadata_and_sorts_clusters(tmp_path: Path):
    clusters_file = tmp_path / "clusters.dat"
    clusters_file.write_text(
        "# EGQCA input for compound=A_{1-x}B_x\n"
        "# total number of sites=16\n"
        "# permutable sites n=16\n"
        "S10_i0_w4 10 16 0.625 4 -1.0\n"
        "S2_i0_w4 2 16 0.125 4 -1.5\n"
        "S1_i0_w2 1 16 0.0625 2 -2.0\n"
    )

    cluster_id, nj, nat, xj, gj, ej_ry, sites, n, compound = read_clusters_dat(
        SimpleNamespace(out_dir=tmp_path), clusters_file, dft_out_file=None,
        sites=None, n=None, compound=None, dft_out_file_subfolder="vc-relax",
    )

    # numeric order: S10 after S2
    assert cluster_id.tolist() == ["S1_i0_w2", "S2_i0_w4", "S10_i0_w4"]
    assert nj.tolist() == [1.0, 2.0, 10.0]
    assert xj.tolist() == [0.0625, 0.125, 0.625]
    assert gj.tolist() == [2.0, 4.0, 4.0]
    assert ej_ry.tolist() == [-2.0, -1.5, -1.0]
    assert nat.tolist() == [16, 16, 16]          # the Nat column of the file, not sites
    assert (sites, n, compound) == (16, 16, "A_{1-x}B_x")


def test_read_clusters_dat_warns_for_flagged_rows(tmp_path: Path):
    clusters_file = tmp_path / "clusters.dat"
    clusters_file.write_text(
        "# total number of sites=16\n# permutable sites n=16\n"
        "S1_i0_w2 1 16 0.0625 2 -2.0    # WARNING: DFT run did not finish or its relaxation did not converge\n"
        "S2_i0_w4 2 16 0.125 4 -1.5\t  # WARNING: Relaxation did not converge\n"
        "S2_i1_w4 2 16 0.125 4 -1.4    # WARNING: DFT run did not finish or its relaxation did not converge\n"
        "S3_i0_w4 3 16 0.1875 4 -1.0    # a user note\n"
    )

    with pytest.warns(UserWarning) as record:
        cluster_id, *_ = read_clusters_dat(
            SimpleNamespace(out_dir=tmp_path), clusters_file, dft_out_file=None,
            sites=None, n=None, compound=None, dft_out_file_subfolder="vc-relax",
        )

    messages = sorted(str(w.message) for w in record)
    assert messages == [
        "clusters.dat: DFT run did not finish or its relaxation did not converge for 2 cluster(s): S1_i0_w2, S2_i1_w4",
        "clusters.dat: Relaxation did not converge for 1 cluster(s): S2_i0_w4",
    ]
    assert len(cluster_id) == 4


@pytest.mark.parametrize("energy, error", [
    ("missing", "contains both, numeric values and 'missing'"),   # 'missing' without DFT outputs
    ("nan", "not finite numbers: Ej of S1_i0_w2"),
])
def test_read_clusters_dat_rejects_bad_energies(tmp_path: Path, energy, error):
    clusters_file = tmp_path / "clusters.dat"
    clusters_file.write_text(
        f"# total number of sites=16\n# permutable sites n=8\nS1_i0_w2 1 16 0.125 2 {energy}\nS2_i0_w4 2 16 0.25 4 -1.5\n"
    )
    with pytest.raises(ValueError, match=error):
        read_clusters_dat(SimpleNamespace(out_dir=tmp_path), clusters_file, None, 16, 8, "AB", "vc-relax")


@pytest.mark.parametrize("model_ids, rows, expected_ids", [
    (["S1_i0_w1", "S2_i0_w1"], ["S2_i0_w1", "S1_i0_w1"], ["S1_i0_w1", "S2_i0_w1"]),   # all clusters: model order
    (["S1_i0_w1", "S2_i0_w1", "S3_i0_w1"], ["S3_i0_w1", "S1_i0_w1"], ["S3_i0_w1", "S1_i0_w1"]),   # subset: file order
])
def test_read_properties_orders_rows(tmp_path: Path, model_ids, rows, expected_ids):
    properties_file = tmp_path / "properties.dat"
    properties_file.write_text("# ID gap DOS\n" + "".join(f"{cid} {cid[1]}.5 0.{cid[1]}\n" for cid in rows))

    prop_j, prop_names, cluster_id = read_properties(SimpleNamespace(cluster_id=np.array(model_ids)), properties_file)

    assert prop_names == ["gap", "DOS"]
    assert cluster_id.tolist() == expected_ids
    np.testing.assert_allclose(prop_j, [[float(f"{c[1]}.5"), float(f"0.{c[1]}")] for c in expected_ids])


@pytest.mark.parametrize("rows, error", [
    (["S1_i0_w1", "S2_i0_w1", "S2_i0_w1"], r"more than once: S2_i0_w1 \(2x\)"),               # subset
    (["S1_i0_w1", "S1_i0_w1", "S2_i0_w1", "S3_i0_w1"], r"more than once: S1_i0_w1 \(2x\)"),   # full size
    (["S1_i0_w1", "S3_d1_i0_w4", "S9_i0_w1"],
     r"2 cluster IDs in .*properties.dat are not in clusters.dat: S3_d1_i0_w4, S9_i0_w1"),
])
def test_read_properties_rejects_bad_ids(tmp_path: Path, rows, error):
    properties_file = tmp_path / "properties.dat"
    properties_file.write_text("# ID p\n" + "".join(f"{cid} 1.0\n" for cid in rows))
    model = SimpleNamespace(cluster_id=np.array(["S1_i0_w1", "S2_i0_w1", "S3_i0_w1", "S4_i0_w1"]))
    with pytest.raises(ValueError, match=error):
        read_properties(model, properties_file)


RY_IN_EV = Ry_to_mev / 1000
DFT_OUTPUTS = {
    "qe": "!    total energy              =     {E_Ry} Ry\n JOB DONE.\n",
    "aims": ("  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :     {E_eV} eV\n"
             "          Have a nice day.\n"),
    "vasp": ("   NSW    =      0    number of steps for IOM\n"
             "  free  energy   TOTEN  =       {E_eV} eV\n"
             " General timing and accounting informations for this job:\n"),
}


def _cluster_tree(tmp_path: Path, energies: dict, rows: dict, code: str = "qe") -> Path:
    """clusters.dat with ``rows`` {id: energy column}, and DFT outputs for ``energies`` {id: Ry or None}."""
    clusters_file = tmp_path / "DFT" / "clusters.dat"
    clusters_file.parent.mkdir(parents=True, exist_ok=True)
    clusters_file.write_text(
        "# total number of sites=2\n# permutable sites n=2\n"
        + "".join(f"{cid} {cid[1]} 2 {int(cid[1]) / 2} 1 {e}\n" for cid, e in rows.items())
    )
    for cid, energy in energies.items():
        out = tmp_path / cid.split("_")[0] / cid / "vc-relax"
        out.mkdir(parents=True)
        if energy is not None:
            text = DFT_OUTPUTS[code].format(E_Ry=energy, E_eV=f"{energy * RY_IN_EV:.10f}")
            (out / "espresso.out").write_text(text)
    return clusters_file


def _energies(clusters_file: Path) -> dict:
    return dict(np.genfromtxt(clusters_file, dtype=str, usecols=(0, 5)))


@pytest.mark.parametrize("code", ["qe", "aims", "vasp"])
def test_extract_total_energies_converts_to_ry(tmp_path: Path, code, recwarn):
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0, "S2_i0_w1": -2.5},
                                  {"S0_i0_w1": "missing", "S2_i0_w1": "missing"}, code)
    extract_total_energies(clusters_file, tmp_path, "espresso.out")

    energies = {cid: float(e) for cid, e in _energies(clusters_file).items()}
    assert energies == pytest.approx({"S0_i0_w1": -1.0, "S2_i0_w1": -2.5}, rel=1e-9)
    # completed runs: no "did not finish" warning
    assert "WARNING" not in clusters_file.read_text()
    assert not [w for w in recwarn if "did not finish" in str(w.message)]


UNCONVERGED_RELAXATIONS = {
    "qe": (" BFGS Geometry Optimization\n"
           " The maximum number of steps has been reached.\n"
           " End of BFGS Geometry Optimization\n"
           "!    total energy              =     {E_Ry} Ry\n JOB DONE.\n"),
    "aims": ("  Geometry relaxation: Modified BFGS - TRM (trust radius method) for lattice optimization.\n"
             "  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :     {E_eV} eV\n"
             "  Present geometry is not yet converged.\n"
             "          Have a nice day.\n"),
    "vasp": ("   NSW    =     15    number of steps for IOM\n"
             "  free  energy   TOTEN  =       {E_eV} eV\n"
             " General timing and accounting informations for this job:\n"),
}
CONVERGED_RELAXATIONS = {
    "qe": UNCONVERGED_RELAXATIONS["qe"].replace(
        " The maximum number of steps has been reached.\n",
        " bfgs converged in  12 scf cycles and  10 bfgs steps\n"),
    "aims": UNCONVERGED_RELAXATIONS["aims"].replace(
        "Present geometry is not yet converged.", "Present geometry is converged."),
    "vasp": UNCONVERGED_RELAXATIONS["vasp"].replace(
        " General timing",
        " reached required accuracy - stopping structural energy minimisation\n General timing"),
}


@pytest.mark.parametrize("code", ["qe", "aims", "vasp"])
@pytest.mark.parametrize("converged", [True, False])
def test_extract_total_energies_flags_unconverged_relaxations(tmp_path: Path, monkeypatch, code, converged, recwarn):
    outputs = CONVERGED_RELAXATIONS if converged else UNCONVERGED_RELAXATIONS
    monkeypatch.setitem(DFT_OUTPUTS, code, outputs[code])
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0, "S2_i0_w1": -2.5},
                                  {"S0_i0_w1": "missing", "S2_i0_w1": "missing"}, code)
    extract_total_energies(clusters_file, tmp_path, "espresso.out")

    energies = {cid: float(e) for cid, e in _energies(clusters_file).items()}
    assert energies == pytest.approx({"S0_i0_w1": -1.0, "S2_i0_w1": -2.5}, rel=1e-9)
    assert ("relaxation did not converge" in clusters_file.read_text()) != converged
    assert any("relaxation did not converge" in str(w.message) for w in recwarn) != converged


def test_extract_total_energies_replaces_old_warning_comment(tmp_path: Path):
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0}, {"S0_i0_w1": "-1.0"})
    text = clusters_file.read_text().replace("-1.0\n", "-1.0    # WARNING: DFT calculation did not finish successfully\n")
    clusters_file.write_text(text)
    extract_total_energies(clusters_file, tmp_path, "espresso.out")

    assert "WARNING" not in clusters_file.read_text()


@pytest.mark.parametrize("text, solver", [
    # QE header that mentions 'aims' and 'vasp' in paths before its own marker would be found
    ("     Pseudo file /home/u/aims_vasp_pp/Pb.upf\n     Program PWSCF v.7.2 starts on\n", "espresso"),
    ("  Invoking FHI-aims ...\n  pseudo_dir /home/u/espresso\n", "aims"),
    (" vasp.6.4.2 20Jul23 (build ...)\n   Quantum ESPRESSO-like comment\n", "vasp"),
    *[(DFT_OUTPUTS[code].format(E_Ry=-1.0, E_eV=-13.6), solver)        # no header: energy line
      for code, solver in [("qe", "espresso"), ("aims", "aims"), ("vasp", "vasp")]],
])
def test_determine_dft_solver(tmp_path: Path, text, solver):
    (tmp_path / "out").write_text(text)
    assert determine_dft_solver(tmp_path / "out") == solver
    assert determine_dft_solver(text.splitlines(keepends=True)) == solver


def test_determine_dft_solver_rejects_unknown_output(tmp_path: Path):
    (tmp_path / "out").write_text("my aims with espresso and vasp\n")
    with pytest.raises(ValueError, match="Could not determine the DFT code"):
        determine_dft_solver(tmp_path / "out")


def test_extract_total_energies_rejects_mixed_codes(tmp_path: Path):
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0, "S2_i0_w1": -2.5},
                                  {"S0_i0_w1": "missing", "S2_i0_w1": "missing"}, "qe")
    aims_out = DFT_OUTPUTS["aims"].format(E_eV=-2.5 * RY_IN_EV)
    (tmp_path / "S2" / "S2_i0_w1" / "vc-relax" / "espresso.out").write_text(aims_out)
    before = clusters_file.read_text()

    with pytest.raises(ValueError, match=r"different codes(.|\n)*espresso \(1\): S0_i0_w1(.|\n)*aims \(1\): S2_i0_w1"):
        extract_total_energies(clusters_file, tmp_path, "espresso.out")
    assert clusters_file.read_text() == before


def test_extract_total_energies_lists_every_cluster_without_an_energy(tmp_path: Path):
    rows = {"S0_i0_w1": "missing", "S1_i0_w1": "missing", "S1_i1_w1": "missing", "S2_i0_w1": "missing"}
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0, "S1_i1_w1": None, "S2_i0_w1": -2.0}, rows)
    (tmp_path / "S2" / "S2_i0_w1" / "vc-relax" / "espresso.out").write_text("no energy here\n")

    with pytest.raises(ValueError) as err:
        extract_total_energies(clusters_file, tmp_path, "espresso.out")

    message = str(err.value)
    assert "3 of the 4 clusters" in message
    assert "no cluster folder (1)" in message and "S1_i0_w1" in message
    assert "no output file (1)" in message and "S1_i1_w1" in message
    assert "no energy in the output file (1)" in message and "S2_i0_w1" in message
    # the completed cluster is written, the others stay missing
    assert _energies(clusters_file) == {"S0_i0_w1": "-1.0", "S1_i0_w1": "missing", "S1_i1_w1": "missing",
                                        "S2_i0_w1": "missing"}


def test_extract_total_energies_matches_ids_and_keeps_existing_energies(tmp_path: Path):
    # S0 has an energy but no output; S1_i1 and S1_i10 share a prefix; S1_i2 is not in clusters.dat
    rows = {"S0_i0_w1": "-5.0", "S1_i1_w1": "missing", "S1_i10_w1": "missing"}
    clusters_file = _cluster_tree(tmp_path, {"S1_i1_w1": -1.0, "S1_i10_w1": -10.0, "S1_i2_w1": -2.0}, rows)

    with pytest.warns(UserWarning) as record:
        extract_total_energies(clusters_file, tmp_path, "espresso.out")

    messages = " ".join(str(w.message) for w in record)
    assert "No DFT energy found for S0_i0_w1" in messages and "S1_i2_w1" in messages
    assert _energies(clusters_file) == {"S0_i0_w1": "-5.0", "S1_i1_w1": "-1.0", "S1_i10_w1": "-10.0"}


def test_read_clusters_dat_reextracts_all_energies_and_writes_volume_dat(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr("egqca.io.set_up_volume_dat", lambda *args, **kwargs: calls.append((args, kwargs)))
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0, "S2_i0_w1": -2.0},
                                  {"S0_i0_w1": "-99.0", "S2_i0_w1": "missing"})

    model = SimpleNamespace(out_dir=tmp_path, clusters_folder=tmp_path)
    cluster_id, *_, Ej_Ry, _sites, _n, _compound = read_clusters_dat(
        model, clusters_file, "espresso.out", None, None, "AB", "vc-relax"
    )

    assert cluster_id.tolist() == ["S0_i0_w1", "S2_i0_w1"]
    assert Ej_Ry.tolist() == [-1.0, -2.0]
    # volume.dat goes into the run folder of the clusters file
    assert calls == [((tmp_path, "espresso.out", "vc-relax"), {"run_folder": clusters_file.parent})]


def test_read_clusters_dat_warns_if_volume_dat_cannot_be_written(tmp_path: Path, monkeypatch):
    def failing_set_up_volume_dat(*args, **kwargs):
        raise ValueError("no cell in output")

    monkeypatch.setattr("egqca.io.set_up_volume_dat", failing_set_up_volume_dat)
    clusters_file = _cluster_tree(tmp_path, {"S0_i0_w1": -1.0, "S2_i0_w1": -2.0},
                                  {"S0_i0_w1": "missing", "S2_i0_w1": "missing"})

    model = SimpleNamespace(out_dir=tmp_path, clusters_folder=tmp_path)
    with pytest.warns(UserWarning, match="Could not write volume.dat"):
        *_, compound = read_clusters_dat(model, clusters_file, "espresso.out", None, None, "AB", "vc-relax")
    assert compound == "AB"


def test_read_volume_dat_converts_lattice_to_pv_energy(tmp_path: Path):
    volume_file = tmp_path / "volume.dat"
    volume_file.write_text(
        "S2_i0_w4 3.0 3.0 3.0 90 90 90\n"
        "S1_i0_w2 2.0 2.0 2.0 90 90 90\n"
    )
    model = SimpleNamespace(out_dir=tmp_path, pressure_GPa=2.0, sites=4, cluster_id=np.array(["S1_i0_w2", "S2_i0_w4"]))

    ep, returned_file = read_volume_dat(
        model, volume_file=volume_file, dft_out_file=None, dft_out_file_subfolder="vc-relax",
        E_elec=np.array([-1.0, -2.0]),
    )

    expected_volumes = np.array([8.0, 27.0]) * angstrom_to_meter**3
    assert returned_file == volume_file
    np.testing.assert_allclose(model.lattice.V_cell, expected_volumes)
    np.testing.assert_allclose(ep, model.pressure_GPa * GPA_TO_PA * expected_volumes * J_to_meV / model.sites)


def test_read_volume_dat_reuses_existing_dft_volume_dat(tmp_path: Path, monkeypatch):
    volume_file = tmp_path / "DFT" / "volume.dat"
    volume_file.parent.mkdir()
    volume_file.write_text(
        "S1_i0_w2 2.0 2.0 2.0 90 90 90\n"
        "S2_i0_w4 3.0 3.0 3.0 90 90 90\n"
    )

    def unexpected_set_up_volume_dat(*args, **kwargs):
        raise AssertionError("volume.dat must not be rewritten")

    monkeypatch.setattr("egqca.io.set_up_volume_dat", unexpected_set_up_volume_dat)
    model = SimpleNamespace(out_dir=tmp_path, clusters_folder=tmp_path, pressure_GPa=2.0, sites=4,
                            datafile=tmp_path / "DFT" / "clusters.dat",
                            cluster_id=np.array(["S1_i0_w2", "S2_i0_w4"]))

    _, returned_file = read_volume_dat(
        model, volume_file=None, dft_out_file="espresso.out", dft_out_file_subfolder="vc-relax",
        E_elec=np.array([-1.0, -2.0]),
    )

    assert returned_file == volume_file
    np.testing.assert_allclose(model.lattice.V_cell, np.array([8.0, 27.0]) * angstrom_to_meter**3)


def test_convert_volume_file(tmp_path: Path):
    volume_file = tmp_path / "volume.dat"
    volume_file.write_text(
        "#ID  a(ang) b(ang) c(ang) alpha(deg) beta(deg) gamma(deg)\n"
        "S0_i0_w1\t2.0\t2.0\t2.0\t60.0\t60.0\t60.0\n"
    )

    def rhombohedral_to_hexagonal(a, b, c, alpha, beta, gamma):
        alpha_rad = np.deg2rad(alpha)
        a_h = a * np.sqrt(2.0 * (1.0 - np.cos(alpha_rad)))
        c_h = a * np.sqrt(3.0 * (1.0 + 2.0 * np.cos(alpha_rad)))
        return {"a": a_h, "b": a_h, "c": c_h, "alpha": 90.0, "beta": 90.0, "gamma": 120.0}

    # a converter returning a dict, default output name
    converted_file = convert_volume_file(volume_file, rhombohedral_to_hexagonal)
    assert converted_file == tmp_path / "volume_converted.dat"
    assert converted_file.read_text().startswith("#ID")
    assert np.genfromtxt(converted_file, dtype=str, usecols=0) == "S0_i0_w1"
    np.testing.assert_allclose(np.genfromtxt(converted_file, usecols=range(1, 7)),
                               [2.0, 2.0, np.sqrt(24.0), 90.0, 90.0, 120.0])
    # a second call (e.g. the analysis script run again) returns the existing file
    content = converted_file.read_text()
    assert convert_volume_file(volume_file, rhombohedral_to_hexagonal) == converted_file
    assert converted_file.read_text() == content

    # a converter returning a tuple, custom output name
    converted_file = convert_volume_file(
        volume_file, lambda a, b, c, alpha, beta, gamma: (a, b, 2.0 * c, alpha, beta, gamma),
        volume_file_name="volume_hex.dat",
    )
    assert converted_file == tmp_path / "volume_hex.dat"
    assert converted_file.read_text().splitlines()[-1] == "S0_i0_w1\t2.0\t2.0\t4.0\t60.0\t60.0\t60.0"


def _write_phonon_dos(tmp_path: Path, modes_per_site: float, sites: int, shift_eV: float = 0.0) -> SimpleNamespace:
    """Folder-mode DOS (eV) of one cluster: a Gaussian with the given number of modes per site."""
    (tmp_path / "S0" / "S0_i0_w1").mkdir(parents=True)
    # grid starts below zero only when the DOS is shifted there on purpose
    energies = np.linspace(1e-4 + min(shift_eV, 0.0) * 2, 0.06, 2001)
    gaussian = np.exp(-0.5 * ((energies - 0.03 - shift_eV) / 0.005) ** 2)
    weights = gaussian * modes_per_site * sites / np.trapezoid(gaussian, energies)
    (tmp_path / "phonons").mkdir()
    np.savetxt(tmp_path / "phonons" / "S0_i0_w1.dos", np.column_stack((energies, weights)), header="Energy(eV) DOS")
    return SimpleNamespace(clusters_folder=tmp_path, cluster_id=np.array(["S0_i0_w1"]))


def _modes_per_site(phonon_data) -> float:
    w, dos = phonon_data[0].T
    return float(np.trapezoid(dos, w))


@pytest.mark.parametrize("modes_per_site", [3.0, 24.0])
def test_read_phonons_normalizes_to_three_modes_per_site(tmp_path: Path, recwarn, modes_per_site):
    # 24: e.g. ASE <= 3.24, where the DOS is not divided by the number of q-points
    model = _write_phonon_dos(tmp_path, modes_per_site=modes_per_site, sites=4)

    phonon_data = read_phonons(model, "phonons", sites=4)

    assert _modes_per_site(phonon_data) == pytest.approx(3.0, rel=1e-6)
    rescaled = [str(w.message) for w in recwarn if "rescaled" in str(w.message)]
    assert bool(rescaled) == (modes_per_site != 3.0)
    assert all("S0_i0_w1 (24)" in message for message in rescaled)


def test_read_phonons_normalizes_before_dropping_imaginary_modes(tmp_path: Path):
    # centred at 0: about half of the modes are imaginary and must stay excluded
    model = _write_phonon_dos(tmp_path, modes_per_site=3.0, sites=4, shift_eV=-0.03)

    with pytest.warns(UserWarning, match="imaginary"):
        phonon_data = read_phonons(model, "phonons", sites=4)

    assert _modes_per_site(phonon_data) == pytest.approx(1.5, rel=1e-2)


def test_read_phonons_ignores_broadening_tails_below_zero(tmp_path: Path, recwarn):
    # grid from -30 meV with a zero DOS there except a tiny tail, as from a Gaussian-broadened DOS
    model = _write_phonon_dos(tmp_path, modes_per_site=3.0, sites=4)
    dos_file = tmp_path / "phonons" / "S0_i0_w1.dos"
    energies, weights = np.loadtxt(dos_file).T
    below = np.linspace(-0.03, 0.0, 301)
    tail = 1e-6 * weights.max() * np.exp(below / 0.002)
    np.savetxt(dos_file, np.column_stack((np.r_[below, energies], np.r_[tail, weights])), header="Energy(eV) DOS")

    read_phonons(model, "phonons", sites=4)

    assert not [w for w in recwarn if "imaginary" in str(w.message)]


def test_read_phonons_reports_lowest_imaginary_frequency(tmp_path: Path):
    # Gaussian at -10 meV (width 5 meV): reported where the DOS falls below 1e-3 of its maximum
    model = _write_phonon_dos(tmp_path, modes_per_site=3.0, sites=4, shift_eV=-0.04)

    with pytest.warns(UserWarning, match=r"down to -2\d\.\d meV") as record:
        read_phonons(model, "phonons", sites=4)

    assert "S0_i0_w1" in str(record[0].message)


@pytest.mark.parametrize(
    "header, eV_to_unit",
    [("Energy(meV) DOS", 1.0e3), ("Frequency(THz) DOS", 241.79892), ("Frequency(GHz) DOS", 241798.92)],
)
def test_read_phonons_detects_prefixed_units(tmp_path: Path, header: str, eV_to_unit: float):
    # meV must not be read as eV, and THz/GHz not as Hz
    model = _write_phonon_dos(tmp_path, modes_per_site=3.0, sites=4)
    dos_file = tmp_path / "phonons" / "S0_i0_w1.dos"
    reference = read_phonons(model, "phonons", sites=4)[0]

    energies, weights = np.loadtxt(dos_file).T
    np.savetxt(dos_file, np.column_stack((energies * eV_to_unit, weights / eV_to_unit)), header=header)
    converted = read_phonons(model, "phonons", sites=4)[0]

    np.testing.assert_allclose(converted, reference, rtol=1e-6)


@pytest.mark.parametrize("text, error", [
    ("#ID a b\nS1_i0_w2 1.0 nan\nS2_i0_w4 2.0 3.0\n", r"NaN values \(b: S1_i0_w2\)"),
    ("#ID a b c\nS1_i0_w2 1.0 2.0\nS2_i0_w4 2.0 3.0\n", "header names 3 properties .* data has 2"),
    ("#ID Band gap\nS1_i0_w2 1.0\nS2_i0_w4 2.0\n", "e.g. 'Band-gap'"),       # a name with a space
])
def test_read_properties_rejects_nan_and_column_mismatch(tmp_path: Path, text, error):
    properties_file = tmp_path / "properties.dat"
    properties_file.write_text(text)
    model = SimpleNamespace(cluster_id=np.array(["S1_i0_w2", "S2_i0_w4"]))
    with pytest.raises(ValueError, match=error):
        read_properties(model, properties_file)


def test_convert_volume_file_leaves_no_file_when_a_row_fails(tmp_path: Path):
    volume_file = tmp_path / "volume.dat"
    volume_file.write_text("S0_i0_w1\t2.0\t2.0\t2.0\t60.0\t60.0\t60.0\nS1_i0_w1\t2.0\t2.0\n")
    identity = lambda a, b, c, alpha, beta, gamma: dict(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    with pytest.raises(ValueError, match="Expected 7 columns"):
        convert_volume_file(volume_file, identity)
    assert not list(tmp_path.glob("*converted*")) and len(list(tmp_path.iterdir())) == 1
