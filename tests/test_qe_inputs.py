import copy
import re
from pathlib import Path

from ase import Atoms
from ase.io import write

from egqca.constants import QE_DEFAULTS
from egqca.workflow import create_DFT_in


def _cluster_tree(tmp_path: Path) -> Path:
    """Two clusters: S0 contains only B, S1 also Mg (higher PseudoDojo cutoff)."""
    clusters = tmp_path / "clusters"
    for cid, symbols in (("S0_i0_w1", "B2"), ("S1_i0_w1", "BMg")):
        d = clusters / cid.split("_")[0] / cid
        (d / "vc-relax").mkdir(parents=True)        # as created by set_up_clusters_and_dft
        atoms = Atoms(symbols, scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=[3.0, 3.0, 3.0], pbc=True)
        write(d / f"{cid}.cif", atoms)
    pseudos = tmp_path / "pseudos"
    pseudos.mkdir()
    for el in ("B", "Mg"):
        (pseudos / f"{el}.upf").write_text("")
    return clusters


def _namelist_value(qe_in: Path, key: str) -> str:
    return re.search(rf"^\s*{key}\s*=\s*(\S+)", qe_in.read_text(), re.M).group(1)


def test_create_dft_in(tmp_path: Path):
    clusters = _cluster_tree(tmp_path)
    # create_dft_folders=False and ML-only trees have no vc-relax/ folder yet
    for relax_dir in clusters.glob("S*/S*/vc-relax"):
        relax_dir.rmdir()
    defaults_before = copy.deepcopy(QE_DEFAULTS)
    qe_par = {"pseudo_dir": str(tmp_path / "pseudos"), "system": {"degauss": 0.01}}
    qe_par_before = copy.deepcopy(qe_par)

    create_DFT_in(clusters, qe_par, {}, {}, kpts=(2, 2, 2))

    # neither the defaults nor the user's parameters are modified
    assert QE_DEFAULTS == defaults_before
    assert qe_par == qe_par_before
    for cid in ("S0_i0_w1", "S1_i0_w1"):
        qe_in = clusters / cid.split("_")[0] / cid / "vc-relax" / "vc-relax.in"
        # a user namelist keeps the other defaults of that namelist
        assert float(_namelist_value(qe_in, "degauss")) == 0.01
        assert _namelist_value(qe_in, "occupations").strip("'\"") == "smearing"
        # every cluster gets the cutoff of the hardest element: Mg 48 Ha, also for the B-only S0
        assert float(_namelist_value(qe_in, "ecutwfc")) == 2 * 48
        assert float(_namelist_value(qe_in, "ecutrho")) == 8 * 2 * 48
        assert _namelist_value(qe_in, "title").strip("'\"") == cid


def test_ph_in_uses_qe_phonon_defaults_and_user_values(tmp_path: Path):
    from egqca.constants import QE_PHONON_DEFAULTS
    from egqca.workflow import create_phonons_in

    clusters = _cluster_tree(tmp_path)
    for d in clusters.glob("S*/S*_i*_w*"):
        (d / "vc-relax" / "pw.out").write_text("")
    defaults_before = copy.deepcopy(QE_PHONON_DEFAULTS)

    create_phonons_in(clusters, qe_par={"pseudo_dir": str(tmp_path / "pseudos"), "tr2_ph": 1e-16},
                      espresso_out_file="pw.out")

    ph_in = clusters / "S1" / "S1_i0_w1" / "phonons" / "ph.in"
    assert _namelist_value(ph_in, "prefix").strip("'\"") == "S1_i0_w1"
    assert float(_namelist_value(ph_in, "tr2_ph")) == 1e-16                # user value
    assert _namelist_value(ph_in, "ldisp") == ".true."                     # default
    assert int(_namelist_value(ph_in, "nq1")) == QE_PHONON_DEFAULTS["nq1"]
    assert "pseudo_dir" not in ph_in.read_text()
    assert QE_PHONON_DEFAULTS == defaults_before


def test_find_pseudos_does_not_match_a_longer_element_symbol(tmp_path: Path):
    from egqca.create_dft_inputs import find_pseudos_for_elements
    import pytest

    for name in ("Ba.pbe-spn.UPF", "b_pbe_v1.4.uspp.F.UPF", "Nb.upf", "N.pbe.upf", "Si.upf"):
        (tmp_path / name).write_text("")

    assert find_pseudos_for_elements(tmp_path, ["B", "N"]) == {"B": "b_pbe_v1.4.uspp.F.UPF", "N": "N.pbe.upf"}
    with pytest.raises(FileNotFoundError):
        find_pseudos_for_elements(tmp_path, ["S"])


def _fcc_60deg_supercell() -> Atoms:
    """2x2x2 supercell of the 60 degree primitive fcc cell, with a1 along x as ASE builds it from a CIF."""
    from ase.geometry import cellpar_to_cell

    frac = [(i / 2, j / 2, k / 2) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    return Atoms("Cu7Ag", scaled_positions=frac, cell=cellpar_to_cell([5.9, 5.9, 5.9, 60, 60, 60]), pbc=True)


def _cubic_ops_of_lattice(cell) -> int:
    """Number of Cartesian signed-permutation rotations (pw.x's cubic group) that map the lattice onto itself."""
    import itertools
    import numpy as np

    lat = np.asarray(cell)
    inv = np.linalg.inv(lat)
    n = 0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            rot = np.zeros((3, 3))
            rot[range(3), perm] = signs
            m = lat @ rot.T @ inv
            n += np.allclose(m, np.round(m), atol=1e-8)
    return n


def test_standard_orientation_aligns_fcc_axes_and_keeps_geometry():
    import numpy as np
    from egqca.create_dft_inputs import standard_orientation

    atoms = _fcc_60deg_supercell()
    rotated = standard_orientation(atoms)

    assert _cubic_ops_of_lattice(atoms.cell) < 48
    assert _cubic_ops_of_lattice(rotated.cell) == 48
    assert np.allclose(rotated.cell.cellpar(), atoms.cell.cellpar())
    assert np.allclose(rotated.get_scaled_positions(wrap=False), atoms.get_scaled_positions(wrap=False))
    assert np.isclose(np.linalg.det(rotated.cell), np.linalg.det(atoms.cell))   # no reflection
    assert np.allclose(atoms.cell.array, _fcc_60deg_supercell().cell.array)      # input untouched


def test_vc_relax_in_uses_standard_orientation(tmp_path: Path):
    from ase.io import read

    clusters = tmp_path / "clusters"
    d = clusters / "S1" / "S1_i0_w1"
    (d / "vc-relax").mkdir(parents=True)
    write(d / "S1_i0_w1.cif", _fcc_60deg_supercell())
    pseudos = tmp_path / "pseudos"
    pseudos.mkdir()
    for el in ("Ag", "Cu"):
        (pseudos / f"{el}.upf").write_text("")

    create_DFT_in(clusters, {"pseudo_dir": str(pseudos), "system": {"ecutwfc": 40}}, {}, {}, kpts=(2, 2, 2))

    qe_in = d / "vc-relax" / "vc-relax.in"
    assert _cubic_ops_of_lattice(read(qe_in, format="espresso-in").cell) == 48
    assert _namelist_value(qe_in, "cell_dofree").strip("'\"") == "all"
