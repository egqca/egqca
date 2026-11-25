from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from egqca.constants import DEFAULT_KPTS
from egqca.workflow import _set_aims_k_grid, create_phonons_in


def _aims_cluster(tmp_path: Path) -> tuple[Path, Path, Atoms]:
    """One NbC cluster whose vc-relax/ holds a relaxed geometry.in.next_step."""
    clusters = tmp_path / "clusters"
    d = clusters / "S0" / "S0_i0_w1"
    (d / "vc-relax").mkdir(parents=True)
    unrelaxed = Atoms("NbC", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=[3.0, 3.0, 3.0], pbc=True)
    write(d / "S0_i0_w1.cif", unrelaxed)
    relaxed = unrelaxed.copy()
    relaxed.set_cell([2.9, 2.9, 2.9], scale_atoms=True)
    write(d / "vc-relax" / "geometry.in.next_step", relaxed, format="aims")

    species = tmp_path / "species"
    species.mkdir()
    for name in ("41_Nb_default", "06_C_default"):
        (species / name).write_text(f"  species {name.split('_')[1]}\n")
    return clusters, species, relaxed


@pytest.mark.parametrize("friction", [True, False])
def test_aims_phonons_use_the_relaxed_supercell(tmp_path: Path, friction):
    clusters, species, relaxed = _aims_cluster(tmp_path)

    create_phonons_in(clusters, aims_par={"species_dir": str(species), "xc": "pbe"},
                      phonon_supercell=(2, 1, 1), aims_calculate_friction=friction)

    geometry = clusters / "S0" / "S0_i0_w1" / "phonons" / "geometry.in"
    atoms = read(geometry, format="aims")
    assert len(atoms) == 2 * len(relaxed)
    assert np.allclose(atoms.cell, relaxed.cell * [[2], [1], [1]])

    # with friction: one calculate_friction line right after every atom, else none
    lines = geometry.read_text().splitlines()
    atom_idx = [i for i, line in enumerate(lines) if line.startswith("atom")]
    assert len(atom_idx) == len(atoms)
    assert sum(line.startswith("calculate_friction") for line in lines) == (len(atoms) if friction else 0)
    if friction:
        assert all(lines[i + 1] == "calculate_friction .true." for i in atom_idx)


def test_aims_phonons_without_relaxed_structure_raise(tmp_path: Path):
    clusters, species, _ = _aims_cluster(tmp_path)
    (clusters / "S0" / "S0_i0_w1" / "vc-relax" / "geometry.in.next_step").unlink()

    with pytest.raises(FileNotFoundError):
        create_phonons_in(clusters, aims_par={"species_dir": str(species), "xc": "pbe"})


@pytest.mark.parametrize("aims_out, message", [
    ("  Geometry relaxation: BFGS\n  Present geometry is not yet converged.\n", "did not finish"),
    ("  Geometry relaxation: BFGS\n  Present geometry is not yet converged.\n          Have a nice day.\n", "did not converge"),
])
def test_aims_relaxed_structure_warns_for_unfinished_relaxation(tmp_path: Path, monkeypatch, aims_out, message):
    import egqca.workflow as workflow

    (tmp_path / "aims.out").write_text(aims_out)
    monkeypatch.setattr(workflow, "read", lambda *args, **kwargs: Atoms("H"))  # skip parsing the fake output

    with pytest.warns(UserWarning, match=message):
        workflow._read_aims_relaxed_structure(tmp_path, "aims.out")


@pytest.mark.parametrize("aims_par, kpts, expected, warns", [
    ({"k_grid": "6 6 6"}, None, "6 6 6", False),                 # user k_grid is kept
    ({}, None, DEFAULT_KPTS, False),
    ({"k_grid": "6 6 6"}, (4, 4, 4), (4, 4, 4), True),           # kpts wins, with a warning
    ({"k_grid_density": 8}, None, None, False),                  # no k_grid with a density
])
def test_set_aims_k_grid(aims_par, kpts, expected, warns, recwarn):
    _set_aims_k_grid(aims_par, kpts)

    assert aims_par.get("k_grid") == expected
    assert any("kpts will take precedence" in str(w.message) for w in recwarn) == warns
