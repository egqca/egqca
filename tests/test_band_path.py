import numpy as np
from pathlib import Path

from ase.build import bulk

from egqca.create_dft_inputs import (
    aims_band_output_lines,
    cluster_bandpath,
    reference_bandpath,
    write_qe_bands,
)


def _clusters():
    """Unrelaxed bcc 2x2x2 reference and a slightly sheared 'relaxed' cluster."""
    ref = bulk("Nb", "bcc", a=3.3, cubic=True).repeat((2, 2, 2))
    relaxed = ref.copy()
    relaxed.set_cell(np.array(ref.cell) + [[0, 0.02, 0], [0, 0, 0], [0, 0, 0.03]], scale_atoms=True)
    return ref, relaxed


def test_qe_bands_inputs_have_the_k_points_of_the_reference(tmp_path: Path):
    ref, relaxed = _clusters()
    (tmp_path / "pseudos").mkdir()
    (tmp_path / "pseudos" / "Nb.upf").write_text("")
    # the relaxed cluster's own Bravais lattice, and so its own path, differs
    assert relaxed.cell.get_bravais_lattice().name != ref.cell.get_bravais_lattice().name
    bands = reference_bandpath(ref.cell, "", 5)
    assert bands["path"] == "GXMGRX,MR"
    assert np.allclose(cluster_bandpath(ref, bands).kpts, cluster_bandpath(relaxed, bands).kpts)
    k_blocks = []
    for name, atoms in (("S0_i0_w1", ref), ("S1_i0_w1", relaxed)):
        qe_par = {"pseudo_dir": str(tmp_path / "pseudos"), "control": {}, "system": {"ecutwfc": 40}}
        write_qe_bands(atoms, tmp_path / name, name, qe_par=qe_par, bands=bands)
        text = (tmp_path / name / "bands.in").read_text()
        k_blocks.append(text[text.index("K_POINTS"):].split("\n\n")[0])
    assert k_blocks[0] == k_blocks[1]


def test_aims_segments_handle_path_jumps():
    ref, relaxed = _clusters()
    bands = reference_bandpath(ref.cell, "", 5)
    lines = [aims_band_output_lines({}, cluster_bandpath(a, bands), bands["segment_npoints"])["output"]
             for a in (ref, relaxed)]

    labels = [tuple(l.split()[-2:]) for l in lines[0]]
    assert labels == [("G", "X"), ("X", "M"), ("M", "G"), ("G", "R"), ("R", "X"), ("M", "R")]
    assert lines[0] == lines[1]
