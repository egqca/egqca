"""clusters_template.dat in the clusters folder and the run folders built from it."""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import egqca.workflow as workflow
from egqca.workflow import (
    CLUSTERS_TEMPLATE_NAME,
    extract_eddp_energies,
    set_up_clusters_dat,
    set_up_volume_dat,
    update_DFT_in,
    write_properties_file,
)

CLUSTER_IDS = ("S0_i0_w1", "S1_i0_w2", "S2_i0_w1")


def _tree(tmp_path: Path, template: bool = True) -> Path:
    """Clusters folder with S0..S2 and, optionally, its clusters template (n=2)."""
    for cid in CLUSTER_IDS:
        (tmp_path / cid[:2] / cid).mkdir(parents=True)
    if template:
        set_up_clusters_dat(tmp_path, 2, 4, 4, "A$_{1-x}$B$_x$")
    return tmp_path


def test_template_lists_every_cluster_with_missing_energies(tmp_path: Path):
    path = set_up_clusters_dat(_tree(tmp_path, template=False), 2, 4, 4, "A$_{1-x}$B$_x$")

    assert path == tmp_path / CLUSTERS_TEMPLATE_NAME
    text = path.read_text()
    assert "# permutable sites n=2" in text and "# total number of sites=4" in text
    assert "S1_i0_w2\t1\t4\t0.5\t2\tmissing\n" in text
    assert not (tmp_path / "DFT").exists()


@pytest.mark.parametrize("create_dft_folders, qe_par, dft_run_file", [
    (False, None, False),                  # ML-only setup: no DFT run folder
    (True, None, True),
    (False, {"pseudo_dir": "pp"}, True),
])
def test_setup_writes_the_template_and_the_dft_run_file_only_for_dft(
        tmp_path: Path, monkeypatch, create_dft_folders, qe_par, dft_run_file):
    out_dir = tmp_path / "clusters"
    cif = tmp_path / "parent.cif"
    cif.write_text("not read")
    monkeypatch.setattr(workflow, "get_elements", lambda *a: (["A", "B"], ["A", "B"], 4, 2, "AB"))
    monkeypatch.setattr(workflow, "run_supercell", lambda *a, **k: _tree(out_dir, template=False))
    monkeypatch.setattr(workflow, "create_DFT_in", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_nat_from_cif", lambda *a: 4)

    workflow.set_up_clusters_and_dft("supercell", str(cif), "1x1x1", str(out_dir),
                                     qe_par=qe_par, create_dft_folders=create_dft_folders)

    template = out_dir / CLUSTERS_TEMPLATE_NAME
    assert template.is_file()
    assert (out_dir / "DFT" / "clusters.dat").is_file() == dft_run_file
    if dft_run_file:
        assert (out_dir / "DFT" / "clusters.dat").read_text() == template.read_text()


@pytest.mark.parametrize("dft_folder_name, folder", [(None, "DFT"), ("DFT_PBE", "DFT_PBE")])
def test_update_dft_in_creates_the_dft_run_file_once(tmp_path: Path, monkeypatch, dft_folder_name, folder):
    tree = _tree(tmp_path)
    monkeypatch.setattr(workflow, "create_DFT_in", lambda **k: None)

    update_DFT_in(tree, dft_folder_name=dft_folder_name)
    run_file = tree / folder / "clusters.dat"
    assert run_file.read_text() == (tree / CLUSTERS_TEMPLATE_NAME).read_text()

    # energies already extracted are never overwritten
    run_file.write_text(run_file.read_text().replace("missing", "-1.0"))
    update_DFT_in(tree, dft_folder_name=dft_folder_name)
    assert "missing" not in run_file.read_text()


def test_set_up_volume_dat_writes_into_the_run_folder(tmp_path: Path, monkeypatch):
    tree = _tree(tmp_path)
    monkeypatch.setattr(workflow, "determine_dft_solver", lambda path: "vasp")
    monkeypatch.setattr(workflow, "read", lambda *a, **k: Atoms(cell=np.diag([2.0, 3.0, 4.0]), pbc=True))

    default = set_up_volume_dat(tree, "OUTCAR")[0]
    custom = set_up_volume_dat(tree, "OUTCAR", run_folder=tree / "DFT_PBE")[0]

    assert default == tree / "DFT" / "volume.dat"
    assert custom == tree / "DFT_PBE" / "volume.dat"
    assert "S1_i0_w2\t2.0\t3.0\t4.0\t90.0\t90.0\t90.0" in custom.read_text()
    with pytest.raises(ValueError, match="dft_out_file is required"):
        set_up_volume_dat(tree)


def test_write_properties_file_writes_into_the_clusters_folder(tmp_path: Path):
    tree = _tree(tmp_path)

    path = write_properties_file(tree, {"p": lambda cluster_dir: len(cluster_dir.name)})

    assert path == tree / "properties.dat"
    assert "S1_i0_w2\t8" in path.read_text()
    with pytest.raises(ValueError, match="No S\\*/S\\*_i\\*_w\\* cluster folders"):
        write_properties_file(tree / "DFT", {"p": lambda cluster_dir: 0})


def test_extract_eddp_energies_reads_the_setup_from_the_template(tmp_path: Path):
    tree = _tree(tmp_path)
    for cid in CLUSTER_IDS:
        (tree / cid[:2] / cid / "eddp").mkdir()
        (tree / cid[:2] / cid / "eddp" / "energy.dat").write_text("converged in 3 steps\nEnergy: -13.6\n")

    clusters_file = extract_eddp_energies(tree)

    text = clusters_file.read_text()
    assert "# permutable sites n=2" in text and "compound=A$_{1-x}$B$_x$" in text
    assert "S1_i0_w2\t1\t4\t0.5\t2\t" in text
    with pytest.raises(ValueError, match="sites=8 differs from 4"):
        extract_eddp_energies(tree, sites=8)
