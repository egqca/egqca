import importlib.util
import sys
from pathlib import Path

import pytest

from egqca._ml_utils import _atomic_write
from egqca.ml import _append_ml_cluster_result, _read_completed_ids, collect_ml_results, reset_ml_clusters


def _run_folder(tmp_path: Path, results: dict[str, str]) -> Path:
    """Cluster tree with DFT/clusters.dat for three clusters and per-cluster result files."""
    (tmp_path / "DFT").mkdir()
    (tmp_path / "DFT" / "clusters.dat").write_text(
        "# compound= AB\n# total number of sites=2\n# permutable sites n=2\n"
        "S0_i0_w1\t0\t2\t0.0\t1\t-10.0\n"
        "S1_i0_w2\t1\t2\t0.5\t2\t-11.0\n"
        "S2_i0_w1\t2\t2\t1.0\t1\t-12.0\n"
    )
    (tmp_path / "mace" / "per_cluster").mkdir(parents=True)
    for cluster_id, text in results.items():
        _write_result(tmp_path / "mace", cluster_id, text)
    return tmp_path / "mace"


def _write_result(run_folder: Path, cluster_id: str, text: str) -> None:
    (run_folder / "per_cluster" / f"{cluster_id}_mace_ml_result.dat").write_text(text)


def _row(cluster_id: str, converged: bool) -> str:
    n_j = int(cluster_id[1])
    return f"# header\n{cluster_id}\t{n_j}\t2\t{n_j / 2}\t1\t-1{n_j}.0\t3\t3\t3\t90\t90\t90\t{converged}\t0.1\n"


def _rows(path: Path) -> dict[str, str]:
    return {l.split()[0]: l for l in path.read_text().splitlines() if l and not l.startswith("#")}


def _collect(run_folder: Path) -> tuple[Path, str]:
    clusters_path, _, log_path = collect_ml_results(run_folder, model="mace")
    return clusters_path, log_path.read_text()


def test_collect_requeues_unreadable_results_and_reports_stable_counts(tmp_path: Path):
    run_folder = _run_folder(tmp_path, {
        "S0_i0_w1": _row("S0_i0_w1", True),
        "S1_i0_w2": "# header only\n",                                            # job killed before the row
        "S2_i0_w1": "# header\nS2_i0_w1\t2\t2\t1.0\t1\t-12.0\t3\t3\t3\t90\t90\t90\tFal",   # cut off
    })

    # unreadable files count as missing, by cluster ID; a repeated collect reports the same
    for _ in range(2):
        with pytest.warns(UserWarning, match="treated as missing"):
            clusters_path, log = _collect(run_folder)
        assert list(_rows(clusters_path)) == ["S0_i0_w1"]
        assert (tmp_path / "missing_clusters.txt").read_text() == "S1, S1_i0_w2\nS2, S2_i0_w1\n"
        assert "Missing clusters:   2 (incl. 2 unreadable result files)" in log
        assert "Converged:          1\n" in log and "Not converged:      0\n" in log
        assert "  S1_i0_w2\n  S2_i0_w1\n" in log and "per_cluster\n" not in log

    # the reruns arrive: nothing missing, the stale list is removed
    _write_result(run_folder, "S1_i0_w2", _row("S1_i0_w2", False))
    _write_result(run_folder, "S2_i0_w1", _row("S2_i0_w1", True))
    clusters_path, log = _collect(run_folder)
    assert "Missing clusters:   0\n" in log
    assert "Converged:          2\n" in log and "Not converged:      1\n" in log
    assert "  S1_i0_w2  final_fmax: 0.1 eV/Å\n" in log
    assert not (tmp_path / "missing_clusters.txt").exists()


def test_collect_counts_rows_and_logs_without_per_cluster_file(tmp_path: Path):
    run_folder = _run_folder(tmp_path, {"S0_i0_w1": _row("S0_i0_w1", True)})
    # S1 and S2 were relaxed by run_ml_relaxation: rows and cluster logs, no per_cluster file
    clusters_path, volume_path, _ = collect_ml_results(run_folder, model="mace")
    with open(clusters_path, "a") as f:
        f.write("S1_i0_w2\t1\t2\t0.5\t2\t-11.0\t  # WARNING: Relaxation did not converge\n")
        f.write("S2_i0_w1\t2\t2\t1.0\t1\t-12.0\n")
    with open(volume_path, "a") as f:
        f.write("S1_i0_w2\t3\t3\t3\t90\t90\t90\nS2_i0_w1\t3\t3\t3\t90\t90\t90\n")
    log_dir = run_folder / "ML_logs" / "clusters"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "S2_i0_w1.log").write_text("WARNING: Final Bravais lattice 'ORC' is not compatible\nRuntime: 1.00 s\n")

    _, log = _collect(run_folder)

    assert "Recorded without per_cluster file (e.g. by run_ml_relaxation): 2\n" in log
    assert "Missing clusters:   0\n" in log
    assert "Converged:          2\n" in log and "Not converged:      1\n" in log
    assert "  S1_i0_w2  final_fmax: unavailable (no per_cluster file)\n" in log
    assert "Bravais lattice violations: 1/3\n" in log


def test_reset_non_converged_clusters_and_collect_again(tmp_path: Path, monkeypatch):
    run_folder = _run_folder(tmp_path, {
        "S0_i0_w1": _row("S0_i0_w1", True),
        "S1_i0_w2": _row("S1_i0_w2", False),
        "S2_i0_w1": _row("S2_i0_w1", True),
    })

    # 1. without collecting first; the model is detected from the result file names
    assert reset_ml_clusters(run_folder) == ["S1_i0_w2"]
    assert not (run_folder / "per_cluster" / "S1_i0_w2_mace_ml_result.dat").exists()

    # 2. collect --reset-nonconverged: rows, result file and DOS of the non-converged cluster go
    _write_result(run_folder, "S1_i0_w2", _row("S1_i0_w2", False))
    (run_folder / "phonons").mkdir()
    for cid in ("S0_i0_w1", "S1_i0_w2"):
        (run_folder / "phonons" / f"{cid}.dos").write_text("0 0\n")
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "collect_ml_results.py"
    spec = importlib.util.spec_from_file_location("collect", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", ["collect", str(run_folder), "--model", "mace", "--reset-nonconverged"])
    script.main()

    clusters_path, volume_path = run_folder / "clusters_mace.dat", run_folder / "volume_mace.dat"
    assert set(_rows(clusters_path)) == set(_rows(volume_path)) == {"S0_i0_w1", "S2_i0_w1"}
    assert not (run_folder / "phonons" / "S1_i0_w2.dos").exists()
    assert (run_folder / "phonons" / "S0_i0_w1.dos").exists()
    assert (tmp_path / "rerun_clusters.txt").read_text() == "S1, S1_i0_w2\n"

    # 3. the rerun converges: collect adds its row; nothing is left to rerun
    _write_result(run_folder, "S1_i0_w2", _row("S1_i0_w2", True))
    collect_ml_results(run_folder, model="mace")
    assert "did not converge" not in _rows(clusters_path)["S1_i0_w2"]
    assert reset_ml_clusters(run_folder, "mace") == []
    assert not (tmp_path / "rerun_clusters.txt").exists()


def test_killed_writes_leave_no_partial_results(tmp_path: Path):
    # a file rewritten atomically keeps its old content, and no temporary file is left
    path = tmp_path / "S0_i0_w1.dos"
    path.write_text("old\n")
    with pytest.raises(KeyboardInterrupt):
        with _atomic_write(path) as f:
            f.write("half of the new con")
            raise KeyboardInterrupt
    assert path.read_text() == "old\n"
    assert list(tmp_path.iterdir()) == [path]

    # a row cut off while appending is not counted, and the next append does not continue it
    clusters = tmp_path / "clusters_mace.dat"
    clusters.write_text("# n=2\nS0_i0_w1\t0\t2\t0.0\t1\t-10.0\nS1_i0_w2\t1\t2\t0.")
    assert _read_completed_ids(clusters) == {"S0_i0_w1"}
    _append_ml_cluster_result(clusters, "S1_i0_w2", -11.0, permutable_sites=2, Nat=2)
    rows = [l for l in clusters.read_text().splitlines() if not l.startswith("#")]
    assert [r.split("\t")[0] for r in rows] == ["S0_i0_w1", "S1_i0_w2"]
    assert rows[1].split("\t")[5] == "-11.0"
    assert _read_completed_ids(clusters) == {"S0_i0_w1", "S1_i0_w2"}
