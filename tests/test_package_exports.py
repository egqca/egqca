import egqca


def test_public_exports():
    # also the optional ML exports, which fall back to None without the ML dependencies
    assert {"EGQCA", "set_up_clusters_and_dft", "geometry_relaxation"} <= set(egqca.__all__)
    assert [name for name in egqca.__all__ if not hasattr(egqca, name)] == []
