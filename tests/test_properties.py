from types import SimpleNamespace

import numpy as np
import pytest

from egqca import EGQCA
from egqca.properties import _output_names, normalize_probabilities

# class n_j = 1 has three clusters; x_j = n_j / 2
MODEL = SimpleNamespace(
    cluster_id=np.array(["S0_i0_w1", "S1_i0_w1", "S1_i1_w1", "S1_i2_w1", "S2_i0_w1"]),
    nj=np.array([0, 1, 1, 1, 2]),
    pj={300.0: np.array([[0.1, 0.4, 0.2, 0.1, 0.2],
                         [0.3, 0.5, 0.0, 0.0, 0.2]])},   # (x, cluster); 2nd x: kept n_j = 1 clusters have no weight
)


def test_normalize_probabilities_renormalizes_each_class():
    # n_j = 1 loses its most probable cluster (index 1); selection given unsorted
    norm_probs, idx_sorted = normalize_probabilities(MODEL, 300.0, [4, 3, 0, 2])

    assert idx_sorted.tolist() == [0, 2, 3, 4]      # sorted by cluster ID
    np.testing.assert_allclose(norm_probs.sum(axis=1), 1.0)
    # class weight 0.7 onto 0.2 : 0.1, ratio 2 : 1 kept
    np.testing.assert_allclose(norm_probs[0], [0.1, 0.7 * 2 / 3, 0.7 / 3, 0.2])
    # no weight on the kept clusters: the class weight 0.5 is split equally
    np.testing.assert_allclose(norm_probs[1], [0.3, 0.25, 0.25, 0.2])
    # the average composition is unchanged
    xj = MODEL.nj / 2
    np.testing.assert_allclose(norm_probs @ xj[idx_sorted], MODEL.pj[300.0] @ xj)


@pytest.mark.parametrize("selection, error", [
    ([0, 1], r"no cluster with n_j = \[2\]"),
    ([0, 1, 1, 4], "more than once: S1_i0_w1"),
])
def test_normalize_probabilities_rejects_invalid_selections(selection, error):
    with pytest.raises(ValueError, match=error):
        normalize_probabilities(MODEL, 300.0, selection)


def test_surrogate_model_file_rejects_duplicate_ids(tmp_path):
    surrogate = tmp_path / "surrogate_T300.0_3clusters.dat"
    surrogate.write_text("# ID\nS0_i0_w1\nS1_i0_w1\nS1_i0_w1\nS2_i0_w1\n")

    with pytest.raises(ValueError, match="more than once: S1_i0_w1"):
        EGQCA._read_surrogate_model_indices(SimpleNamespace(cluster_id=MODEL.cluster_id), surrogate)


def test_output_names():
    names = _output_names(["E_g(eV)", "DOS(E_F+0eV)", "DOS(E_F+0.1eV)", "s(E_F+0eV)", r"\Delta V/V_0"])
    # unique short names are kept, shared ones keep their argument
    assert names == {
        "E_g(eV)": "E_g",
        "DOS(E_F+0eV)": "DOS_E_F+0eV",
        "DOS(E_F+0.1eV)": "DOS_E_F+0.1eV",
        "s(E_F+0eV)": "s",
        r"\Delta V/V_0": "Delta_V",
    }
    with pytest.raises(ValueError, match="distinct output file names"):
        _output_names(["a(b c)", "a(b_c)"])
