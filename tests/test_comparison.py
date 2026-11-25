import importlib
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from egqca import (
    discover_egqca_inputs,
    make_egqca_inputs,
    plot_average_properties_comparison,
    plot_occurrence_probability_difference_from_ideal,
    plot_thermodynamic_comparison,
    run_egqca_comparison,
)
from egqca.comparison import _selected_property_indices


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# dummy\n")


def test_discover_egqca_inputs(tmp_path: Path):
    # one run folder per model; a cluster file without volume file is skipped
    _touch(tmp_path / "mace" / "clusters_mace.dat", tmp_path / "mace" / "volume_mace.dat",
           tmp_path / "orphan" / "clusters_orphan.dat")
    inputs = discover_egqca_inputs(tmp_path)
    assert [(i.cluster_file, i.volume_file, i.label) for i in inputs] == [
        (tmp_path / "mace" / "clusters_mace.dat", tmp_path / "mace" / "volume_mace.dat", "mace")
    ]

    # equal default labels get their folder prefixed
    _touch(*(tmp_path / "dft" / run / name for run in ("A", "B") for name in ("clusters.dat", "volume.dat")))
    _touch(*(tmp_path / "ml" / run / f"{name}_mace.dat" for run in ("x", "y") for name in ("clusters", "volume")))
    assert [i.label for i in discover_egqca_inputs(tmp_path / "dft")] == ["A_clusters", "B_clusters"]
    assert [i.label for i in discover_egqca_inputs(tmp_path / "ml")] == ["x_mace", "y_mace"]


def test_make_egqca_inputs_labels(tmp_path: Path):
    a, b = tmp_path / "A" / "clusters.dat", tmp_path / "B" / "clusters.dat"
    deep = [tmp_path / run / "A" / "clusters.dat" for run in ("run1", "run2")]
    _touch(a, b, *deep)
    labels = lambda inputs: [i.label for i in inputs]

    explicit = make_egqca_inputs([(tmp_path / "other" / "clusters_pet.dat", tmp_path / "v" / "volume_pet.dat")],
                                 labels=["PET custom"])
    assert (explicit[0].volume_file, explicit[0].label) == (tmp_path / "v" / "volume_pet.dat", "PET custom")
    # equal default labels: folder prefix, as many folders as needed, a number within one folder
    assert labels(make_egqca_inputs([(a, None), (b, None)])) == ["A_clusters", "B_clusters"]
    assert labels(make_egqca_inputs([(f, None) for f in deep])) == ["run1_A_clusters", "run2_A_clusters"]
    assert labels(make_egqca_inputs([(a, None), (a, None)])) == ["clusters", "clusters_2"]
    # a default label colliding with a user label is prefixed; equal user labels are refused
    assert labels(make_egqca_inputs([(a, None, "clusters"), (b, None)])) == ["clusters", "B_clusters"]
    with pytest.raises(ValueError, match="'PET' for .*A.*B"):
        make_egqca_inputs([(a, None), (b, None)], labels=["PET", "PET"])


def test_average_properties_comparison_matches_property_files_by_position(tmp_path: Path, monkeypatch):
    comparison = importlib.import_module("egqca.comparison")
    read = []
    monkeypatch.setattr(
        comparison, "read_properties",
        lambda model, path: (read.append((model.name, Path(path).name)) or (np.ones((1, 1)), ["p"], ["S0_i0_w1"])),
    )

    def model(name):
        return SimpleNamespace(
            name=name, comparison_label="clusters", Tlist=[600.0], cluster_id=np.array(["S0_i0_w1"]),
            x_plot=np.array([0.5]), gqca_mean=lambda prop, T: np.array([1.0]),
            gqca_std=lambda prop, T: np.array([0.0]), xlabel_compound=lambda: "x",
        )

    models = [model("A"), model("B")]
    plot_average_properties_comparison(models, [tmp_path / "a.dat", tmp_path / "b.dat"], show=False)
    assert read == [("A", "a.dat"), ("B", "b.dat")]

    with pytest.raises(ValueError, match="share the label"):
        plot_average_properties_comparison(models, {"clusters": tmp_path / "a.dat"}, show=False)
    models[1].comparison_label = "other"
    with pytest.raises(ValueError, match="no entry for the model label"):
        plot_average_properties_comparison(models, {"clusters": tmp_path / "a.dat"}, show=False)
    plt.close("all")


def test_comparison_of_two_entropy_approximations(alloy):
    clusters_file, volume_file = alloy
    inputs = make_egqca_inputs([
        {"cluster_file": clusters_file, "volume_file": volume_file},
        {"cluster_file": clusters_file, "volume_file": volume_file, "entropy_approximation": "non-overlapping"},
    ])
    assert [item.label for item in inputs] == ["clusters", "clusters_non-overlapping"]

    models = run_egqca_comparison(inputs, x=np.array([0.12, 0.48]), Tlist=[600.0], run_miscibility_gap=False)
    assert [m.entropy_approximation for m in models] == ["overlapping", "non-overlapping"]
    assert not np.allclose(models[0].F[600.0], models[1].F[600.0])

    # "Fv" is skipped without phonon data, so ask for the configurational quantities
    thermo_figs = plot_thermodynamic_comparison(models, quantities=("U", "F", "S"), show=False)
    assert set(thermo_figs) >= {"U", "F", "S"}
    property_figs = plot_average_properties_comparison(models, [volume_file] * 2, columns=["a", "c"], show=False)
    assert len(property_figs[600.0].axes[0].lines) == 4      # 2 models x 2 columns
    plt.close("all")

    # a generator is fine too, although the start-up summary reads the inputs first
    models = run_egqca_comparison(
        (item for item in inputs[:1]), x=np.array([0.12]), Tlist=[600.0], run_miscibility_gap=False,
        entropy_approximation="non-overlapping",
    )
    assert models[0].entropy_approximation == "non-overlapping"


def test_plot_occurrence_probability_difference_from_ideal():
    model = SimpleNamespace(
        comparison_label="reference",
        Tlist=[600.0],
        x=np.array([0.25, 0.75]),
        x_plot=np.array([0.25, 0.75]),
        cluster_id=np.array(["S0_i0_w1", "S1_i0_w1"]),
        pj={600.0: np.array([[0.8, 0.2], [0.25, 0.75]])},
        pj0=np.array([[0.75, 0.25], [0.25, 0.75]]),
        xlabel_compound=lambda: "x",
    )

    figures = plot_occurrence_probability_difference_from_ideal([model], show=False)

    ax = figures[600.0].axes[0]
    assert len(ax.lines) == len(model.cluster_id) + 1
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.05, 0.0])
    plt.close("all")


def test_short_property_names_select_only_unique_columns():
    names = ["E_g(eV)", "DOS(E_F+0eV)", "DOS(E_F+0.1eV)"]

    assert _selected_property_indices(names, ["E_g", "DOS(E_F+0.1eV)", 1]) == [0, 2, 1]
    with pytest.raises(KeyError, match="several"):
        _selected_property_indices(names, ["DOS"])
    with pytest.raises(KeyError, match="none"):
        _selected_property_indices(names, ["N"])
