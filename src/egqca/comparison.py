from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from ._banner import describe_grid, entry_point
from .egqca_model import EGQCA, cluster_number_markers, label_cluster_curves
from .io import read_properties
from .properties import _short_property_name, normalize_probabilities


@dataclass(frozen=True)
class EGQCAInput:
    """Input file pair used for comparing several EGQCA calculations.

    ``entropy_approximation`` ("overlapping" or "non-overlapping") overrides the
    default of :func:`run_egqca_comparison` for this input, so the same cluster
    file can be compared under both approximations. ``phonon_file_name`` is passed
    to :class:`EGQCA` to include the vibrational free energy of this input.
    """

    cluster_file: Path
    volume_file: Path | None = None
    label: str | None = None
    entropy_approximation: str | None = None
    phonon_file_name: str | None = None


def _suffix_from_cluster_file(cluster_file: Path) -> str:
    stem = cluster_file.stem
    if stem == "clusters":
        return ""
    if stem.startswith("clusters_"):
        return stem.removeprefix("clusters_")
    if stem.startswith("clusters"):
        return stem.removeprefix("clusters").lstrip("_-")
    return stem


def _default_label(cluster_file: Path, volume_file: Path | None = None) -> str:
    suffix = _suffix_from_cluster_file(cluster_file)
    if suffix:
        return suffix
    if volume_file is not None and cluster_file.parent != volume_file.parent:
        return cluster_file.parent.name
    return cluster_file.stem


def _matching_volume_file(cluster_file: Path) -> Path:
    suffix = _suffix_from_cluster_file(cluster_file)
    if suffix:
        return cluster_file.with_name(f"volume_{suffix}.dat")
    return cluster_file.with_name("volume.dat")


def _folder_prefix(cluster_file: Path, depth: int) -> str:
    """Names of the ``depth`` folders above ``cluster_file``, joined by ``_``."""
    parts = Path(cluster_file).resolve().parent.parts[1:]
    return "_".join(parts[-depth:]) if depth else ""


def _unique_labels(items: list[tuple[EGQCAInput, bool]]) -> list[EGQCAInput]:
    """Make the labels of comparison inputs unique.

    ``items`` pairs each input with whether its label was set by the user. Equal
    user labels raise a ValueError. A default label that collides with another label
    gets the names of its parent folders prefixed (``A_clusters``, ``B_clusters``),
    as many as are needed to tell the folders apart; inputs from one folder get a
    number appended instead (``clusters_2``).
    """
    inputs = [item for item, _ in items]
    explicit = [is_explicit for _, is_explicit in items]

    explicit_counts = Counter(item.label for item, e in zip(inputs, explicit) if e)
    duplicates = sorted(label for label, count in explicit_counts.items() if count > 1)
    if duplicates:
        files = {
            label: [str(item.cluster_file) for item, e in zip(inputs, explicit) if e and item.label == label]
            for label in duplicates
        }
        raise ValueError(
            "Comparison labels must be unique, but these labels are used more than once: "
            + "; ".join(f"{label!r} for {', '.join(paths)}" for label, paths in files.items())
        )

    labels = [item.label for item in inputs]
    counts = Counter(labels)
    for label in [l for l, c in counts.items() if c > 1]:
        group = [i for i, l in enumerate(labels) if l == label and not explicit[i]]
        folders = {Path(inputs[i].cluster_file).resolve().parent for i in group}
        clashes_with_explicit = any(e and l == label for l, e in zip(labels, explicit))
        if not group or (len(folders) < 2 and not clashes_with_explicit):
            continue
        max_depth = max(len(f.parts) - 1 for f in folders)
        depth = next(
            (d for d in range(1, max_depth + 1)
             if len({_folder_prefix(inputs[i].cluster_file, d) for i in group}) == len(folders)),
            max_depth,
        )
        for i in group:
            labels[i] = f"{_folder_prefix(inputs[i].cluster_file, depth)}_{label}"

    # same folder (or still equal): number the later ones
    used = {l for l, e in zip(labels, explicit) if e}
    for i, label in enumerate(labels):
        if explicit[i]:
            continue
        new, k = label, 2
        while new in used:
            new, k = f"{label}_{k}", k + 1
        labels[i] = new
        used.add(new)

    return [replace(item, label=label) for item, label in zip(inputs, labels)]


def discover_egqca_inputs(
    folder: str | Path,
    *,
    cluster_pattern: str = "*/clusters*.dat",
    labels: dict[str | Path, str] | None = None,
    require_volume: bool = True,
) -> list[EGQCAInput]:
    """Find ``clusters*.dat`` files and matching ``volume*.dat`` files in one folder.

    Labels default to the cluster-file suffix, e.g. ``clusters_mace.dat`` becomes
    ``mace``. ``labels`` may map either a full cluster path or a filename to a
    custom label. Default labels that are equal get their folder prefixed
    (``A/clusters.dat``, ``B/clusters.dat`` -> ``A_clusters``, ``B_clusters``);
    equal custom labels raise a ValueError.
    """

    folder = Path(folder)
    label_map = {str(key): value for key, value in (labels or {}).items()}
    items = []
    for cluster_file in sorted(folder.glob(cluster_pattern)):
        if not cluster_file.is_file():
            continue
        volume_file = _matching_volume_file(cluster_file)
        if not volume_file.exists():
            if require_volume:
                continue
            volume_file = None
        label = label_map.get(str(cluster_file)) or label_map.get(cluster_file.name)
        explicit = label is not None
        if not explicit:
            label = _default_label(cluster_file, volume_file)
        items.append((EGQCAInput(cluster_file=cluster_file, volume_file=volume_file, label=label), explicit))
    return _unique_labels(items)


def make_egqca_inputs(
    pairs: Iterable[
        EGQCAInput
        | tuple[str | Path, str | Path | None]
        | tuple[str | Path, str | Path | None, str]
        | dict[str, str | Path | None]
    ],
    *,
    labels: Iterable[str] | None = None,
) -> list[EGQCAInput]:
    """Create comparison inputs from explicitly provided cluster/volume paths.

    Each entry is an :class:`EGQCAInput`, a ``(cluster_file, volume_file[, label])``
    tuple or a dict with the keys ``cluster_file``, ``volume_file``, ``label``,
    ``entropy_approximation`` and ``phonon_file_name`` (only ``cluster_file`` is
    required). An input with
    an ``entropy_approximation`` but no label gets it appended to the default
    label, e.g. ``clusters_non-overlapping``. Default labels that are equal get
    their folder prefixed (``A_clusters``, ``B_clusters``), or a number appended if
    they come from one folder (``clusters_2``); equal custom labels raise a
    ValueError.
    """

    label_list = list(labels) if labels is not None else None
    items = []
    for idx, pair in enumerate(pairs):
        if isinstance(pair, EGQCAInput):
            item = pair
        elif isinstance(pair, dict):
            cluster_file = Path(pair["cluster_file"])
            volume_value = pair.get("volume_file")
            volume_file = Path(volume_value) if volume_value is not None else None
            item = EGQCAInput(
                cluster_file=cluster_file,
                volume_file=volume_file,
                label=pair.get("label"),
                entropy_approximation=pair.get("entropy_approximation"),
                phonon_file_name=pair.get("phonon_file_name"),
            )
        else:
            cluster_file = Path(pair[0])
            volume_file = Path(pair[1]) if pair[1] is not None else None
            item_label = pair[2] if len(pair) > 2 else None
            item = EGQCAInput(cluster_file=cluster_file, volume_file=volume_file, label=item_label)

        override_label = label_list[idx] if label_list is not None else None
        label = override_label or item.label
        explicit = label is not None
        if not explicit:
            label = _default_label(item.cluster_file, item.volume_file)
            if item.entropy_approximation is not None:
                label = f"{label}_{item.entropy_approximation}"
        items.append((replace(item, label=label), explicit))
    return _unique_labels(items)


def _comparison_summary(b):
    b.arguments["inputs"] = list(b.arguments["inputs"])
    return [
        *[(f"Input {i + 1}", f"{item.label}: {item.cluster_file}"
           f" [{item.entropy_approximation or b.arguments['entropy_approximation']}]")
          for i, item in enumerate(b.arguments["inputs"])],
        ("Composition x", describe_grid(b.arguments["x"])),
        ("Temperatures", describe_grid(b.arguments["Tlist"], "K")),
        ("Pressure (GPa)", b.arguments["pressure_GPa"]),
        ("Miscibility gap", "yes" if b.arguments["run_miscibility_gap"] else "no"),
    ]


@entry_point("EGQCA comparison", _comparison_summary)
def run_egqca_comparison(
    inputs: Iterable[EGQCAInput],
    x: np.ndarray,
    Tlist: list[float],
    *,
    run_miscibility_gap: bool = True,
    T_min_misc_gap: float = 30,
    misc_gap_max_iter: int = 200,
    pressure_GPa: float = 0.0,
    recompute_output_subfolder: str | None = None,
    entropy_approximation: str = "overlapping",
) -> list[EGQCA]:
    """Instantiate and recompute EGQCA models without saving comparison figures.

    ``entropy_approximation`` is used for every input that does not set its own
    (see :class:`EGQCAInput`).
    """

    inputs = list(inputs)
    labelled = _unique_labels([
        (item if item.label is not None
         else replace(item, label=_default_label(item.cluster_file, item.volume_file)),
         item.label is not None)
        for item in inputs
    ])
    models = []
    for item, labelled_item in zip(inputs, labelled):
        model = EGQCA(
            datafile=item.cluster_file,
            volume_file=item.volume_file,
            phonon_file_name=item.phonon_file_name,
            out_folder_name=item.label,
            pressure_GPa=pressure_GPa,
            entropy_approximation=item.entropy_approximation or entropy_approximation,
        )
        model.comparison_label = labelled_item.label
        x_prepared = model._prepare_main(x, Tlist, output_subfolder=recompute_output_subfolder)
        model.init_thermo()
        model._ensure_probabilities(x_prepared, Tlist)
        for T in model.Tlist:
            model.thermo(T)
        if run_miscibility_gap:
            model.calc_miscibility_gap(
                x_prepared,
                T_min=T_min_misc_gap,
                dx_plot=0.01,
                max_iter=misc_gap_max_iter,
            )
        models.append(model)
    return models


def _model_label(model: EGQCA) -> str:
    return getattr(model, "comparison_label", None) or model.datafile.stem


def _temperatures(models: list[EGQCA], temperatures: Iterable[float] | None) -> list[float]:
    if temperatures is not None:
        return list(np.asarray(list(temperatures), dtype=float))
    return list(np.asarray(models[0].Tlist, dtype=float))


def plot_thermodynamic_comparison(
    models: list[EGQCA],
    *,
    quantities: Iterable[str] = ("U", "F", "Fv"),
    temperatures: Iterable[float] | None = None,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """Plot thermodynamic quantities for several EGQCA models.

    Each quantity gets one figure with one subplot per temperature.
    Values are shown in meV/site.
    """

    quantity_info = {
        "U": (r"$\Delta H$", "Mixing enthalpy"),
        "F": (r"$\Delta G$", "Mixing free energy"),
        "S": (r"$T\Delta S$", "Mixing entropy term"),
        "Fv": (r"$\Delta F_{vib}$", "Vibrational free energy"),
    }
    Ts = _temperatures(models, temperatures)
    figures = {}
    for quantity in quantities:
        if quantity == "Fv" and not any(model.phonon_data is not None for model in models):
            continue
        ylabel, title = quantity_info[quantity]
        fig, axes = plt.subplots(1, len(Ts), figsize=(6 * len(Ts), 5), squeeze=False)
        for ax, T in zip(axes[0], Ts):
            for model in models:
                values_by_T = getattr(model, quantity)
                if T not in values_by_T:
                    continue
                ax.plot(model.x_plot, values_by_T[T], label=_model_label(model), linewidth=2)
            ax.set_title(f"{title}, T = {T:g} K")
            ax.set_xlabel(model.xlabel_compound())
            ax.set_ylabel(f"{ylabel} [meV/site]")
            ax.legend()
        fig.tight_layout()
        figures[quantity] = fig
    if show:
        plt.show()
    return figures


def _write_cluster_numbers(models: list[EGQCA]) -> None:
    """Key for ``enumerate_clusters``: cluster_numbers.dat in each model's output folder."""
    for model in models:
        model.write_cluster_numbers(model.out_dir)


def plot_excess_enthalpy_comparison(models: list[EGQCA], *, show: bool = True,
                                    enumerate_clusters: bool = False) -> plt.Figure:
    """Plot cluster excess enthalpies for several EGQCA models in one axes.

    ``enumerate_clusters`` draws the numbers 1, 2, ... (clusters.dat order) as markers and
    writes the key ``cluster_numbers.dat`` into each model's output folder.
    """

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in models:
        # same x scale as all other plots: x = 0 ... composition number of the compound
        x_cluster = model.xj * model.extract_composition_number(model.compound)
        if enumerate_clusters:
            # the cluster numbers are the markers, in the model's color
            (handle,) = ax.plot([], [], marker=r"$\#$", linestyle=" ", label=_model_label(model))
            cluster_number_markers(ax, x_cluster, model.delta_j, handle.get_color())
            ax.update_datalim(np.column_stack((x_cluster, model.delta_j)))
            ax.autoscale_view()
        else:
            ax.plot(
                x_cluster,
                model.delta_j,
                marker="o",
                linestyle=" ",
                alpha=0.75,
                label=_model_label(model),
            )
    if enumerate_clusters:
        _write_cluster_numbers(models)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(models[0].xlabel_compound())
    ax.set_ylabel(r"$\Delta_j$ [meV/site]")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_miscibility_gap_comparison(models: list[EGQCA], *, show: bool = True) -> plt.Figure:
    """Plot spinodal and binodal lines from several recomputed models.

    Models whose scan did not converge onto a critical point are still drawn when
    they have lines, marked "(no T_c)" in the legend.
    """

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in models:
        label = _model_label(model)
        if getattr(model, "T_c_miscibility_gap", None) is None:
            label += " (no T_c)"
        if getattr(model, "binodal_line", np.empty(0)).size:
            ax.plot(model.binodal_line[:, 1], model.binodal_line[:, 0], linestyle="--", label=f"{label} binodal")
        if getattr(model, "spinodal_line", np.empty(0)).size:
            ax.plot(model.spinodal_line[:, 1], model.spinodal_line[:, 0], linestyle="-", label=f"{label} spinodal")
    ax.set_xlabel(models[0].xlabel_compound())
    ax.set_ylabel("T [K]")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_occurrence_probabilities_comparison(
    models: list[EGQCA],
    *,
    x_values: Iterable[float] | None = None,
    show: bool = True,
    enumerate_clusters: bool = False,
) -> dict[float, plt.Figure]:
    """Plot occurrence probabilities vs temperature with one subplot per model.

    ``enumerate_clusters`` writes the cluster number on each curve, near the high-T end, see
    :func:`plot_excess_enthalpy_comparison`.
    """

    if x_values is None:
        x = models[0].x
        x_values = x[np.unique([len(x) // 4, len(x) // 2, 3 * len(x) // 4])]
    figures = {}
    for x_value in x_values:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
        for ax, model in zip(axes[0], models):
            x_idx = np.abs(model.x - x_value).argmin()
            lines = []
            for idx_cluster in range(len(model.cluster_id)):
                probabilities = [model.pj[T][x_idx, idx_cluster] for T in model.Tlist]
                lines += ax.plot(model.Tlist, probabilities, linewidth=1)
            if enumerate_clusters:
                T_label = model.Tlist[0] + 0.9 * (model.Tlist[-1] - model.Tlist[0])
                label_cluster_curves(ax, lines, np.full(len(lines), T_label))
            ax.set_title(_model_label(model))
            ax.set_xlabel(f"T [K] at x = {model.x[x_idx]:.3g}")
            ax.set_ylabel(r"$p_j$")
        fig.tight_layout()
        figures[float(x_value)] = fig
    if enumerate_clusters:
        _write_cluster_numbers(models)
    if show:
        plt.show()
    return figures


def plot_occurrence_probability_difference_from_ideal(
    models: list[EGQCA],
    *,
    temperatures: Iterable[float] | None = None,
    show: bool = True,
    enumerate_clusters: bool = False,
) -> dict[float, plt.Figure]:
    """Plot ``p_j - p_j^ideal`` versus composition for all clusters.

    ``enumerate_clusters`` writes the cluster number at the largest deviation of each
    curve, see :func:`plot_excess_enthalpy_comparison`.

    ``p_j^ideal`` is the ideal solid-solution, or a priori, cluster probability.
    One figure is created per temperature. If several comparison models are
    provided, each model gets one subplot and all clusters are overlaid.
    """

    Ts = _temperatures(models, temperatures)
    figures = {}
    for T in Ts:
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(6 * len(models), 5),
            squeeze=False,
        )
        for ax, model in zip(axes[0], models):
            if T not in model.pj:
                continue
            if getattr(model, "pj0", None) is None or np.shape(model.pj0) != np.shape(
                model.pj[T]
            ):
                model.apriori(model.x)

            probability_difference = model.pj[T] - model.pj0
            lines = []
            for idx_cluster in range(len(model.cluster_id)):
                lines += ax.plot(
                    model.x_plot,
                    probability_difference[:, idx_cluster],
                    linewidth=1,
                    alpha=0.75,
                )
            if enumerate_clusters:
                i_max = np.abs(probability_difference).argmax(axis=0)
                label_cluster_curves(ax, lines, model.x_plot[i_max])
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{_model_label(model)}, T = {T:g} K")
            ax.set_xlabel(model.xlabel_compound())
            ax.set_ylabel(r"$p_j - p_j^{ideal}$")
        fig.tight_layout()
        figures[float(T)] = fig
    if enumerate_clusters:
        _write_cluster_numbers(models)
    if show:
        plt.show()
    return figures


def _selected_property_indices(property_names: list[str], columns: Iterable[int | str] | None) -> list[int]:
    if columns is None:
        return list(range(len(property_names)))
    indices = []
    for column in columns:
        if isinstance(column, int):
            indices.append(column)
        elif column in property_names:
            indices.append(property_names.index(column))
        else:
            # a short name (label without unit/argument) only selects a column if it is unique
            matches = [idx for idx, name in enumerate(property_names) if _short_property_name(name) == column]
            if len(matches) != 1:
                found = [property_names[idx] for idx in matches] if matches else property_names
                raise KeyError(
                    f"Property {column!r} matches {'several' if matches else 'none'} of the columns "
                    f"{found}. Select it by its full label or column index."
                )
            indices.append(matches[0])
    return indices


def plot_average_properties_comparison(
    models: list[EGQCA],
    property_files: Iterable[str | Path] | dict[str, str | Path],
    *,
    columns: Iterable[int | str] | None = None,
    temperatures: Iterable[float] | None = None,
    show_std: bool = False,
    show: bool = True,
) -> dict[float, plt.Figure]:
    """Average selected property columns and overlay them in one figure per temperature."""

    if isinstance(property_files, dict):
        file_by_label = {str(key): Path(value) for key, value in property_files.items()}
        labels = [_model_label(model) for model in models]
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"Several models share the label(s) {duplicates}, so property_files cannot be "
                "matched by label. Pass property_files as a list in the order of the models."
            )
        unknown = [label for label in labels if label not in file_by_label]
        if unknown:
            raise ValueError(
                f"property_files has no entry for the model label(s) {unknown}; "
                f"given: {sorted(file_by_label)}."
            )
        property_file_list = [file_by_label[label] for label in labels]
    else:
        property_file_list = [Path(path) for path in property_files]
        if len(property_file_list) != len(models):
            raise ValueError("property_files must have the same length as models.")

    Ts = _temperatures(models, temperatures)
    figures = {}
    for T in Ts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ylabel_parts = []
        for model, property_file in zip(models, property_file_list):
            prop_values, prop_names, prop_cluster_ids = read_properties(model, property_file)
            selected_indices = _selected_property_indices(prop_names, columns)
            prop_cluster_ids = np.asarray(prop_cluster_ids)
            cluster_index_by_id = {cluster_id: idx for idx, cluster_id in enumerate(model.cluster_id)}
            prop_cluster_indices = [cluster_index_by_id[cluster_id] for cluster_id in prop_cluster_ids]

            for idx in selected_indices:
                prop = prop_values[:, idx]
                if len(prop_cluster_ids) == len(model.cluster_id):
                    avg = model.gqca_mean(prop, T)
                    std = model.gqca_std(prop, T)
                else:
                    norm_probs, idx_sorted = normalize_probabilities(model, T, prop_cluster_indices)
                    prop_position_by_cluster_idx = {
                        cluster_idx: pos for pos, cluster_idx in enumerate(prop_cluster_indices)
                    }
                    avg = norm_probs @ prop[[prop_position_by_cluster_idx[cluster_idx] for cluster_idx in idx_sorted]]
                    std = None
                prop_name = prop_names[idx]
                ylabel_parts.append(prop_name)
                line_label = f"{_model_label(model)}: {prop_name}"
                ax.plot(model.x_plot, avg, linewidth=2, label=line_label)
                if show_std and std is not None:
                    ax.fill_between(model.x_plot, avg - std, avg + std, alpha=0.12)
        ax.set_title(f"Averaged properties, T = {T:g} K")
        ax.set_xlabel(models[0].xlabel_compound())
        ax.set_ylabel(", ".join(dict.fromkeys(ylabel_parts)))
        ax.legend()
        fig.tight_layout()
        figures[float(T)] = fig
    if show:
        plt.show()
    return figures
