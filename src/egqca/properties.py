import warnings
from collections import Counter
import numpy as np
from pathlib import Path
from .io import read_properties, sort_by_id
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
from matplotlib.lines import lineStyles
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .egqca_model import EGQCA

def calculate_averages(
    self: "EGQCA",
    properties_file: str | Path | list[str | Path] | tuple[str | Path, ...],
    Tlist: list[float],
    x_grid: np.ndarray,
    idx_clusters_surrogate: dict[float, dict[float | str, list[int]]] | None = None,
    ylabels: list[str] | None = None,
    columns: list[int] | list[list[int]] | None = None,
    legend_labels: list[str] | None = None,
    figsize: tuple = (6, 6),
) -> dict[str, dict[str, dict[float, list[dict]]]]:
    """
    Average the quantities of one or several properties files at the given temperatures
    ``Tlist`` and compositions ``x_grid`` and plot them together (overlaid per property).

    This is a thin orchestrator that delegates the averaging to :func:`compute_averages`
    (one call per properties file) and the figure/``.dat`` output to :func:`plot_averages`.

    Parameters
    ----------
    properties_file : str or Path or list
        A single properties file or a list of properties files. When several files are
        given, their curves for the same property are overlaid in one figure.
    Tlist : list of float
        Temperatures in K.
    x_grid : numpy.ndarray
        Composition grid.
    idx_clusters_surrogate : dict or None
        Optional surrogate cluster sets per temperature,
        ``{T: {name: [cluster indices]}}``. ``name`` is the percentage from
        :meth:`EGQCA.rank_clusters` or the file name of a surrogate model.
    ylabels : list of str or None
        Optional override for the property labels (applied to every file).
    columns : list or None
        Optional subset of property columns to use (0-based, excluding the ID column).
        Either a single list applied to every file, or a list of lists (one per file).
    legend_labels : list of str or None
        Optional legend names for the files (e.g. method names). Defaults to the file
        stems. Used to label and group the overlaid curves.
    figsize : tuple
        Figure size as ``(width, height)``.

    Returns
    -------
    dict
        Nested dict ``{file_label: {ylabel: {T: [{"label", "avg", "std"}, ...]}}}``.
        ``std`` is None for subset/surrogate curves (only the full cluster set has std).
    """
    if isinstance(properties_file, (str, Path)):
        properties_files = [Path(properties_file)]
    else:
        properties_files = [Path(pf) for pf in properties_file]
    if not properties_files:
        raise ValueError("At least one properties file must be provided.")

    # legend names per file (default: file stem)
    if legend_labels is None:
        labels = [pf.stem for pf in properties_files]
    elif len(legend_labels) != len(properties_files):
        raise ValueError("legend_labels must have one entry per properties file.")
    else:
        labels = list(legend_labels)

    # ``columns`` may be a single selection for all files or one selection per file
    per_file_columns: list[list[int] | None]
    if columns is not None and len(columns) > 0 and all(isinstance(c, (list, tuple)) for c in columns):
        if len(columns) != len(properties_files):
            raise ValueError("When columns is a list of lists it must have one entry per properties file.")
        per_file_columns = [list(c) for c in columns]
    else:
        per_file_columns = [columns] * len(properties_files)

    results_by_file: list[tuple[str, dict[str, dict[float, list[dict]]]]] = []
    for properties_file_path, label, cols in zip(properties_files, labels, per_file_columns):
        results = compute_averages(
            self,
            properties_file_path,
            Tlist,
            idx_clusters_surrogate=idx_clusters_surrogate,
            ylabels=ylabels,
            columns=cols,
        )
        results_by_file.append((label, results))

    plot_averages(self, results_by_file, x_grid, Tlist, figsize=figsize)

    return {label: results for label, results in results_by_file}


def compute_averages(
    self: "EGQCA",
    properties_file: str | Path,
    Tlist: list[float],
    idx_clusters_surrogate: dict[float, dict[float | str, list[int]]] | None = None,
    ylabels: list[str] | None = None,
    columns: list[int] | None = None,
) -> dict[str, dict[float, list[dict]]]:
    """
    Average the quantities in a single ``properties_file`` over the cluster ensemble.

    For each selected property column and each temperature the following curves are computed:
      * the full cluster set (mean and std) when the file contains every cluster;
      * a re-normalized average over the available subset when it contains fewer clusters;
      * a re-normalized average per surrogate cluster set in ``idx_clusters_surrogate``.

    Parameters
    ----------
    properties_file : str or Path
        Properties file to average.
    Tlist : list of float
        Temperatures in K.
    idx_clusters_surrogate : dict or None
        Optional surrogate cluster sets per temperature,
        ``{T: {name: [cluster indices]}}``. ``name`` is the percentage from
        :meth:`EGQCA.rank_clusters` or the file name of a surrogate model.
    ylabels : list of str or None
        Optional override for the property labels.
    columns : list of int or None
        Optional 0-based indices selecting which property columns to use (excluding
        the leading ID column). None uses all columns.

    Returns
    -------
    dict
        ``{ylabel: {T: [{"label", "avg", "std", "kind", "n_clusters"}, ...]}}``.
        ``std`` is None for subset/surrogate curves.
    """
    # read in property from properties.dat file
    # Columns: ID, any quantitiy, another quantity, ...
    Prop_j, Prop_names, Prop_cluster_id = read_properties(self, properties_file)  # shape (num_clusters, num_properties)

    # select a subset of property columns (by index, excluding the ID column)
    if columns is not None:
        columns = list(columns)
        Prop_j = Prop_j[:, columns]
        Prop_names = [Prop_names[i] for i in columns]

    if ylabels is None:
        ylabels = Prop_names
    elif len(ylabels) != len(Prop_names):
        ylabels = Prop_names
        warnings.warn("The number of ylabels does not match the number of properties in the properties file. Using the header from the properties file instead.")

    prop_cluster_id = np.asarray(Prop_cluster_id)
    prop_cluster_count = len(prop_cluster_id)
    uses_all_clusters = prop_cluster_count == len(self.cluster_id)
    cluster_index_by_id = {cluster_id: idx for idx, cluster_id in enumerate(self.cluster_id)}
    prop_cluster_indices = [cluster_index_by_id[cluster_id] for cluster_id in prop_cluster_id]
    prop_position_by_cluster_idx = {cluster_idx: pos for pos, cluster_idx in enumerate(prop_cluster_indices)}

    all_results: dict[str, dict[float, list[dict]]] = {}

    for idx, ylabel in enumerate(ylabels):
        Prop = Prop_j[:, idx]
        all_results[ylabel] = {}
        for T in Tlist:
            curve_specs = []

            # average over all clusters
            if uses_all_clusters:
                full_avg = self.gqca_mean(Prop, T)
                full_std = self.gqca_std(Prop, T)
                curve_specs.append(
                    {
                        "label": f"All {len(self.cluster_id)} clusters",
                        "avg": full_avg,
                        "std": full_std,
                        "kind": "full",
                        "n_clusters": len(self.cluster_id),
                    }
                )
            # if not all clusters are in the properties file
            else:
                norm_probs, idx_sorted = normalize_probabilities(self, T, prop_cluster_indices)
                subset_avg = norm_probs @ Prop[[prop_position_by_cluster_idx[cluster_idx] for cluster_idx in idx_sorted]]
                curve_specs.append(
                    {
                        "label": f"{prop_cluster_count}/{len(self.cluster_id)} clusters",
                        "avg": subset_avg,
                        "std": None,
                        "kind": "subset",
                        "n_clusters": prop_cluster_count,
                    }
                )

            # average over the given surrogate models
            if idx_clusters_surrogate is not None:
                for surrogate_name, idx_surrogate in (idx_clusters_surrogate[T].items()):
                    idx_surrogate_available = [
                        cluster_idx for cluster_idx in idx_surrogate
                        if cluster_idx in prop_position_by_cluster_idx
                    ]
                    if not idx_surrogate_available:
                        continue

                    norm_probs, idx_sorted = normalize_probabilities(self, T, idx_surrogate_available)
                    prop_positions = [prop_position_by_cluster_idx[cluster_idx] for cluster_idx in idx_sorted]
                    surrogate_avg = norm_probs @ Prop[prop_positions]
                    num_clusters = len(idx_sorted)
                    curve_specs.append(
                        {
                            "label": f"{num_clusters} clusters",
                            "avg": surrogate_avg,
                            "std": None,
                            "kind": "surrogate",
                            "n_clusters": num_clusters,
                        }
                    )

            if curve_specs:
                all_results[ylabel][T] = curve_specs

    return all_results


def _short_property_name(label: str) -> str:
    """Property label without its unit or argument, e.g. 'E_g' for 'E_g(eV)'."""
    return re.split(r"[(/]", label)[0].strip()


def _output_names(ylabels: list[str]) -> dict[str, str]:
    """Filename stem per property label, unique within one call.

    The short name (see :func:`_short_property_name`) is kept where it is unique;
    labels that share it, e.g. 'DOS(E_F+0eV)' and 'DOS(E_F+0.1eV)', get their full
    label instead, so that their outputs do not overwrite each other.
    """
    def safe(text: str) -> str:
        return re.sub(r"[^\w.+-]+", "_", text).strip("_")

    short = {label: safe(_short_property_name(label)) for label in ylabels}
    counts = Counter(short.values())
    names = {label: short[label] if counts[short[label]] == 1 else safe(label) for label in ylabels}
    if len(set(names.values())) != len(names):
        raise ValueError(f"The property labels {ylabels} do not give distinct output file names.")
    return names


def plot_averages(
    self: "EGQCA",
    results_by_file: list[tuple[str, dict[str, dict[float, list[dict]]]]],
    x_grid: np.ndarray,
    Tlist: list[float],
    figsize: tuple = (6, 6),
) -> None:
    """
    Overlay the averaged curves of several files in one figure per (property, temperature)
    and save the corresponding ``.dat`` and ``.pdf`` outputs.

    Curves coming from different files share a property axis (overlaid) and are coloured by
    file; within a file the full/subset curve is solid and surrogate curves use distinct
    line styles.

    Parameters
    ----------
    results_by_file : list of tuple
        ``[(file_label, {ylabel: {T: [curve_spec, ...]}}), ...]`` as returned by
        :func:`compute_averages`.
    x_grid : numpy.ndarray
        Composition grid the curves are plotted on.
    Tlist : list of float
        Temperatures in K.
    figsize : tuple
        Figure size as ``(width, height)``.
    """
    x_label = self.xlabel_compound()
    multi_file = len(results_by_file) > 1
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    file_colors = {file_label: color_cycle[i % len(color_cycle)] for i, (file_label, _) in enumerate(results_by_file)}
    visible_linestyles = [style for style in lineStyles if style not in ("None", " ", "", "-")]

    # collect every property label across all files, preserving first-seen order
    ylabels: list[str] = []
    for _, results in results_by_file:
        for ylabel in results:
            if ylabel not in ylabels:
                ylabels.append(ylabel)

    output_names = _output_names(ylabels)
    for ylabel in ylabels:
        name = output_names[ylabel]
        for T in Tlist:
            # gather the curves for this property/temperature from every file
            entries = []
            curves_per_file: dict[str, int] = {}
            for file_label, results in results_by_file:
                file_curves = results.get(ylabel, {}).get(T, [])
                curves_per_file[file_label] = len(file_curves)
                for curve in file_curves:
                    entries.append((file_label, curve))
            if not entries:
                continue

            columns = [x_grid]
            header = ["x"]
            surrogate_linestyles = {file_label: cycle(visible_linestyles) for file_label, _ in results_by_file}

            with mpl.rc_context(self.style):
                plt.subplots(figsize=figsize, constrained_layout=True)
                plt.xlim(x_grid[0], x_grid[-1])
                plt.xticks((np.arange(np.floor(x_grid[0]), x_grid[-1] + 0.1, step=np.ceil(np.max(x_grid)) / 4)))
                plt.xlabel(x_label)
                plt.ylabel(rf"${ylabel}$")

                for file_label, curve in entries:
                    if curve["kind"] == "surrogate":
                        linestyle = next(surrogate_linestyles[file_label])
                        linewidth = 2.0
                    elif curve["kind"] == "full":
                        linestyle = "-"
                        linewidth = 3.0
                    else:  # subset
                        linestyle = "-"
                        linewidth = 2.0

                    plot_kwargs = {"linestyle": linestyle, "linewidth": linewidth}
                    if multi_file:
                        plot_kwargs["color"] = file_colors[file_label]

                    # for multiple files, prefix with the file label; drop the cluster-count
                    # suffix when this file contributes only a single curve to the figure
                    if not multi_file:
                        label = curve["label"]
                    elif curves_per_file[file_label] == 1:
                        label = file_label
                    else:
                        label = f"{file_label}: {curve['label']}"
                    line, = plt.plot(x_grid, curve["avg"], label=label, **plot_kwargs)
                    if curve["std"] is not None:
                        plt.fill_between(
                            x_grid,
                            curve["avg"] - curve["std"],
                            curve["avg"] + curve["std"],
                            color=line.get_color(),
                            alpha=0.1,
                        )

                    col_prefix = re.sub(r"\s+", "_", file_label) + "_" if multi_file else ""
                    columns.append(curve["avg"])
                    header.append(f"{col_prefix}avg_{curve['n_clusters']}_clusters")
                    if curve["std"] is not None:
                        columns.append(curve["std"])
                        header.append(f"{col_prefix}std_{curve['n_clusters']}_clusters")

                plt.legend()
                output_stem = f"{name}_T{T}"
                np.savetxt(
                    self.current_output_dir / self._add_phonon_suffix(f"{output_stem}.dat"),
                    np.column_stack(columns),
                    header="\t".join(header),
                    fmt="%.8e",
                )
                plt.savefig(self.current_output_dir / self._add_phonon_suffix(f"{output_stem}.pdf"), bbox_inches="tight", dpi=300)
                plt.close()


def normalize_probabilities(self: "EGQCA", T: float, idx_surrogate: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Renormalize the probabilities of a reduced cluster set per n_j class.

    Within each class the selected clusters are scaled to carry the full class
    weight P_n, keeping their relative probabilities:

        p̃_j = p_j * P_n / sum(p_i for selected i in the class)

    The weights sum to 1 and the average composition stays sum(p̃_j x_j) = x. Where
    all selected clusters of a class have zero probability, P_n is split equally.

    The selected clusters are sorted by their cluster ID with :func:`~egqca.io.sort_by_id`,
    and that is the order of the returned columns. The sorted cluster indices are returned
    alongside so that the caller can bring its property values into the same order.

    Parameters
    ----------
    T : float
        Temperature in K.
    idx_surrogate : list of int
        Cluster indices belonging to the surrogate model.

    Returns
    -------
    tuple
        ``(norm_probs, idx_sorted)`` with ``norm_probs`` of shape
        ``(len(x), len(idx_sorted))``.
    """
    probjs = self.pj[T].copy()
    probjs = probjs.T

    # sort the selected clusters by ID; this fixes the order of the returned columns
    idx_surrogate = np.asarray(list(idx_surrogate), dtype=int)
    unique, counts = np.unique(idx_surrogate, return_counts=True)
    if np.any(counts > 1):
        duplicates = [str(np.asarray(self.cluster_id)[i]) for i in unique[counts > 1]]
        raise ValueError(f"The selected clusters contain these clusters more than once: {', '.join(duplicates)}")
    _, idx_surrogate = sort_by_id(np.asarray(self.cluster_id)[idx_surrogate], idx_surrogate)

    # the probability of an n_j class is moved onto its selected clusters, so every
    # class needs at least one of them, otherwise the weights do not sum to 1
    missing_nj = sorted(set(np.unique(self.nj).tolist()) - set(np.asarray(self.nj)[idx_surrogate].tolist()))
    if missing_nj:
        raise ValueError(
            f"The selected clusters contain no cluster with n_j = {missing_nj}. Every n_j needs at "
            "least one cluster, e.g. add the missing clusters to the properties file."
        )

    norm_probs = []
    idx_sorted = []
    for i in np.unique(self.nj):
        idx_nj = np.where(self.nj == i)[0]
        # keep the ID-sorted order within each n_j group
        idx_surrogate_nj = [idx for idx in idx_surrogate.tolist() if idx in set(idx_nj.tolist())]
        p_selected = probjs[idx_surrogate_nj, :]
        class_weight = probjs[idx_nj, :].sum(axis=0)
        selected_weight = p_selected.sum(axis=0)
        has_weight = selected_weight > 0
        scale = np.divide(class_weight, selected_weight, out=np.zeros_like(class_weight), where=has_weight)
        equal_share = class_weight / len(idx_surrogate_nj)
        norm_probs.extend(np.where(has_weight, p_selected * scale, equal_share))
        idx_sorted.extend(idx_surrogate_nj)

    return np.array(norm_probs).T, np.asarray(idx_sorted, dtype=int)

def r2_value(y:np.ndarray, y_hat:np.ndarray) -> float:
    """ Coefficient of determination R² between y and y_hat """
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    return 1 - ss_res/ss_tot
