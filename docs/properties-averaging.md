# Properties Averaging

`average_properties()` computes composition-dependent averages of additional cluster-resolved properties after the EGQCA probabilities have been calculated. Typical examples are density of states values, orbital occupations, lattice-related quantities, or any scalar quantity that can be assigned to each cluster.

## Properties File Format

The properties file must contain a cluster ID in the first column and one or more numeric property columns after it.

```text
# ID DOS s p d
S0_i0_w1      1.23   0.10   0.40   0.73
S1_i00_w16    1.45   0.12   0.46   0.87
S2_i000_w64   1.61   0.14   0.50   0.97
```

Each header token after `ID` names one property column and becomes its y-axis label and output-name base. Multiple property columns are handled independently.

Requirements:

- The first column must match cluster IDs from `clusters.dat`.
- Numeric property values start in column 2.
- The file may contain all clusters or only a subset.
- The header line is the one matching `#\s*ID`; its remaining tokens are the property names. Without such a line the columns are named `Property_1`, `Property_2`, and so on.
- Property names are used as LaTeX in the axis labels (the plotting style enables `text.usetex`), so a name such as `N(\varepsilon_F)` is rendered as a formula.

## Basic Usage

If you have already run `thermodynamic_assessment()`, cached probabilities are reused:

```python
m.thermodynamic_assessment(x_grid, Tlist)
m.average_properties("properties.dat")
```

You can also provide the grid explicitly:

```python
m.average_properties(
    properties_file="properties.dat",
    x=x_grid,
    Tlist=[800.0],
)
```

## Optional Parameters

| Parameter | Description |
| --- | --- |
| `ylabels` | List of y-axis labels overriding the property column headers from the file. Ignored with a warning if its length does not match the number of selected columns. |
| `columns` | 0-based column indices (excluding the ID column) selecting which property columns to use. Either one list applied to every properties file, or a list of lists with one entry per file. |
| `legend_labels` | One legend label per properties file. Defaults to the file stems. |
| `figsize` | Figure size as `(width, height)` tuple, default `(6, 6)`. |
| `surrogate_model` | Surrogate file or list of surrogate files from `cluster_importance()`. Mutually exclusive with `idx_clusters_surrogate`. |

All output is written to the `results/` subfolder of the output directory. When the model was set up with `phonon_file_name`, the folder is `results_phonons/` and every file name carries a `_phonons` suffix, e.g. `DOS_T800.0_phonons.dat`.

## Comparing Several Properties Files

`properties_file` may also be a list or tuple of paths. All files are averaged independently and their curves for the same property are overlaid in the same figure:

```python
m.average_properties(
    properties_file=["properties_run1.dat", "properties_run2.dat"],
    x=x_grid,
    Tlist=[800.0],
    legend_labels=["run 1", "run 2"],
)
```

## Full-Cluster Averages

If the properties file contains one row for every cluster in `clusters.dat`, EGQCA computes:

- the full GQCA average
- the standard deviation, drawn as a shaded band around the curve
- one `.dat` file and one `.pdf` figure per property and temperature

Output names use:

```text
<property>_T<temperature>.dat
<property>_T<temperature>.pdf
```

Example:

```text
DOS_T800.0.dat
DOS_T800.0.pdf
```

`<property>` is the property label truncated at the first `(` or `/`, so a column header such as `N(e_F)` produces files named `N_T800.0.*`.

## Subset Properties

If the properties file contains only a subset of clusters, EGQCA renormalizes the probabilities over the available clusters and computes an approximate average. The renormalization is done per number of substituted atoms $n_j$: within each $n_j$ class the available clusters take over the full probability $P_n$ of the class, in proportion to their own probabilities, $\tilde p_j = p_j \, P_n / \sum_{i \in \text{subset}, n_i = n_j} p_i$. The weights sum to 1 and the average composition stays $x$, so every $n_j$ needs at least one cluster in the subset. The same applies to surrogate models. No standard deviation is reported for such a curve.

The file names do not change; the number of clusters appears in the column headers of the `.dat` file and in the figure legend (`<N>/<total> clusters`).

## Averaging with Surrogate Model Files

Surrogate model files from `cluster_importance()` can be passed with `surrogate_model`:

```python
m.average_properties(
    properties_file="total-dos.dat",
    x=x_grid,
    Tlist=[800.0],
    surrogate_model="cluster_reduction/surrogate_T800.0_18clusters.dat",
)
```

The surrogate file is read as a list of cluster IDs. EGQCA then averages the property using only the surrogate clusters that are also present in the properties file.

## Several Surrogate Files in One Figure

`surrogate_model` may also be a list or tuple of paths. Each surrogate file is read separately, and all resulting averages are plotted in the same figure:

```python
from pathlib import Path

surrogate_files = sorted(Path("cluster_reduction").glob("surrogate_*.dat"))

m.average_properties(
    properties_file="total-dos.dat",
    x=x_grid,
    Tlist=[800.0],
    surrogate_model=surrogate_files,
)
```

All surrogate curves end up in the same figure and in the same `<property>_T<temperature>.dat` file; the cluster counts appear in the column headers, not in the file name. The plotted surrogate curves cycle through Matplotlib line styles so that several reduced models can be compared in one plot, while the full or subset curve is drawn as a solid line.

Surrogate clusters that are missing from the properties file are skipped, and a surrogate whose clusters are all missing is dropped entirely.

## Output Columns

The output `.dat` file always starts with the composition grid:

```text
x
```

One column follows per plotted curve, plus a standard-deviation column for the full-cluster curve:

| Column name | Meaning |
| --- | --- |
| `avg_<N>_clusters` | Average over the `N` clusters of this curve — the full set, the subset present in the properties file, or one surrogate. |
| `std_<N>_clusters` | Standard deviation, written only for the full cluster set. |

When several properties files are averaged in one call, every column name is prefixed with the file label, for example `DFT_avg_44_clusters` and `mace_avg_44_clusters`.

When all clusters are present and surrogate files are also provided, the output file contains both the full result and the surrogate results. This is the recommended way to validate cluster-reduced property calculations.

## Complete Example

```python
import egqca
import numpy as np
from pathlib import Path

m = egqca.EGQCA(
    datafile="clusters.dat",
    volume_file="volume.dat",
    pressure_GPa=300,
)

x_grid = np.arange(1e-6, 1.0, 1.0e-3)
Tlist = [800.0]

m.thermodynamic_assessment(x_grid, Tlist)
m.cluster_importance(x_grid, Tlist, top_percentage=[0.02, 0.05, 0.1])

surrogate_files = sorted(Path("cluster_reduction").glob("surrogate_*.dat"))

m.average_properties(
    properties_file="total-dos.dat",
    x=x_grid,
    Tlist=Tlist,
    surrogate_model=surrogate_files,
)
```

This creates one combined figure per property, comparing the full-cluster result with each surrogate model.
