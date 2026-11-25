# Cluster Reduction

The cluster reduction workflow identifies smaller surrogate cluster sets that retain the most important statistical weight of the full EGQCA ensemble. This is useful when a property is expensive to compute for every cluster.

The entry point is:

```python
m.cluster_importance(
    x=x_grid,
    Tlist=[800.0],
    top_percentage=[0.02, 0.05, 0.1, 0.2],
)
```

## What the Algorithm Does

For every requested temperature, EGQCA first computes the full cluster occurrence probabilities:

$$
p_j(x, T)
$$

where `j` labels a cluster. The reduction algorithm then ranks clusters by their integrated probability weight over the composition grid:

$$
W_j = \int p_j(x, T)\,dx
$$

The ranking is performed separately within each cluster-composition group `n_j`. This is important because a useful surrogate model should retain representatives across the different substitution counts rather than selecting only the globally most probable composition class.

For each value in `top_percentage`, EGQCA:

1. Groups cluster indices by `n_j`.
2. Computes the integrated probability weight for each cluster.
3. Selects the highest-weight clusters within each `n_j` group (min 1).
4. Writes the selected cluster IDs and their probability weights to a surrogate file.

## Output Files

Cluster reduction results are written under:

```text
<out_folder_name>/cluster_reduction/
```

For a temperature of `800.0 K`, the files are named like:

```text
surrogate_T800.0_18clusters.dat
surrogate_T800.0_29clusters.dat
surrogate_T800.0_44clusters.dat
```

If the model was set up with `phonon_file_name`, the folder and the file names carry a `_phonons` suffix, for example `cluster_reduction_phonons/surrogate_T800.0_18clusters_phonons.dat`.

Each surrogate file contains:

```text
# Selected clusters: <N>/<total>
# Cluster ID     Probability weight
<cluster_id>     <integrated_weight>
...
```

These files can be passed directly to `average_properties()` as `surrogate_model` inputs. Only the first column is read back, so any file listing cluster IDs works as a surrogate; IDs that are not in `clusters.dat` raise an error.

`cluster_importance()` also writes `occurence_probabilities_T<T>.dat`, whose first column is the composition grid and not a list of cluster IDs.

To average properties over the selected subsets, pass the surrogate files to `average_properties()` (see the example below). The method returns `self.S_clusters`, a nested dict `{T: {percentage: cluster IDs}}`.

## Example

```python
import egqca
import numpy as np
from pathlib import Path

m = egqca.EGQCA(
    datafile="clusters.dat",
    volume_file="volume.dat",
    pressure_GPa=300,
    out_folder_name="DFT",
)

x_grid = np.arange(1e-6, 1.0, 1.0e-3)
Tlist = [800.0]

m.cluster_importance(
    x_grid,
    Tlist,
    top_percentage=[0.02, 0.05, 0.1, 0.2, 0.5],
)
```

You can then average a property on several surrogate cluster sets:

```python
surrogate_dir = Path("DFT/cluster_reduction")
surrogate_files = sorted(Path(surrogate_dir).glob("surrogate_*.dat"))

m.average_properties(
    properties_file="total-dos.dat",
    x=x_grid,
    Tlist=Tlist,
    surrogate_model=surrogate_files,
)
```

## Choosing `top_percentage`

`top_percentage` controls the surrogate size by selecting a percentage of clusters (at least one cluster is retained for each $n_j$). Smaller values produce cheaper property workflows, but they may miss rare clusters that still matter for a property. Larger values approach the full ensemble.

A practical strategy is:

1. Generate several surrogate files, for example `2%`, `5%`, `10%`, and `20%`.
2. Average the same property for all surrogate files in one figure.
3. Check whether the property curve is converged with respect to the surrogate size.
4. Use the smallest surrogate that gives stable results for the quantity of interest.

## Relation to Probability Normalization

When properties are averaged on a reduced cluster set, EGQCA renormalizes the cluster probabilities within the available surrogate clusters. This allows the reduced model to produce a composition-dependent average even though not all clusters are present in the property file or surrogate file.

The surrogate model is therefore an approximation to the full statistical ensemble. It should be validated for each property, temperature range, and composition range where it is used.
