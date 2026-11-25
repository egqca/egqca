## Entropy Approximation

For a crystal with $N$ permutable sites described by $M$ clusters, the configurational mixing entropy is

$$
\Delta S = -N k_B\big[x\ln x + (1-x)\ln(1-x)\big] - M k_B \sum_{j=1}^{J} p_j \ln\left(\frac{p_j}{p^0_j}\right),
$$

where $p^0_j$ are the cluster probabilities of the ideal (random) solid solution. The second sum is the Kullback-Leibler divergence $D_\mathrm{KL}(p \Vert p^0)$. The number of clusters $M$ depends on how the clusters are assumed to fill the crystal, and is selected with `entropy_approximation`:

| `entropy_approximation` | Clusters | $M$ | Boltzmann factor in $p_j$ |
|---|---|---|---|
| `"overlapping"` (default) | every permutable site starts its own cluster, so clusters share sites | $N$ | $\exp\big(-\Delta\varepsilon_j N_\mathrm{s} / (n k_B T)\big)$ |
| `"non-overlapping"` | the crystal is tiled by clusters that share no sites | $N/n$ | $\exp\big(-\Delta\varepsilon_j N_\mathrm{s} / (k_B T)\big)$ |


Here $\Delta\varepsilon_j$ is the mixing energy of cluster $j$ per site (plus the vibrational contribution, if phonons are given), so $\Delta\varepsilon_j N_\mathrm{s}$ is the mixing energy of the whole cluster. 

The `non-overlapping` entropy never gives a miscibility gap.
