"""Synthetic test data: a made-up alloy A$_{2-x}$B$_x$.

The n = 6 permutable sites form a ring with dihedral (D6) symmetry, which gives
13 symmetry-distinct clusters; their degeneracies g_j add up to C(6, n_j) in every
n_j class. A cluster has 9 sites, so sites != n as in real data. Energies come from
a nearest/next-nearest neighbour model, lattice parameters from Vegard's law.
"""
from itertools import product
from pathlib import Path

import pytest

N_PERMUTABLE = 6
SITES = 9


def ring_clusters(n: int = N_PERMUTABLE) -> list[tuple[tuple[int, ...], int]]:
    """Symmetry-distinct decorations of an n-site ring and their multiplicities, by n_j."""
    orbits = {}
    for config in product((0, 1), repeat=n):
        orbit = {tuple(config[(s * i + r) % n] for i in range(n)) for r in range(n) for s in (1, -1)}
        orbits.setdefault(min(orbit), len(orbit))
    return sorted(orbits.items(), key=lambda item: (sum(item[0]), item[0]))


def cluster_energy(config: tuple[int, ...], J: float) -> float:
    """Total energy in Ry: linear in x plus J per unlike nearest and 0.3 J per unlike next-nearest pair."""
    n = len(config)
    x = sum(config) / n
    unlike_nn = sum(config[i] != config[(i + 1) % n] for i in range(n))
    unlike_nnn = sum(config[i] != config[(i + 2) % n] for i in range(n))
    return -50.0 * (1 - x) - 56.0 * x + J * (unlike_nn + 0.3 * unlike_nnn)


def write_alloy(folder: Path, J: float = -0.01) -> tuple[Path, Path]:
    """Write clusters.dat and volume.dat (space separated) into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    clusters = [
        "# compound= A$_{2-x}$B$_x$",
        f"# total number of sites={SITES}",
        f"# permutable sites n={N_PERMUTABLE}",
        "#ID     nj  Nat  xj      gj   Ej (Ry)",
    ]
    volumes = ["#ID  a(Å)  b(Å)  c(Å)  alpha(°)  beta(°)  gamma(°)"]
    index = {}
    for config, g in ring_clusters():
        nj = sum(config)
        i = index[nj] = index.get(nj, -1) + 1
        cluster_id = f"S{nj}_i{i}_w{g}"
        x = nj / N_PERMUTABLE
        clusters.append(f"{cluster_id}  {nj}  {SITES}  {x:.10f}  {g}  {cluster_energy(config, J):.8f}")
        a, c = 3.0 + 0.1 * x, 3.5 - 0.05 * x
        volumes.append(f"{cluster_id}  {a:.6f}  {a:.6f}  {c:.6f}  90.00  90.00  120.00")
    (folder / "clusters.dat").write_text("\n".join(clusters) + "\n")
    (folder / "volume.dat").write_text("\n".join(volumes) + "\n")
    return folder / "clusters.dat", folder / "volume.dat"


@pytest.fixture
def alloy(tmp_path) -> tuple[Path, Path]:
    """clusters.dat and volume.dat of the synthetic alloy, in ``tmp_path / "data"``."""
    return write_alloy(tmp_path / "data")
