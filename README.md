# EGQCA

**Extended Generalized Quasichemical Approximation** for (quasi-)binary alloys
$A_{1-x}B_xC$, with optional vibrational and PV (enthalpy) contributions.

EGQCA computes thermodynamic properties of an alloy at arbitrary composition $x$
and temperature $T$ by statistically averaging over a set of small clusters.
From the cluster energies it gives the mixing free energy, enthalpy and entropy,
the cluster occurrence probabilities, the miscibility gap (binodal and spinodal),
and composition-averaged properties such as lattice parameters.

The method is described in
[doi:10.1016/j.mtphys.2024.101547](https://doi.org/10.1016/j.mtphys.2024.101547),
the code in [arXiv:2611.34551](https://doi.org/10.48550/arXiv.2611.34551).

## Features

- Generation of all non-equivalent clusters of a supercell from a CIF file
  (via the [supercell program](https://orex.github.io/supercell/))
- Input files for DFT relaxations with FHI-aims, Quantum ESPRESSO or VASP, and
  automatic extraction of the energies and lattice parameters
- Relaxation and phonons with machine-learned interatomic potentials (UPET, MACE,
  MatterSim, GRACE, or any ASE calculator)
- Gibbs free energy with mixing energy, configurational entropy, vibrational
  free energy from phonon DOS, and PV term
- Miscibility gap, averaged properties and cluster reduction

## Installation

```bash
pip install egqca
```

The ML relaxation needs the package of the chosen model family. Install each
into its own environment, as their dependencies can conflict:

```bash
pip install "egqca[upet]"        # UPET / PET models
pip install "egqca[mace]"        # MACE
pip install "egqca[mattersim]"   # MatterSim
pip install "egqca[grace]"       # GRACE
```

Python 3.10 or newer is required.

To install the development version from source:

```bash
git clone https://github.com/egqca/egqca
cd egqca
pip install -e .
```

## Quick start

With existing input files, a calculation looks like this:

```python
import numpy as np
import egqca

m = egqca.EGQCA(
    datafile="clusters.dat",
    compound="A$_{1-x}$B$_x$C",
    volume_file="volume.dat",          # optional, for the PV term
    pressure_GPa=0.0,                  # optional
    phonon_file_name="ph_dos",         # optional, folder with <cluster_id>.dos files
)

x_grid = np.arange(1e-6, 1.0, 1.0e-3)
m.thermodynamic_assessment(x_grid, Tlist=[600, 800, 1000, 1400])
m.average_properties("volume.dat")
```

## Example

[`examples/NbV/`](https://github.com/egqca/egqca/tree/main/examples/NbV) covers the full workflow with ML potentials for bcc
Nb$_{1-x}$V$_x$ with 8-atom clusters:

| Script | Step |
|---|---|
| `01_generate_clusters.py` | generate the 22 non-equivalent clusters |
| `02_ml_relaxation_phonons.py` | ML relaxation and phonons with `pet-omatpes-l` |
| `03_run_egqca.py` | EGQCA without phonons, with phonons, and with phonons + PV |

The results of step 2 are included, so `03_run_egqca.py` runs directly.

## Documentation

The full documentation is available at [egqca.github.io/egqca](https://egqca.github.io/egqca/).

## Citation

If you use EGQCA, please cite

- D. Spath *et al.*, EGQCA Code: automated high-throughput thermodynamic
  modeling of solid solutions,
  [arXiv:2611.34551](https://doi.org/10.48550/arXiv.2611.34551) (2026)
- [doi:10.1016/j.mtphys.2024.101547](https://doi.org/10.1016/j.mtphys.2024.101547)

## License

EGQCA is released under the [MIT License](https://github.com/egqca/egqca/blob/main/LICENSE).

