# EGQCA
**Extended Generalized Quasichemical Approximation**
for (quasi-)binary alloys with optional vibrational and PV (enthalpy) contributions.

Implementation based on:
https://doi.org/10.1016/j.mtphys.2024.101547

## Overview

The EGQCA code computes thermodynamic properties of (quasi)-binary alloys $A_{1-x}B_x$C at arbitrary composition $x$ by statistically averaging over smaller clusters.

The core package dependencies cover the standard EGQCA workflow. The optional ML relaxation helper requires the extra package set installed via `pip install egqca[upet]` (UPET models); MACE, MatterSim and GRACE models need `egqca[mace]`, `egqca[mattersim]` and `egqca[grace]` instead. See [ML Workflow](ml-workflow.md) for model choices, relaxation settings, and ML phonons.

The recommended entry point is the automated workflow. Starting from a CIF file and a supercell size, `egqca` can generate all non-equivalent clusters, create the folder structure for the DFT calculations, and prepare the EGQCA input files automatically. Writing `clusters.dat` by hand is still possible for custom setups, but it is tedious and error-prone.

For a given supercell size $N_1 \times N_2 \times N_3$, all non-equivalent clusters can be generated using the [supercell program](https://orex.github.io/supercell/). The supercell size should be treated as a **convergence parameter**.

## Gibbs Free Energy

The total Gibbs free energy may include:

- **Mixing energies**
- **Mixing entropy**, in the overlapping or non-overlapping cluster approximation (see [Entropy Approximation](entropy.md))
- **PV term** (enthalpy per site), if `volume_file` is provided and `pressure_GPa` is specified
- **Vibrational free energy**, if `phonon_file_name` of the phonon-dos files is provided

## Minimal Example

If the required input files already exist, a minimal EGQCA calculation looks like this:

```python
import egqca
import numpy as np

m = egqca.EGQCA(
    datafile="path/to/clusters.dat",
    compound="A$_{1-x}$B$_x$C",
    volume_file="path/to/volume.dat",  # optional
    pressure_GPa=0.0,                   # optional
    phonon_file_name="/path/to/phonon/files"    # optional
)

x_grid = np.arange(1e-6, 1.0, 1.0e-3)
Tlist = [600, 800, 1000, 1400]

m.thermodynamic_assessment(x_grid, Tlist)

m.average_properties("path/to/properties.dat")
```

# Full Workflow

This following workflow is the recommended way to use `egqca`. It avoids manual preparation of `clusters.dat` and `volume.dat`, keeps the directory structure consistent, and streamlines the transition from cluster generation to ground state energy calculations and EGQCA post-processing.

When you follow this workflow, `egqca` generates all non-equivalent clusters, creates the corresponding folder structure, prepares input files for the DFT calculations (FHI-aims, Quantum ESPRESSO or VASP) or performs the relaxation with an specified MLIP, and writes the EGQCA input files in the correct format.

## Step 1 – Generate Clusters and DFT Input

```python
import egqca

out_dir = "/path/to/output/directory/"

egqca.set_up_clusters_and_dft(
    supercell_program="/path/to/supercell/executable",
    path_to_cif="/path/to/cif/file",
    supercell_size="2x2x1",
    out_dir=out_dir,
    aims_par={},                # optional
    qe_par={},                  # optional
    vasp_par={},                # optional
    kpts=(6, 6, 6),             # optional
)
```

This step requires:

- The path to the supercell executable, which can be downloaded from the [supercell website](https://orex.github.io/supercell/)
- The path to the CIF file of the structure. Under `_atom_site_occupancy`, the substituting atom must be listed at the same site as the original atom with occupancy `0`
- The cluster size. The supercell program generates all non-equivalent clusters for that size.
- The path to the output directory

Optional inputs:

- Parameters for the DFT input files through [`aims_par`](dft-files.md#fhi-aims), [`qe_par`](dft-files.md#quantum-espresso), [`vasp_par`](dft-files.md#vasp) and [`kpts`](dft-files.md#required-solver-specific-paths)

This step produces:
- The folder structure `S*/S*_i*_w*/` with one folder per non-equivalent cluster (`S*`: number of substituted atoms, `i*`: cluster index, `w*`: degeneracy)
- `clusters_template.dat`, the cluster setup that every route (DFT, ML) reads
- `DFT/clusters.dat` without energies, if DFT inputs or folders are requested
- The DFT input files if the parameters are given. 

More details on the cluster set up are given in [Cluster Setup](cluster-setup.md).

## Step 2 – Calculate ground state energies

Run DFT calculations for each cluster and store the output files, for example `scf.out` or `aims.out`, in the `vc-relax/` subfolder of the corresponding cluster folder (`S*/S*_i*_w*/vc-relax/`). EGQCA reads the energies and lattice parameters from `S*/S*_i*_w*/<dft_out_file_subfolder>/<dft_out_file_name>`; `dft_out_file_subfolder` defaults to `"vc-relax"` and can be changed when creating the `EGQCA` object (Step 3).

If you want to estimate relaxed energies and lattice parameters with the ML workflow instead, or in addition, run:

```python
import egqca

egqca.run_ml_relaxation(
    clusters_folder=out_dir,
    ML_model="mattersim-5m",                # optional
    out_folder_name=None,                   # optional
    pressure_GPa=None,                      # optional
    relaxation_constraint="FixSymmetry",    # optional
    save_relaxed_structure=True,            # optional
    calc_phonons=False,                     # optional
)
```

This writes everything into the run folder `out_folder_name` (default: `ML_model`), including `clusters_<model>.dat` and `volume_<model>.dat`. The same name always means the same folder: an incomplete run is continued, a complete one is kept and its results are returned. 

Note, each model family needs its own optional dependencies, installed into a separate environment, e.g. `pip install egqca[mattersim]`.
Further information can be found in [ML Workflow](ml-workflow.md).Any other MLIP with an ASE calculator can be used the same way ([Custom Calculators](ml-workflow.md#custom-calculators)).

The `cluster_importance()` function is available as an additional analysis step to identify a reduced cluster set that still captures the thermodynamic behavior well. It only requires the ground-state energies and returns a subset of the most important clusters based on their probability weight. See [Cluster Reduction](cluster-reduction.md) for the algorithm and output format.

## Step 3 – Run the EGQCA Calculation

`EGQCA` reads the cluster energies from the file given as `datafile`. Depending on Step 2, this is

- **ML energies:** `clusters_<ML-model>.dat` from the ML run folder, e.g. `mattersim-5m/clusters_mattersim-5m.dat`, with the matching `volume_<ML-model>.dat` as `volume_file` (see [From ML Results to EGQCA](ml-workflow.md#from-ml-results-to-egqca)), or
- **DFT energies:** `clusters.dat` in the DFT run folder (`DFT/` from Step 1, or the folder given to `update_DFT_in`), filled with the DFT energies as described below.

For DFT, the energies for each cluster are extracted from `./S*/S*_i*_w*/vc-relax/<dft_out_file_name>` and written automatically to `clusters.dat` whenever any energy in the file is still `missing`; all energies are then re-extracted. A cluster without folder, output file or energy in its output keeps its entry in `clusters.dat`; if any cluster is still `missing` afterwards, EGQCA writes the extracted energies and stops with an error listing the missing clusters. Of course, the energies can also be added manually to `clusters.dat`. Clusters whose DFT run did not finish successfully, or whose relaxation did not converge (QE: no `bfgs converged`, FHI-aims: no `Present geometry is converged.`, VASP: no `reached required accuracy`), are marked with a `# WARNING:` comment on their row; the warning is printed again whenever the file is read.

Likewise, `volume.dat` is written next to the clusters file (into the DFT run folder) from the same DFT output files, so the lattice parameters are available for the PV term at finite `pressure_GPa`.

```python
import egqca
import numpy as np

out_dir = "/path/to/output/directory/"

m = egqca.EGQCA(
    datafile=out_dir + "DFT/clusters.dat",
    dft_out_file_name="scf.out",
)

x_grid = np.arange(1e-6, 1.0, 1.0e-3)
Tlist = [600.0, 800.0, 1000.0, 1400.0]

m.thermodynamic_assessment(x_grid, Tlist)
m.average_properties("path/to/properties.dat")
```

- `thermodynamic_assessment(x_grid, Tlist)` computes thermodynamic quantities such as free energy, entropy, enthalpy, excess quantities, and the miscibility gap.
- `average_properties(properties_file, Tlist=None, x=None)` averages the properties listed in a separate properties file using the previously computed or newly calculated cluster probabilities. It can also compare several properties files or several cluster-reduction surrogate files in one plot; see [Properties Averaging](properties-averaging.md).

Each of the three entry points writes into its own subfolder of the output directory (`datafile.parent`, or `out_folder_name` below it):

| Method | Output folder |
|--------|---------------|
| `thermodynamic_assessment()` | `thermodynamic_assessment/` |
| `average_properties()` | `results/` |
| `cluster_importance()` | `cluster_reduction/` |

When `phonon_file_name` is given, `_phonons` is appended to the folder name and to every file written inside it, so a run with and a run without vibrational contributions never overwrite each other. Data files are written as `.dat`, figures as `.pdf`.

# Input Files

## `clusters.dat`

`clusters.dat` is the main input of EGQCA. The workflow above writes it automatically; if you write it by hand, follow the format described here.

By default, the header contains the following keywords:
- `compound= A\$\_{1-x}\$B\$\_{x}\$C`
- `total number of sites= 12` 
- `permutable sites n= 8` 

If this information is not included in the header, it must be provided explicitly when initializing the `EGQCA` class. See also [docstrings](docstrings.md). Files written by the ML workflow carry one extra header line naming the model, `# uMLIP model for relaxation: <model>`.

### Required Columns

| Column | Description |
|--------|------------|
| **ID** | Cluster identifier |
| **n_j** | Number of substituted atoms in the cluster |
| **Nat** | Total number of atoms in the cluster |
| **x_j** | Cluster composition |
| **g_j** | Degeneracy of the cluster |
| **Total Energy** | Total energy of the cluster (in Ry), or `missing` |

All energies are converted to meV per site using the `total number of sites` from the header. A `missing` energy requires `dft_out_file_name`, from which all energies are then extracted; without it, `missing` entries are rejected. Entries that are not finite numbers (e.g. `nan` or a typo) are rejected as well.

This file contains the information required to compute the cluster occurrence probabilities

$$
p(x, T)
$$

by minimizing the Gibbs free energy.

## `volume.dat` (Optional, for PV term)

Required columns:

| Column | Description |
|--------|------------|
| **ID** | Cluster identifier |
| **a (Å)** | Lattice parameter |
| **b (Å)** | Lattice parameter |
| **c (Å)** | Lattice parameter |
| **alpha (°)** | Lattice angle |
| **beta (°)** | Lattice angle |
| **gamma (°)** | Lattice angle |

Used to compute the volume-dependent enthalpy contribution. The file is only read when `pressure_GPa` is non-zero. `convert_volume_file()` rewrites such a file through a user-supplied conversion function, for example to turn the lattice parameters of a supercell into those of the primitive cell.

## Phonon File (Optional)

`phonon_file_name` supports two layouts, detected automatically:

- **Folder mode** (ML workflow): pass a directory path; EGQCA reads `<dir>/<cluster_id>.dos` for each cluster. This matches the output of `run_ml_relaxation(..., calc_phonons=True)`.
- **Per-subfolder mode** (DFT workflow): pass a file name; EGQCA reads `S*/S*_i*_w*/phonons/<phonon_file_name>` for each cluster.

Requirements on each DOS file:
- Column 1: phonon energy or frequency
- Column 2: phonon DOS
- A header line stating the unit of column 1: `eV`, `cm^-1`, or `Hz`. The unit is detected from the first matching line of the file; without it the file is rejected. Internally the values are converted to the angular frequency $\omega$, and each DOS is rescaled to exactly 3 modes per site. A warning lists the files that were off by more than 5 %, which usually means a wrong file, a wrong number of sites, or an ML DOS from ASE ≤ 3.24.

These data are used to compute the vibrational free energy contribution. Modes with $\omega \le 0$, i.e. imaginary frequencies of a dynamically unstable cluster, are dropped after this rescaling, and a warning reports how many clusters are affected and how much DOS weight was excluded.

# Summary of Workflow

1. Generate clusters and DFT input files with the automated setup.
2. Run the DFT calculations for the generated clusters, and optionally run the ML relaxation workflow or an EDDP relaxation.
3. Run `thermodynamic_assessment(...)` to extract energies and compute thermodynamic properties.
4. Run `average_properties(...)` if you want to average additional properties on the thermodynamic grid.
5. Compare several energy sources — DFT, one or more ML models, EDDP — or both entropy approximations with the helpers in the comparison module (see [Entropy Approximation](entropy.md) and [docstrings](docstrings.md)).
6. Use manual file preparation only for custom or advanced workflows.
