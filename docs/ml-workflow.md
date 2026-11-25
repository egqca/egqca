# ML Workflow

EGQCA can relax clusters with machine-learned interatomic potentials and, optionally, compute their phonon DOS. 

UPET, MACE, MatterSim, GRACE and EDDP models are supported out of the box (see [Available Models](#available-models)). You can also bring your own ASE calculator, e.g. a fine-tuned model (see [Custom Calculators](#custom-calculators)). Each built-in family has its own optional extra:

```bash
pip install egqca[upet]        # UPET (PET models)
pip install egqca[mace]        # MACE
pip install egqca[mattersim]   # MatterSim (the default model)
pip install egqca[grace]       # GRACE
```

The backends are not compatible with each other: UPET, MACE and MatterSim need their own PyTorch versions and GRACE (`tensorpotential`) uses TensorFlow. Install each model family into its own environment.

GRACE models are downloaded on first use to `$GRACE_CACHE` (default `~/.cache/grace`). On compute nodes without internet access, point `GRACE_CACHE` to a shared folder and download the model beforehand, e.g. `grace_models download GRACE-3L-OMAT-large-ft-AM`.

## Geometry Relaxation

Each cluster is relaxed with `FrechetCellFilter` (plus a scalar pressure if `pressure_GPa` is set) and an ASE optimizer, starting from the cluster's CIF in the usual layout `S*/S*_i*_w*/S*_i*_w*.cif`. There are two ways to run the relaxations:

| | Sequential Run | Parallel Job Array |
| --- | --- | --- |
| Call | `egqca.run_ml_relaxation()` in Python | `scripts/ml_relaxation_parallel.py`, then `collect_ml_results()` |
| Clusters | one after another in one process | one cluster or a small group of clusters per job, e.g. in a SLURM job array |
| ML model | loaded once and shared by all clusters | loaded once per job |
| Models | model strings and [custom calculators](#custom-calculators) | model strings and [custom calculators](#custom-calculators) |
| Use for | small and medium cluster sets, a single GPU | large cluster sets on a cluster |

Both write the same [run folder](#run-folder-and-resuming) and take the same [parameters](#parameters).

### Sequential Run

```python
import egqca

egqca.run_ml_relaxation(
    clusters_folder="/path/to/cluster/folder",
    ML_model="mattersim-5m",
    pressure_GPa=300,
)
```

`run_ml_relaxation()` relaxes all clusters one after another. The ML calculator is loaded once per `(model, device)` and shared by all relaxations and phonon runs.

### Parallel Job Array

For a SLURM job array, run one script call per job and collect the results afterwards:

```bash
# one job per cluster
python scripts/ml_relaxation_parallel.py \
    /path/to/clusters/S2/S2_i03_w4 \
    /path/to/clusters/mattersim-5m \
    --model mattersim-5m --steps 5000 --fmax 1e-4

# or a group of clusters per job
python scripts/ml_relaxation_parallel.py \
    /path/to/clusters/S2/S2_i03_w4 /path/to/clusters/S2/S2_i04_w8 /path/to/clusters/S2/S2_i05_w2 \
    /path/to/clusters/mattersim-5m \
    --model mattersim-5m

# once all jobs are done; the argument is the run folder from above
python scripts/collect_ml_results.py /path/to/clusters/mattersim-5m --model mattersim-5m
```

The arguments are one or more cluster folders followed by the run folder. Several clusters in one call are relaxed one after another in the same process, so the ML model is loaded only once per job; this pays off when loading the model takes longer than relaxing a cluster. A failing cluster does not stop the others: the script continues and exits with an error at the end that lists the failed clusters. The script takes the same settings and defaults as `run_ml_relaxation()` (see [Parameters](#parameters) and `--help`). Each job writes its result to `<run_folder>/per_cluster/<cluster_id>_<model>_ml_result.dat`, next to the usual logs, CIF and phonon files.

`collect_ml_results()` (the script or `egqca.collect_ml_results()`) merges these into `clusters_<model>.dat` and `volume_<model>.dat` and reports missing clusters. The cluster setup and the list of all clusters come from `<clusters folder>/clusters_template.dat`; pass `clusters_template=` to use another file.

Two more scripts: `ml_relaxation_one_file.py` relaxes a single CIF file without assuming the folder strucutre, and `ml_rerun_nonconverged.py` relaxes the non-converged clusters of a run folder again (see [Redoing Non-Converged Clusters](#redoing-non-converged-clusters)).

### Redoing Non-Converged Clusters

A cluster whose relaxation did not converge is flagged in `clusters_<model>.dat` and counts as done, so a resumed run skips it. To relax it again, reset it first: `reset_ml_clusters()` removes its rows, its `per_cluster/` result file and its phonon DOS, and lists it in `<clusters_folder>/rerun_clusters.txt`. Without `cluster_ids`, it resets every non-converged cluster, including results that have not been collected yet.

- Parallel workflow: collect with `--reset-nonconverged`, queue `rerun_clusters.txt` (same format as `missing_clusters.txt`) with `ml_relaxation_parallel.py --start-from-relaxed` and a higher `--steps`, then collect again. `steps` is not part of the recorded run settings, so it may differ from the first run.
- Sequential workflow: `rerun_nonconverged_ml()`, or the script `ml_rerun_nonconverged.py <clusters_folder> --model <model> [--model-name ...] [--device cuda] [--steps 20000]`, resets the clusters and relaxes them with the recorded settings. The model and its name must be those of the original run; a calculator factory is given as `py:<module or file.py>:<function>`, as for the parallel script. `device` is not recorded, so pass it again for a GPU run.

Both continue each cluster from its last structure in `cif_relaxed/` (`start_from_relaxed=True`, `--start-from-relaxed`), so the steps already done are not lost; `--from-original` restarts from the original CIF. `optimizer_kwargs` values that are not plain JSON (e.g. a `Path`) are recorded only as text and must be passed again for a rerun.

The header warning "Relaxation did not converge for some of the clusters" in `clusters_<model>.dat` and `volume_<model>.dat` is kept in step with the rows: it is present exactly while a row is flagged.

### Run Folder and Resuming

All output goes into one run folder inside `clusters_folder`, named after the model unless `out_folder_name` is given:

```text
<run_folder>/
├── clusters_<model>.dat                  relaxed energies
├── volume_<model>.dat                    relaxed lattice parameters
├── cif_relaxed/<cluster_id>_<model>.cif  with save_relaxed_structure=True (default)
├── phonons/<cluster_id>.dos              with calc_phonons=True
├── phonons/<cluster_id>_cache/           with calc_phonons=True and save_phonon_cache=True (default)
└── ML_logs/<model>.log, ML_logs/clusters/<cluster_id>.log
```

An `eddp:<path>` model is named after its file, e.g. `eddp:/models/NbV_v3.eddp` becomes `eddp-NbV_v3`. Its path and a sha256 of the `.eddp` file and all its `.ddp` members are stored in `<run folder>/model_identity.json` (and in the run log); a run folder is never resumed with a different model file, even one with the same name. If the run folder exists but is incomplete, the run continues it and appends to the existing log files; clusters that only lack the phonon DOS reuse their relaxed CIF instead of being relaxed again. A run folder is only ever continued with the settings it was started with: the first run records them in `<run folder>/run_settings.json` (model, pressure, relaxation constraint, cell mask, Bravais supercell and positions, `fmax`, optimizer and its arguments) and, once phonons are requested, the phonon supercell in `phonons/phonon_settings.json`. A run or per-cluster job with other settings stops with an error that lists the differences; use another `out_folder_name` / `output_dir` for a new run. Only `steps` and `device` may differ, and phonons may be added to a run that relaxed without them. A complete run folder is kept and its results are returned; the same name always means the same folder, in the sequential run, the parallel script and the rerun. Use another `out_folder_name` for a new run with the same model.

A cluster counts as done once its result is in these files. The `.dos` files, per-cluster result files, relaxed CIFs and phonon cache metadata are written to a temporary file and renamed into place, so a job killed while writing leaves no partial file that would count as done; an interrupted row at the end of `clusters_<model>.dat` or `volume_<model>.dat` is ignored and removed by the next append. Run folders from before this was the case may hold a truncated `.dos` file of a killed job; delete the `.dos` files of clusters whose job was killed, and they are computed again.

### Parameters

| Parameter | Script option | Default | Meaning |
| --- | --- | --- | --- |
| `clusters_folder` | (cluster folder, 1st argument) | required | Folder with the `S*/S*_i*_w*/` cluster tree; the run folder is created inside it. |
| `ML_model` | `--model` | `"mattersim-5m"` | Model string (see [Available Models](#available-models)), an ASE calculator, or a function returning one (see [Custom Calculators](#custom-calculators)). The script takes a model string or a function as `py:<module or file.py>:<function>`. |
| `ML_model_name` | `--model-name` | `None` | Name of the model in the run folder, file names and logs. Required for a calculator or function; defaults to the model string. |
| `out_folder_name` | (run folder, last positional argument) | `None` | Run folder name; defaults to the model name. |
| `pressure_GPa` | `--pressure-gpa` | `None` | External pressure during the relaxation, in GPa. |
| `steps` | `--steps` | `5000` | Maximum optimizer steps per cluster. |
| `fmax` | `--fmax` | `1e-4` | Force convergence threshold in eV/Å. |
| `device` | `--device` | `"cpu"` | `"cpu"` or `"cuda"`; only for model strings. |
| `optimizer` / `optimizer_kwargs` | `--optimizer` / `--optimizer-kwargs` | `FIRE` / `None` | ASE optimizer class and its arguments, e.g. `optimizer=BFGS`, `optimizer_kwargs={"maxstep": 0.1}`. The script takes the class name and the arguments as a JSON object, e.g. `--optimizer-kwargs '{"maxstep": 0.1}'`. |
| `relaxation_constraint` | `--relaxation-constraint` | `"FixSymmetry"` | Symmetry constraint, see [Symmetry Constraints](#symmetry-constraints). |
| `CellFilter_mask` | `--cell-filter-mask` | `None` | Six bools for the Cartesian strain components `[xx, yy, zz, yz, xz, xy]` (ASE Voigt order), `True` = free; `None` leaves all free. These are not the cell parameters `a, b, c, alpha, beta, gamma`; they only coincide for an orthogonal, axis-aligned cell. |
| `bravais_supercell` | `--bravais-supercell` | `(1, 1, 1)` | Supercell factors of the input relative to the primitive cell (Bravais constraint only). |
| `fix_bravais_positions` | `--fix-bravais-positions` | `False` | Under a Bravais constraint, also freeze the atomic positions at their starting fractional coordinates (no internal relaxation, only the cell relaxes). |
| `bravais_tol` | `--bravais-tol` | `0.01` | Relative tolerance of the Bravais lattice detection (Bravais constraint only). |
| `save_relaxed_structure` | `--dont-save-relaxed-cif` | `True` | Write the relaxed CIFs. |
| `start_from_relaxed` | `--start-from-relaxed` | `False` | Start a cluster that is relaxed (again) from its structure in `cif_relaxed/`, if there is one. |
| `calc_phonons` | `--phonons` | `False` | Compute a phonon DOS per cluster after the relaxation, see [Phonons](#phonons). |
| `N_phonon_supercell` | `--supercell` | `2` | Phonon supercell: `N` for `N x N x N`, or `[N1, N2, N3]`. |
| `save_phonon_cache` | `--no-phonon-cache` | `True` | Keep the phonon forces so the DOS can be re-sampled later. |
| `ml_clusters_filename` / `ml_volume_filename` | – | `None` | Names of the result files; default `clusters_<model>.dat` / `volume_<model>.dat`. |

The script has one option of its own, `--permutable-sites`. Every cluster always gets its log `<run folder>/ML_logs/clusters/<cluster_id>.log`; `collect_ml_results` reads the settings, the Bravais-lattice check and the runtime from it. The number of permutable sites n (for x = n_j / n) is read from `<clusters folder>/clusters_template.dat`, which `set_up_clusters_and_dft` computes from the CIF. A tree without it gets it written once, from `DFT/clusters.dat` of an older setup or from a scan of the cluster CIFs with n = the highest `S*` folder number. Both workflows use the same rule. If the `S*` folders cover only a sub-range of compositions and there is no template, the folder number is wrong: pass `--permutable-sites` (or `permutable_sites` in `run_ml_relaxation` and `collect_ml_results`), which then goes into the template. A value that differs from the template is an error.

### Symmetry Constraints

`relaxation_constraint` accepts:

- `"FixSymmetry"` (default): ASE's `FixSymmetry`, keeps the space group of each cluster.
- `None`: no constraint; the cell is controlled only by `CellFilter_mask`.
- a Bravais-lattice key or crystal system, e.g. `"RHL"` or `"rhombohedral"`: a `FixBravaisCell` constraint that keeps the cell metric of that Bravais type. The constraint keeps the symmetry of the initial cell, it cannot impose a new one: the initial cell (S0) must already be of the requested type (within `bravais_tol`), otherwise the run stops with an error. This is also required for the cluster set, since the supercell program generates the clusters from the symmetry of S0. Set `bravais_supercell` for supercell inputs.
- `True` (`"auto"` in the scripts): like the above, with the Bravais type detected from each cluster's initial cell. Only the cell geometry is used, so all clusters built from one template get the same Bravais type, even where the decoration lowers the symmetry. This keeps all clusters on one cell family; use an explicit key or `"FixSymmetry"` otherwise.

The Bravais type of the initial cell is detected with a relative tolerance `bravais_tol` (default `0.01`, about 1 % in cell lengths and angles). A cell within this tolerance of a higher-symmetry lattice gets the full symmetry of that lattice and is projected onto it, e.g. a cubic cell with 0.25 % strain relaxes as an exactly cubic cell. The detected type is the one printed as initial and target lattice.

Combining `relaxation_constraint` with `CellFilter_mask` applies both and can interact in unexpected ways; EGQCA warns about it.


## Available Models

| Model string | Backend | Notes |
| --- | --- | --- |
| `pet-mad-xs`, `pet-mad-s` | UPET | PET-MAD, broad materials coverage. |
| `pet-omat-xs`, `pet-omat-s`, `pet-omat-m`, `pet-omat-l`, `pet-omat-xl` | UPET | PET-OMat, inorganic materials. |
| `pet-oam-l`, `pet-oam-xl` | UPET | PET-OAM; `pet-oam-xl` is a common high-accuracy choice. |
| `pet-omad-xs`, `pet-omad-s`, `pet-omad-l` | UPET | OMat/MAD variants. |
| `pet-omatpes-l` | UPET | OMatPES. |
| `pet-spice-s`, `pet-spice-l` | UPET | SPICE family. |
| `mace`, `mace-mp` | MACE | Alias for `mace-medium-mpa-0`. |
| `mace-small`, `mace-medium`, `mace-large` | MACE | MACE-MP-0; MIT. |
| `mace-small-0b`, `mace-medium-0b` | MACE | MACE-MP-0b; MIT. |
| `mace-small-0b2`, `mace-medium-0b2`, `mace-large-0b2` | MACE | MACE-MP-0b2; MIT. |
| `mace-medium-0b3` | MACE | MACE-MP-0b3; MIT. |
| `mace-medium-mpa-0` | MACE | MACE-MPA-0; MIT. Recommended default. |
| `mace-small-omat-0`, `mace-medium-omat-0` | MACE | MACE-OMAT-0; ASL. Most accurate for phonons. |
| `mace-omat-0` | MACE | Alias for `mace-medium-omat-0`. |
| `mace-matpes-pbe-0`, `mace-matpes-r2scan-0` | MACE | MACE-MATPES-0; ASL. |
| `mace-mh-0`, `mace-mh-1` | MACE | MACE-MH multihead, `matpes_r2scan` head; ASL. |
| `mace-mp-<url-or-path>` | MACE | Custom checkpoint URL or file. |
| `mattersim-1m`, `mattersim-5m` | MatterSim | MatterSim v1, 1M and 5M parameters. |
| `GRACE-<name>` | GRACE | Any model from `grace_models list`, case-insensitive, e.g. `GRACE-3L-OMAT-large-ft-AM` (ASL, fp32 only). |
| `eddp:<path-to-model.eddp>` | AIRSS `repose` | Local EDDP ensemble, evaluated by `repose` for energies/forces/stress; the `.ddp` directory from the manifest must sit next to it. |

The available foundation models change upstream; if a model string stops working, check the backend documentation: [UPET](https://pypi.org/project/upet/), [MACE](https://github.com/ACEsuit/mace-foundations), [MatterSim](https://microsoft.github.io/mattersim/), [GRACE](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/).

!!! note "GRACE"
    TensorFlow has no per-calculator device setting, so for `device="cpu"` EGQCA hides the GPUs from TensorFlow. This only works before TensorFlow has initialised them, so use one device per Python process. CPU threads can be limited with `TF_NUM_INTRAOP_THREADS` and `TF_NUM_INTEROP_THREADS`.

## Custom Calculators

`run_ml_relaxation` is not limited to the models in [Available Models](#available-models). `ML_model` accepts

- a **model string** of a built-in family (table above),
- any **ASE calculator** object, or
- a **function or class** that returns an ASE calculator when called without arguments.

So any MLIP with an ASE interface can be used, e.g. a fine-tuned or self-trained MACE, NequIP or Allegro model, SevenNet, ORB, CHGNet, or a classical/ML potential in LAMMPS through ASE's LAMMPS calculator. The calculator must provide **energy, forces and stress**.

```python
from functools import partial
from mace.calculators import MACECalculator

egqca.run_ml_relaxation(
    "/path/to/cluster/folder",
    ML_model=partial(MACECalculator, model_paths="my_finetuned.model", default_dtype="float64"),
    ML_model_name="mace-finetuned",
)
```

The custom calculator goes through exactly the same workflow as a built-in model: each cluster CIF is relaxed with `FrechetCellFilter` (plus `pressure_GPa`, `relaxation_constraint` and `CellFilter_mask` as set), and the run folder gets the same `clusters_<name>.dat`, `volume_<name>.dat`, relaxed CIFs, logs and, with `calc_phonons=True`, phonon DOS files. The results go into `EGQCA` as described in [From ML Results to EGQCA](#from-ml-results-to-egqca).

- `ML_model_name` is required. It names the run folder and output files and identifies the run when resuming, so give different models different names.
- One calculator instance serves all clusters and their phonons. A function is only called once a cluster needs relaxing, so resuming a finished run never loads the model.
- `device` is not applied; configure device and precision yourself when building the calculator. Double precision is recommended for the default `fmax=1e-4`.
- `geometry_relaxation()` and `phonons()` accept the same arguments.
- The script of the [Parallel Job Array](#parallel-job-array) takes the function as `--model py:<module or file.py>:<function>`, with `--model-name` for `ML_model_name`. The function must build the calculator when called without arguments, so put the model path and settings inside it:

```python
# my_calc.py
from mace.calculators import MACECalculator

def make_calculator():
    return MACECalculator(model_paths="my_finetuned.model", default_dtype="float64")
```

```bash
python scripts/ml_relaxation_parallel.py <cluster folders> /path/to/clusters/mace-finetuned \
    --model py:/path/to/my_calc.py:make_calculator --model-name mace-finetuned
python scripts/collect_ml_results.py /path/to/clusters/mace-finetuned --model mace-finetuned
```

## EDDP Relaxations Outside egqca

There are two ways to use an EDDP (ephemeral data-derived potential, AIRSS):

- **Inside egqca:** `ML_model="eddp:<path-to-model.eddp>"` in `run_ml_relaxation`. `repose` then serves as the calculator, and the run behaves like any other model (see [Available Models](#available-models)).
- **Outside egqca:** if the clusters were already relaxed with AIRSS `repose` on their own, `extract_eddp_energies` only collects the energies from the `repose` logs. This is described below.

Once each cluster folder holds its `repose` log as `<cluster>/eddp/energy.dat` (another file name can be selected with `energy_filename`), build the EGQCA input file with:

```python
egqca.extract_eddp_energies(out_dir)
```

The number of permutable sites, the number of sites, the atoms per cluster and the compound are read from `<out_dir>/clusters_template.dat`; they can be passed as `permutable_sites`, `sites`, `Nat` and `compound` for a tree without template (a value that differs from an existing template is an error).

This writes `eddp/clusters_eddp.dat` in the same format as `clusters.dat`, converting the `Energy:` line of each log from eV to Ry. Clusters without a converged relaxation are additionally listed in `eddp/not_converged_clusters.dat`, one cluster ID per line with the reason and a per-`S*` breakdown in the header; their energies are reported as `missing` whenever the log has no `Energy:` line at all. No `volume` file is written, so this route provides no lattice parameters for the PV term.

For other MLIPs run outside egqca there is no dedicated reader. Either wrap the model as an ASE calculator and use [Custom Calculators](#custom-calculators), or enter the energies (in Ry) into `clusters.dat` yourself.

## Phonons

With `calc_phonons=True`, each relaxed cluster gets a finite-displacement phonon calculation with ASE's `Phonons` (constraints removed, displacement `delta=0.01` Å, supercell from `N_phonon_supercell`). The DOS is sampled on a 20x20x20 q-grid with 500 points and 0.1 meV smearing and written to `phonons/<cluster_id>.dos` as two columns, `Energy(eV) DOS`. See the [ASE phonon documentation](https://ase-lib.org/ase/phonons.html).

The DOS integrates to 3 modes per atom, including imaginary (negative-energy) modes of dynamically unstable clusters. The global log lists those clusters, and when EGQCA reads the files these modes are dropped with a warning.

With `save_phonon_cache=True` the forces are kept in `phonons/<cluster_id>_cache/`. An interrupted phonon run resumes from there (a cache of another structure, supercell, displacement or model is discarded instead), and the DOS can be re-sampled without calling the ML model:

```python
egqca.phonon_dos_from_cache(
    f"{run_folder}/phonons/S2_i03_w4_cache",
    kpts=(30, 30, 30), npts=2000, width=5e-4,
    save_dos=f"{run_folder}/phonons/S2_i03_w4.dos",
)
```

## From ML Results to EGQCA

Pass the run folder's files to `EGQCA`. A directory as `phonon_file_name` is read as `<dir>/<cluster_id>.dos`; a plain file name is looked up as `S*/S*_i*_w*/phonons/<phonon_file_name>` (DFT layout).

```python
import numpy as np
import egqca

base = "/path/to/clusters"
egqca.run_ml_relaxation(base, ML_model="mace-medium-mpa-0", calc_phonons=True, pressure_GPa=300)

run_folder = f"{base}/mace-medium-mpa-0"
m = egqca.EGQCA(
    datafile=f"{run_folder}/clusters_mace-medium-mpa-0.dat",
    volume_file=f"{run_folder}/volume_mace-medium-mpa-0.dat",
    pressure_GPa=300,
    phonon_file_name=f"{run_folder}/phonons",
)
m.thermodynamic_assessment(np.arange(1e-6, 1.0, 1e-3), [800.0])
```

Record the exact model string (and checkpoint version) in reports, and converge `N_phonon_supercell` for production results.
