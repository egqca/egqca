"""Step 2: relax every cluster with an ML potential and compute its phonon DOS.

Run 01_generate_clusters.py first. The PET models need the `upet` package.

The result of this step is already included in this folder:
    clusters_pet-omatpes-l.dat   energies (Ry)
    volume_pet-omatpes-l.dat     lattice parameters
    ph_dos/<cluster_id>.dos      phonon DOS
so 03_run_egqca.py can be run without it.
"""

from pathlib import Path

import egqca

here = Path(__file__).resolve().parent

energies, phonon_dos = egqca.run_ml_relaxation(
    clusters_folder=here / "Nb1-xVx_2x2x1",   # created by 01_generate_clusters.py
    ML_model="pet-omatpes-l",                 # any built-in model string or an ASE calculator
    calc_phonons=True,                        # phonon DOS of each relaxed cluster
    N_phonon_supercell=2,                     # phonon supercell: N x N x N, or [N1, N2, N3]
    relaxation_constraint=True,               # keep the Bravais lattice of each cluster
    pressure_GPa=None,                        # relax at 0 GPa
    device="cpu",                             # "cuda" if a GPU is available
)

# Creates the run folder Nb1-xVx_2x2x1/pet-omatpes-l/ with
#   clusters_pet-omatpes-l.dat   energies in Ry
#   volume_pet-omatpes-l.dat     lattice parameters
#   phonons/<cluster_id>.dos     phonon DOS
#   cif_relaxed/, ML_logs/       relaxed structures and optimizer logs
# An interrupted run continues where it stopped when the script is started again.
