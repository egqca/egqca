"""Step 1: generate all non-equivalent 8-atom clusters of bcc Nb_{1-x}V_x.

Needs the external supercell program: https://orex.github.io/supercell/
Set its path below.
"""

from pathlib import Path

import egqca

here = Path(__file__).resolve().parent

egqca.set_up_clusters_and_dft(
    supercell_program="/path/to/supercell",   # the external supercell binary
    path_to_cif=here / "Nb_bcc.cif",          # parent cell: bcc Nb, V on the Nb site with occupancy 0
    supercell_size="2x2x1",                   # 2x2x1 conventional cells = 8 atoms
    out_dir=here / "Nb1-xVx_2x2x1",           # must not exist yet or be empty
    create_dft_folders=False,                 # ML-only run; True also creates vc-relax/, bands/, phonons/
    # qe_par=..., aims_par=..., vasp_par=..., kpts=(6, 6, 6)   # optional DFT input files, see docs/dft-files.md
)

# Creates
#   Nb1-xVx_2x2x1/S*/S*_i*_w*/S*_i*_w*.cif   one folder per cluster
#                                            (S: number of V atoms, i: index, w: degeneracy g_j)
#   Nb1-xVx_2x2x1/clusters_template.dat      cluster list read by the DFT/ML run folders
# These are the same 22 clusters, with the same IDs, as in clusters_pet-omatpes-l.dat.
