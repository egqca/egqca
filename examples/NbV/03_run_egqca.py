"""Step 3: EGQCA thermodynamics of bcc Nb_{1-x}V_x from the pet-omatpes-l data.

1) mixing energies + configurational entropy, plus averaged lattice parameters
2) as 1), plus the vibrational free energy from the phonon DOS
3) as 2), plus the PV term at 10 GPa

To use your own results from 02_ml_relaxation_phonons.py instead, point the paths
to Nb1-xVx_2x2x1/pet-omatpes-l/ and its phonons/ folder.
"""

from pathlib import Path

import numpy as np

import egqca

here = Path(__file__).resolve().parent
clusters = here / "clusters_pet-omatpes-l.dat"
volumes = here / "volume_pet-omatpes-l.dat"
compound = "Nb$_{1-x}$V$_x$"                 # only used for plot labels

x_grid = np.arange(1e-6, 1.0, 1.0e-3)
Tlist = [600, 800, 1000, 1400]

# --- 1) without phonons ------------------------------------------------------ #
m = egqca.EGQCA(datafile=clusters, compound=compound)
m.thermodynamic_assessment(x_grid, Tlist)
# results (.dat + .pdf) in thermodynamic_assessment/

# lattice parameters a, b, c averaged with the occurrence probabilities p_j(x, T)
m.average_properties(
    volumes,                                 # any file with the cluster ID in column 0
    columns=[0, 1, 2],                       # a, b, c (0-based, ID column excluded); skips the angles
    ylabels=[r"a\ (\mathrm{\AA})", r"b\ (\mathrm{\AA})", r"c\ (\mathrm{\AA})"],
)
# results in results/

# --- 2) with phonons --------------------------------------------------------- #
m_vib = egqca.EGQCA(
    datafile=clusters,
    compound=compound,
    phonon_file_name=here / "ph_dos",        # folder with <cluster_id>.dos files
)
m_vib.thermodynamic_assessment(x_grid, Tlist)
# results in thermodynamic_assessment_phonons/

# --- 3) with phonons + PV term ----------------------------------------------- #
m_pv = egqca.EGQCA(
    datafile=clusters,
    compound=compound,
    phonon_file_name=here / "ph_dos",
    volume_file=volumes,                     # cell volumes of the relaxed clusters
    pressure_GPa=10.0,                       # the volume file is ignored at 0 GPa
    out_folder_name="P10GPa",                # keeps the results apart from run 2
)
m_pv.thermodynamic_assessment(x_grid, Tlist)
# results in P10GPa/thermodynamic_assessment_phonons/
