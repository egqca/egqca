import scipy.constants as cst

############################################################################
# ------------------------------ Constants  ------------------------------ #
############################################################################
Ry_to_mev = cst.value('Rydberg constant times hc in eV')*1000       # meV
kB = cst.value('Boltzmann constant in eV/K')*1000                   # meV/K
hbar = cst.value('reduced Planck constant in eV s')*1000            # meV*s
cm_minus1_to_Hz = cst.c * 100.0                                     # Hz per cm^-1
meV_to_Hz = 1 / (cst.value('Planck constant in eV/Hz')*1000)        # Hz per meV
J_to_meV = 1.0 / cst.e *1000                                        # meV/J
angstrom_to_meter = 1.0e-10                                         # m
GPA_TO_PA = 1.0e9                                                   # Pa


############################################################################
# ----------------------------- DFT defaults ----------------------------- #
############################################################################
DEFAULT_KPTS = (12, 12, 12)

QE_DEFAULTS = {
    "control": {
        "calculation": "vc-relax",
        "prefix": "job",
        "outdir": "./tmp",
        "tstress": True,
        "tprnfor": True,
        "etot_conv_thr": 1e-7,
        "forc_conv_thr": 1e-6,
    },
    "system": {
        "occupations": "smearing", 
        "smearing": "mp", 
        "degauss": 0.02,
    },
    "electrons": {
        "conv_thr": 1e-10, 
        "mixing_beta": 0.7 , 
        "electron_maxstep": 200},
    "ions": {
        "ion_dynamics": "bfgs"},
    "cell": {
        "cell_dynamics": "bfgs", 
        "cell_dofree": 'all' , 
        "press": 0},
}

QE_PHONON_DEFAULTS = { 
    "fildvscf": "dvscf_q",
    "outdir": "./tmp",
    "ldisp": True,
    "tr2_ph": 1e-14,
    "nq1": 2,
    "nq2": 2,
    "nq3":2,
}

# in Hartree
PSEUDO_DOJO_ECUTWFC = {
    "H": 42,
    "He": 49,
    "Li": 52,
    "Be": 59,
    "B": 44,
    "C": 45,
    "N": 48,
    "O": 52,
    "F": 48,
    "Ne": 50,
    "Na": 48,
    "Mg": 48,
    "Al": 26,
    "Si": 24,
    "P": 28,
    "S": 32,
    "Cl": 33,
    "Ar": 37,
    "K": 43,
    "Ca": 38,
    "Sc": 45,
    "Ti": 46,
    "V": 48,
    "Cr": 66,
    "Mn": 69,
    "Fe": 65,
    "Co": 56,
    "Ni": 58,
    "Cu": 60,
    "Zn": 48,
    "Ga": 61,
    "Ge": 68,
    "As": 58,
    "Se": 56,
    "Br": 44,
    "Kr": 34,
    "Rb": 29,
    "Sr": 36,
    "Y": 42,
    "Zr": 49,
    "Nb": 49,
    "Mo": 46,
    "Tc": 48,
    "Ru": 50,
    "Rh": 50,
    "Pd": 49,
    "Ag": 47,
    "Cd": 66,
    "In": 49,
    "Sn": 57,
    "Sb": 58,
    "Te": 57,
    "I": 44,
    "Xe": 42,
    "Cs": 29,
    "Ba": 28,
    "La": 65,
    "Lu": 58,
    "Hf": 72,
    "Ta": 72,
    "W": 41,
    "Re": 42,
    "Os": 43,
    "Ir": 40,
    "Pt": 50,
    "Au": 44,
    "Hg": 50,
    "Tl": 50,
    "Pb": 50,
    "Bi": 50,
    "Po": 50,
    "Rn": 42,
}


# ----- FHI-aims defaults (control.in) -----
AIMS_DEFAULTS = dict(
    xc="pbe",
    relax_geometry="bfgs 1e-3",  # force threshold in eV/Å in aims
    relax_unit_cell="full",
    compute_forces=True,
)


# ----- VASP defaults (INCAR) -----
VASP_DEFAULTS = {
    "SYSTEM": "vasp",
    "NCORE": 1,
    "ISMEAR": 0,        # ISMEAR determines how the partial occupancies fnk are set for each orbital.
                        #  ISMEAR = 0: Gaussian smearing with width SIGMA.
    "SIGMA": 0.1,
    "ENCUT": 600,
    "IBRION": 2,        # determines how the crystal structure changes during the calculation: IBRION=2 conjugate gradient
    "ISIF": 3,          # Determines if the stress tensor is calculated and which ionic degrees of freedom are varied. 3 = all
    "NSW": 15,           # NSW sets the maximum number of ionic steps.
    "EDIFFG": -0.0001,  # EDIFFG defines the break condition for the ionic relaxation loop.
                        # negative: stopped when the norms of all the forces are smaller than |EDIFFG|
    "EDIFF": 1E-07,
    "LREAL": "Auto"
}


############################################################################
# ---------------------------- Atomic masses ----------------------------- #
############################################################################
amu = cst.atomic_mass
ELEMENT_ORDINALS = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,
    "Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,"Sc":21,
    "Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,
    "Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,
    "Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,
    "Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,
    "Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,
    "Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,
    "Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,"Pa":91,
    "U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,"Bk":97,"Cf":98,"Es":99,"Fm":100,
    "Md":101,"No":102,"Lr":103,"Rf":104,"Db":105,"Sg":106,"Bh":107,"Hs":108,
    "Mt":109,"Ds":110,"Rg":111,"Cn":112,"Fl":114,"Lv":116
    }

ATOMIC_MASSES = {
"H":1.00794,"He":4.002602,"Li":6.941,"Be":9.0122,"B":10.811,"C":12.0107,"N":14.0067,"O":15.999,
"F":18.9984032,"Ne":20.1797,"Na":22.989769,"Mg":24.305,"Al":26.981538,"Si":28.0855,"P":30.973762,
"S":32.065,"Cl":35.453,"Ar":39.948,"K":39.0983,"Ca":40.078,"Sc":44.955912,"Ti":47.867,"V":50.9415,
"Cr":51.9961,"Mn":54.938045,"Fe":55.845,"Co":58.933195,"Ni":58.6934,"Cu":63.546,"Zn":65.38,
"Ga":69.723,"Ge":72.64,"As":74.92160,"Se":78.96,"Br":79.904,"Kr":83.798,"Rb":85.4678,"Sr":87.62,
"Y":88.90585,"Zr":91.224,"Nb":92.90638,"Mo":95.96,"Tc":98,"Ru":101.07,"Rh":102.90550,"Pd":106.42,
"Ag":107.8682,"Cd":112.411,"In":114.818,"Sn":118.710,"Sb":121.760,"Te":127.60,"I":126.90447,
"Xe":131.293,"Cs":132.90545,"Ba":137.327,"La":138.90547,"Ce":140.116,"Pr":140.90765,"Nd":144.242,
"Pm":145,"Sm":150.36,"Eu":151.964,"Gd":157.25,"Tb":158.92535,"Dy":162.500,"Ho":164.93032,
"Er":167.259,"Tm":168.93421,"Yb":173.04,"Lu":174.9668,"Hf":178.49,"Ta":180.94788,"W":183.84,
"Re":186.207,"Os":190.23,"Ir":192.217,"Pt":195.084,"Au":196.966569,"Hg":200.59,"Tl":204.3833,
"Pb":207.2,"Bi":208.98040,"Po":209,"At":210,"Rn":222,"Fr":223,"Ra":226,"Ac":227,"Th":232.03806,
"Pa":231.03588,"U":238.02891,"Np":237,"Pu":244,"Am":243,"Cm":247,"Bk":247,"Cf":251,"Es":252,
"Fm":257,"Md":258,"No":259,"Lr":262,"Rf":267,"Db":270,"Sg":271,"Bh":270,"Hs":277,"Mt":276,"Ds":281,
"Rg":280,"Cn":285,"Fl":289,"Lv":293
}
