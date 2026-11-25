"""
Writers for the DFT input files of a single cluster structure.

The functions here are called by the workflow in :mod:`egqca.workflow` once per
cluster and cover relaxation, band-structure, and phonon inputs for:

1. Quantum ESPRESSO (pw.x / ph.x) -> ``vc-relax.in``, ``bands.in``, ``ph.in``
2. FHI-aims -> ``control.in`` + ``geometry.in``
3. VASP -> ``POSCAR``, ``POTCAR``, ``INCAR``, ``KPOINTS``

Notes
-----
- These functions only write inputs; they do not run any DFT code.
- Quantum ESPRESSO needs a pseudopotential for every element in ``pseudo_dir``.
- FHI-aims needs a valid species defaults directory in ``species_dir``.
- VASP needs one POTCAR per element under ``potcar_dir``.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import numpy as np
import spglib
from ase.dft.kpoints import parse_path_string
from ase.io.aims import write_aims, write_control
from ase.io.espresso import write_espresso_in, write_espresso_ph
from ase.io.vasp import write_vasp
from .constants import DEFAULT_KPTS, PSEUDO_DOJO_ECUTWFC



##############################################################
# ------------------------- Helpers ------------------------ #
# ############################################################
def find_pseudos_for_elements(pseudo_dir: str | Path, elements: list[str]) -> dict[str, Path]:
    """
    Return a mapping {element: path_to_upf} for elements found in pseudo_dir.
    Picks the first match (sorted) for each element.

    Matches files like:
      Mg*.upf, mg*.UPF, Mg_*.UPF, Mg.pbe*.UPF, etc.
    """
    pseudo_dir = Path(pseudo_dir).expanduser().resolve()
    if not pseudo_dir.is_dir():
        raise NotADirectoryError(f"pseudo_dir is not a directory: {pseudo_dir}")

    found: dict[str, Path] = {}

    for el in elements:
        # Match .UPF case-insensitively by checking suffix in code
        candidates = []
        for p in pseudo_dir.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if not name.lower().endswith(".upf"):
                continue
            # Common convention: filename starts with element symbol (Mg..., O..., Si...).
            # The symbol must not be followed by another letter, otherwise B would also
            # match Ba.upf, N Nb.upf and S Si.upf.
            if re.match(rf"{re.escape(el)}(?![a-z])", name, re.IGNORECASE):
                candidates.append(name)

        candidates.sort()
        if len(candidates) > 1:
            warnings.warn("Found more than one pseudopotential for "+el+"! Using the first one.", UserWarning)
            found[el] = candidates[0]
        elif candidates:
            found[el] = candidates[0]
        else:
            raise FileNotFoundError(
            f"Missing pseudopotential for: {el} \nLooked in: {Path(pseudo_dir).expanduser().resolve()}"
        )

    return found

def elements_from_atoms(atoms) -> list[str]:
    return sorted(set(atoms.get_chemical_symbols()))

def standard_orientation(atoms, symprec: float = 1e-3):
    """Rigidly rotate ``atoms`` so the symmetry axes of its lattice lie along x, y, z.

    A CIF stores no orientation, and ASE places a1 along x and a2 in the xy-plane. For
    e.g. a 60 degree fcc cell this puts [111] along z, and pw.x, which only tests
    rotations about the Cartesian axes, misses most symmetry operations. The rotation
    is spglib's standard orientation of the bare lattice, so the decoration does not
    matter. Cell lengths, angles and fractional positions are unchanged.

    Returns a rotated copy; ``atoms`` is not modified.
    """
    lattice = atoms.cell.array
    dataset = spglib.get_symmetry_dataset((lattice, [[0, 0, 0]], [1]), symprec=symprec)
    rotated = atoms.copy()
    if dataset is None:
        warnings.warn("spglib could not analyse the lattice; keeping the cell orientation.", UserWarning)
        return rotated
    rotated.set_cell(lattice @ np.asarray(dataset.std_rotation_matrix).T, scale_atoms=True)
    return rotated

def _default_ecutwfc_Ha(elements: list[str]) -> float:
    """Default plane-wave cutoff in Hartree, taken from the PseudoDojo recommendations.

    The maximum over the elements is used. If the cutoff of any element is unknown, the
    maximum of the whole table is used instead, which is safe but expensive - set
    ``qe_par["system"]["ecutwfc"]`` explicitly to avoid it.
    """
    unknown = [e for e in elements if e not in PSEUDO_DOJO_ECUTWFC]
    if unknown:
        fallback = max(PSEUDO_DOJO_ECUTWFC.values())
        warnings.warn(
            f"No PseudoDojo cutoff for {', '.join(unknown)}. Falling back to the table "
            f"maximum ecutwfc = {2*fallback} Ry, which is likely far higher than needed. "
            "Set qe_par['system']['ecutwfc'] explicitly."
        )
        return fallback
    return max(PSEUDO_DOJO_ECUTWFC[e] for e in elements)


def plot_bandpath(bp, outdir: Path):
    # plot band path
    file = outdir/"bandpath.pdf"
    if not file.is_file():
        ax_bz = bp.plot()
        ax_bz.figure.savefig(file, dpi=300)   


def _segment_npoints(cell, path: str, special_points: dict, npoints: int) -> list[int]:
    """Points per segment of ``path``, proportional to the segment lengths in ``cell``.

    One entry per (start, end) pair, in the order of ``parse_path_string(path)``.
    """
    rec = cell.reciprocal()
    segments = [(a, b) for sec in parse_path_string(path) for a, b in zip(sec[:-1], sec[1:])]
    lengths = [np.linalg.norm((np.asarray(special_points[b]) - np.asarray(special_points[a])) @ rec)
               for a, b in segments]
    total = sum(lengths)
    return [max(2, int(round(npoints * L / total))) for L in lengths]


def reference_bandpath(cell, path: str = "", kpoint_density: float = 20) -> dict:
    """Band path of a reference cell, to be used unchanged for every cluster.

    All clusters are the same supercell of the parent lattice, but relaxation lowers
    their symmetry, so ASE would give each its own Bravais lattice and default path.
    Using the special points of the reference (scaled coordinates) and the same
    number of points gives every cluster exactly the same k-points.

    Returns ``bands`` settings with ``path``, ``special_points``, ``npoints`` and
    ``segment_npoints`` (for FHI-aims), plus the original ``kpoint_density``.
    """
    bp = cell.bandpath(path or None, density=kpoint_density)
    special_points = {k: np.asarray(v, dtype=float) for k, v in bp.special_points.items()}
    return {
        "path": bp.path,
        "special_points": special_points,
        "npoints": len(bp.kpts),
        "segment_npoints": _segment_npoints(cell, bp.path, special_points, len(bp.kpts)),
        "kpoint_density": kpoint_density,
    }


def cluster_bandpath(atoms, bands: dict):
    """Band path of one cluster: from :func:`reference_bandpath` settings if given,
    else ASE's path of the cluster's own lattice (``path``, ``kpoint_density``)."""
    if "special_points" in bands:
        return atoms.cell.bandpath(bands["path"], npoints=bands["npoints"],
                                   special_points=bands["special_points"])
    return atoms.cell.bandpath(bands["path"] or None, density=bands["kpoint_density"])


##############################################################
# --------------------------- QE --------------------------- #
# ############################################################
def write_qe_vc_relax(atoms, outdir: str | Path, title: str, qe_par: dict, kpts = DEFAULT_KPTS):
    """Create the vc-relax file for a relaxation with Quantum Espresso.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure read from the CIF file.
    outdir : str or Path
        Path to the vc-relax folder.
    title : str
        Cluster name, used as title and prefix.
    qe_par : dict
        Parameters for the Quantum Espresso input file. The keys have to match the
        QE variable names.
    kpts : tuple
        Monkhorst-Pack k-point grid, e.g. ``(12, 12, 12)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    elements = elements_from_atoms(atoms)

    # Check we have pseudos for every element
    pseudos = find_pseudos_for_elements(qe_par["pseudo_dir"], elements)

    # /CONTROL
    qe_par["control"]["title"] = title
    qe_par["control"]["prefix"] = title
    qe_par["control"]["calculation"] = "vc-relax"

    # /SYSTEM
    # Default values for ecutwfc
    if not ("ecutwfc" in qe_par or "ecutwfc" in qe_par["system"]):
        ecutwfc_Ha = _default_ecutwfc_Ha(elements)
        qe_par["system"]["ecutwfc"] = 2*ecutwfc_Ha

    if not ("ecutrho" in qe_par or "ecutrho" in qe_par["system"]):
        qe_par["system"]["ecutrho"] = 8*qe_par.get("ecutwfc", qe_par["system"].get("ecutwfc"))

    # Orientation in which pw.x finds the full lattice symmetry
    atoms = standard_orientation(atoms)

    write_espresso_in(outdir / "vc-relax.in" , atoms, input_data=qe_par, pseudopotentials=pseudos, crystal_coordinates=True, kpts=kpts)


def write_qe_bands(atoms, outdir: str | Path, title: str, qe_par: dict, bands: dict):
    """
    Create pw.x input file for calculation='bands'
    using same structure and pseudopotentials as vc-relax.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    outdir : str or Path
        Path to the bands folder.
    title : str
        Cluster name, used as title and prefix.
    qe_par : dict
        Parameters for the Quantum Espresso input file. The keys have to match the
        QE variable names.
    bands : dict
        Band path settings with the keys ``path`` and ``kpoint_density``.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    elements = elements_from_atoms(atoms)

    # Check we have pseudos for every element
    pseudos = find_pseudos_for_elements(qe_par["pseudo_dir"], elements)

    # /CONTROL
    qe_par["control"]["title"] = title
    qe_par["control"]["prefix"] = title
    qe_par["control"]["calculation"] = "bands"

    # /SYSTEM
    # Default values for ecutwfc
    if not ("ecutwfc" in qe_par or "ecutwfc" in qe_par["system"]):
        ecutwfc_Ha = _default_ecutwfc_Ha(elements)
        qe_par["system"]["ecutwfc"] = 2*ecutwfc_Ha
    # same basis as the vc-relax
    if not ("ecutrho" in qe_par or "ecutrho" in qe_par["system"]):
        qe_par["system"]["ecutrho"] = 8*qe_par.get("ecutwfc", qe_par["system"].get("ecutwfc"))
    if not ("nbnd" in qe_par or "nbnd" in qe_par["system"]):
        qe_par["system"]["nbnd"] = 40    # default

    bp = cluster_bandpath(atoms, bands)
    plot_bandpath(bp, outdir)

    # WRITE QE INPUT
    write_espresso_in(
        outdir / "bands.in",
        atoms,
        input_data=qe_par,
        pseudopotentials=pseudos,
        kpts=bp,    # path in brillouin zone for bands calculation
        crystal_coordinates=True
    )


def write_qe_phonons(outdir: str | Path, title: str, qe_par: dict):
    """
    Create the ph.x input file for a phonon calculation with Quantum Espresso.

    Parameters
    ----------
    outdir : str or Path
        Path to the phonons folder.
    title : str
        Cluster name, used as prefix so that ph.x finds the pw.x output.
    qe_par : dict
        Parameters for the ph.x input file. The keys have to match the QE variable
        names.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qe_par = dict(qe_par, prefix=title)

    with open(outdir/"ph.in", "w") as fd:
        write_espresso_ph(fd, qe_par)


##############################################################
# ------------------------ FHI-aims ------------------------ #
# ############################################################
def write_aims_vc_relax(atoms, outdir: Path, aims_par:dict):
    """ 
        Create the geometry.in and control.in for a scf calculation with FHI-aims 
    """
    outdir.mkdir(parents=True, exist_ok=True)

    write_aims(outdir/"geometry.in", atoms=atoms)

    write_control(outdir/"control.in", atoms=atoms, parameters=aims_par)


def write_aims_bands(atoms, outdir: Path | str, aims_par:dict, bands: dict):
    """
        Create geometry.in and control.in for an FHI-aims band-structure run.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # write geometry.in
    write_aims(outdir/"geometry.in", atoms)

    # Remove relax directives and pre-existing band outputs (on a copy: aims_par is shared).
    aims_par = dict(aims_par)
    keys_to_remove = ["relax_geometry", "relax_unit_cell", "output band"]
    for key in keys_to_remove:
        aims_par.pop(key, None)

    bp = cluster_bandpath(atoms, bands)
    plot_bandpath(bp, outdir)

    segment_npoints = bands.get("segment_npoints") or _segment_npoints(
        atoms.cell, bp.path, bp.special_points, len(bp.kpts))
    aims_par = aims_band_output_lines(aims_par, bp, segment_npoints)

    write_control(outdir/"control.in", atoms=atoms, parameters=aims_par)


def aims_band_output_lines(aims_par, bp, segment_npoints, fmt="{: .6f}"):
    """
    Build FHI-aims band segments from an ASE band path.

    Sets ``aims_par["output"]`` to a list of strings like
      "band k1x k1y k1z   k2x k2y k2z   N  L1  L2"
    with one entry per segment of ``bp.path`` (discontinuities ``,`` and labels
    with more than one character are handled); ``segment_npoints`` gives N.
    (ASE uses 'G' for Gamma.)
    """
    sp = bp.special_points
    segments = [(a, b) for sec in parse_path_string(bp.path) for a, b in zip(sec[:-1], sec[1:])]

    value = []
    for (a, b), n in zip(segments, segment_npoints):
        ka, kb = sp[a], sp[b]
        value.append("band "
            + " ".join(fmt.format(x) for x in ka) + "   "
            + " ".join(fmt.format(x) for x in kb) + "   "
            + f"{n:d} {a} {b}"
            )

    aims_par["output"] = value

    return aims_par



def write_aims_ph(atoms, outdir: Path | str, qpath: str = "", qpoint_density: float = 20,
                  aims_ph_param: dict | None = None, phonon_supercell: tuple[int, int, int] = (1,1,1),
                  calculate_friction: bool = False):
    """Create geometry.in and control.in for an FHI-aims DFPT phonon run.

    ``geometry.in`` holds ``atoms.repeat(phonon_supercell)``. With
    ``calculate_friction``, a ``calculate_friction .true.`` line follows every atom.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Default q-path from ASE Brillouin-zone path if no custom path is given.
    # if qpath:
    #     qp = atoms.cell.bandpath(qpath, density=qpoint_density)
    # else:
    #     qp = atoms.cell.bandpath(density=qpoint_density)

    # rec_cell = atoms.cell.reciprocal()
    # phonon_lines = ["DFPT phonon"]

    # for section in parse_path_string(qp.path):
    #     for start_lbl, end_lbl in zip(section[:-1], section[1:]):
    #         q_start = np.array(qp.special_points[start_lbl], dtype=float)
    #         q_end = np.array(qp.special_points[end_lbl], dtype=float)

    #         dq = (q_end - q_start) @ rec_cell
    #         n_pts = max(2, int(round(np.linalg.norm(dq) * qpoint_density)) + 1)

    #         phonon_lines.append(
    #             "output phonon band "
    #             f"{q_start[0]:.8f} {q_start[1]:.8f} {q_start[2]:.8f} "
    #             f"{q_end[0]:.8f} {q_end[1]:.8f} {q_end[2]:.8f} {n_pts}"
    #         )

    geometry = outdir/"geometry.in"
    write_aims(geometry, atoms=atoms.repeat(phonon_supercell))
    if calculate_friction:
        lines = geometry.read_text().splitlines(keepends=True)
        geometry.write_text("".join(
            line + "calculate_friction .true.\n" if line.startswith("atom") else line
            for line in lines
        ))

    write_control(outdir/"control.in", atoms=atoms, parameters=aims_ph_param)


##############################################################
# -------------------------- VASP -------------------------- #
# ############################################################
def _vasp_sorted_elements(atoms):
    """Return atoms sorted by element symbol and the corresponding ordered unique element list."""
    sort_idx = np.argsort(atoms.get_chemical_symbols())
    atoms_sorted = atoms[sort_idx]
    seen: set[str] = set()
    ordered_elements: list[str] = []
    for sym in atoms_sorted.get_chemical_symbols():
        if sym not in seen:
            ordered_elements.append(sym)
            seen.add(sym)
    return atoms_sorted, ordered_elements


def _write_vasp_potcar(outdir: Path, potcar_dir: Path, ordered_elements: list[str], potentials: dict[str, str]):
    """Concatenate per-element POTCAR files into outdir/POTCAR."""
    potcar_parts = []
    for el in ordered_elements:
        if el in potentials:
            potcar_file = potcar_dir / potentials[el] / "POTCAR"
        else:
            potcar_file = potcar_dir / el / "POTCAR"
        if not potcar_file.is_file():
            raise FileNotFoundError(
                f"Missing POTCAR for element {el}. Expected: {potcar_file}\n"+
                  "You can specify a custom filename for each element in vasp_par['potentials'], e.g. 'potentials': {{'Sr': 'Sr_sv'}}"
            )
        potcar_parts.append(potcar_file.read_text())
    (outdir / "POTCAR").write_text("".join(potcar_parts))


def _write_vasp_incar(outdir: Path, vasp_par: dict, overrides: dict | None = None, skip_keys: set | None = None):
    """Write INCAR from vasp_par dict, applying overrides and skipping skip_keys."""
    skip = (skip_keys or set()) | {"potcar_dir", "potentials"}
    par = dict(vasp_par)
    if overrides:
        par.update(overrides)
    lines = []
    for key, val in par.items():
        if key in skip:
            continue
        lines.append(f"{key} = {'T' if val is True else 'F' if val is False else val}")
    (outdir / "INCAR").write_text("\n".join(lines) + "\n")


def write_vasp_vc_relax(atoms, outdir: str | Path, vasp_par: dict, kpts=DEFAULT_KPTS):
    """Write VASP vc-relax input files: POSCAR, INCAR, KPOINTS, POTCAR.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure read from the CIF file.
    outdir : str or Path
        Path to the vc-relax folder.
    vasp_par : dict
        INCAR parameters; must contain ``potcar_dir``.
    kpts : tuple
        Monkhorst-Pack k-point grid, e.g. ``(12, 12, 12)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atoms_sorted, ordered_elements = _vasp_sorted_elements(atoms)

    write_vasp(outdir / "POSCAR", atoms_sorted, vasp5=True, direct=True)

    _write_vasp_potcar(outdir, Path(vasp_par["potcar_dir"]), ordered_elements, vasp_par.get("potentials", {}))

    kx, ky, kz = kpts
    (outdir / "KPOINTS").write_text(
        f"Automatic k-mesh\n0\nMonkhorst-Pack\n{kx} {ky} {kz}\n0 0 0\n"
    )

    _write_vasp_incar(outdir, vasp_par)


def write_vasp_bands(atoms, outdir: str | Path, vasp_par: dict, bands: dict):
    """Write VASP band-structure input files: POSCAR, INCAR, KPOINTS, POTCAR.

    Requires a completed vc-relax run (CHGCAR) in the sibling vc-relax directory.
    Set ICHARG=11 in INCAR so VASP reads the pre-converged charge density.

    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure.
    outdir : str or Path
        Path to the bands folder.
    vasp_par : dict
        Base INCAR parameters; must contain ``potcar_dir``.
    bands : dict
        Band path settings with the keys ``path`` (str) and ``kpoint_density`` (int).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    atoms_sorted, ordered_elements = _vasp_sorted_elements(atoms)

    write_vasp(outdir / "POSCAR", atoms_sorted, vasp5=True, direct=True)

    _write_vasp_potcar(outdir, Path(vasp_par["potcar_dir"]), ordered_elements, vasp_par.get("potentials", {}))

    # Band path; KPOINTS line mode: the same number of points on every segment
    npoints = bands.get("kpoint_density", 20)
    bp = cluster_bandpath(atoms, bands)

    plot_bandpath(bp, outdir)

    # KPOINTS in line mode
    kpts_lines = ["Band structure k-path", str(npoints), "Line-mode", "Reciprocal", ""]
    for section in parse_path_string(bp.path):
        for start, end in zip(section[:-1], section[1:]):
            k_s = bp.special_points[start]
            k_e = bp.special_points[end]
            kpts_lines.append(f" {k_s[0]:.8f} {k_s[1]:.8f} {k_s[2]:.8f}  ! {start}")
            kpts_lines.append(f" {k_e[0]:.8f} {k_e[1]:.8f} {k_e[2]:.8f}  ! {end}")
            kpts_lines.append("")
    (outdir / "KPOINTS").write_text("\n".join(kpts_lines))

    # INCAR: non-SCF bands run
    _write_vasp_incar(
        outdir, vasp_par,
        overrides={"NSW": 0, "IBRION": -1, "ICHARG": 11, "LORBIT": 11, "ISIF": 0, "LCHARG": False},
        skip_keys={"EDIFFG"},
    )
