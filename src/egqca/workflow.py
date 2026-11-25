import copy
import shutil
import subprocess, re
import numpy as np
from pathlib import Path
from .constants import ATOMIC_MASSES, DEFAULT_KPTS, Ry_to_mev, QE_PHONON_DEFAULTS, QE_DEFAULTS, AIMS_DEFAULTS, VASP_DEFAULTS
from .create_dft_inputs import (
    _default_ecutwfc_Ha,
    reference_bandpath,
    write_qe_vc_relax,
    write_aims_vc_relax,
    write_qe_bands,
    write_qe_phonons,
    write_aims_bands,
    write_aims_ph,
    write_vasp_vc_relax,
    write_vasp_bands,
)
from ase.io import read
from ase.io.espresso import read_espresso_out
from ase.io.aims import read_aims_output
import warnings
from ._banner import entry_point

from collections import Counter
from math import gcd
from ._ml_utils import _atomic_write, _original_cluster_cif, _extract_snumber_from_name

CLUSTERS_TEMPLATE_NAME = "clusters_template.dat"
# clusters.dat of the DFT run folder in trees set up before clusters_template.dat existed
LEGACY_DFT_CLUSTERS = Path("DFT") / "clusters.dat"

try:
    from .ml import geometry_relaxation
except ImportError:
    geometry_relaxation = None

def _dft_codes(bound):
    codes = [name for name, key in (("Quantum ESPRESSO", "qe_par"), ("FHI-aims", "aims_par"), ("VASP", "vasp_par"))
             if bound.arguments[key]]
    return ", ".join(codes) if codes else "none"


@entry_point("Cluster and DFT setup", lambda b: [
    ("CIF file", b.arguments["path_to_cif"]),
    ("Supercell size", b.arguments["supercell_size"]),
    ("Output folder", b.arguments["out_dir"]),
    ("Substitution mode", b.arguments["substitution_mode"]),
    ("DFT inputs for", _dft_codes(b)),
    ("k-points", b.arguments["kpts"]),
    ("CIF files only", b.arguments["cif_only"]),
])
def set_up_clusters_and_dft(supercell_program:str, path_to_cif:str, supercell_size:str,
                            out_dir:str,
                            qe_par: dict | None = None,
                            aims_par: dict | None = None,
                            vasp_par: dict | None = None,
                            kpts = None,
                            create_dft_folders: bool = False,
                            substitution_mode: str = "auto",
                            cif_only: bool = False):
    """ 
    Create all non-equivalent clusters of size supercell_size and set up the corresponding folder 
    structure containing templates for the DFT input files.

    The cluster setup (compound, sites, permutable sites n, one row per cluster) is
    written to ``<out_dir>/clusters_template.dat``, which every run folder (DFT, ML,
    EDDP) reads. The DFT run folder ``<out_dir>/DFT`` with a copy of it as
    ``clusters.dat`` is only created when ``create_dft_folders`` is set or one of
    ``qe_par``, ``aims_par``, ``vasp_par`` is given; :func:`update_DFT_in` creates it
    later otherwise.

    Parameters
    ----------
    supercell_program : str
        Path to the supercell executable.
    path_to_cif : str
        Path to the CIF file. See the documentation for the expected format.
    supercell_size : str
        Size of the supercell in the format ``1x1x1``, with the letter x as separator.
    out_dir : str
        Path to the output directory. Must not contain any subdirectory yet.
    qe_par : dict or None
        Quantum ESPRESSO input parameters. Input files are only written if it
        contains ``pseudo_dir``.
    aims_par : dict or None
        FHI-aims input parameters. Input files are only written if it contains
        ``species_dir``.
    vasp_par : dict or None
        VASP input parameters. Input files are only written if it contains
        ``potcar_dir``.
    kpts : tuple or None
        k-point grid for the DFT calculations, e.g. ``(12, 12, 12)``.
    create_dft_folders : bool
        Create the ``vc-relax``, ``bands``, and ``phonons`` subfolder of every
        cluster even when no solver parameters are given. Set it to False for an
        ML-only run, which reads none of them. They are written regardless when
        ``qe_par``, ``aims_par`` or ``vasp_par`` asks for input files, since the
        writers create their own.
    substitution_mode : str
        How the substituting atoms are distributed when more than one site is
        substituted: ``"auto"``, ``"independent"``, or ``"coupled"``.
    cif_only : bool
        Write all cluster CIF files directly into ``out_dir`` and skip the folder
        organisation, the DFT setup, and the clusters template.
    """

    qe_par = {} if qe_par is None else dict(qe_par)
    aims_par = {} if aims_par is None else dict(aims_par)
    vasp_par = {} if vasp_par is None else dict(vasp_par)

    # Check if out_dir exists and is empty, if not raise error
    path = Path(out_dir)
    if path.exists():
        if any(p.is_dir() for p in path.iterdir()):
            raise OSError(f"{path} is not empty. Please choose an empty directory or remove the existing files.")

    # Check if cif-file exists
    if not Path(path_to_cif).is_file():
        raise FileNotFoundError(f"{path_to_cif} is not a valid path to a cif-file.")

    # get elements
    elements, permutable_elements_label, sites, permutable_sites, compound = get_elements(path_to_cif, supercell_size, substitution_mode)

    # create supercells
    run_supercell(supercell_program, path_to_cif, supercell_size, permutable_sites,
                  permutable_elements_label, out_dir, cif_only=cif_only,
                  create_dft_folders=create_dft_folders)

    if cif_only:
        print("CIF files created. Skipping DFT input and clusters.dat setup.\n")
        return

    # Create input for QE, AIMS, and VASP
    if any([qe_par, aims_par, vasp_par]) or create_dft_folders:
        create_DFT_in(
            out_dir,
            qe_par=qe_par,
            aims_par=aims_par,
            vasp_par=vasp_par,
            kpts=kpts,
        )

    Nat = _nat_from_cif(path_to_cif, supercell_size)


    # the clusters template (energies missing) that every run folder starts from;
    # the DFT run folder gets its copy only when DFT inputs or folders are set up
    set_up_clusters_dat(out_dir, permutable_sites, sites, Nat, compound)
    if create_dft_folders or any([qe_par, aims_par, vasp_par]):
        _copy_template_to_run_folder(out_dir, Path(out_dir) / "DFT")
    print("Set up finished!\n")

        


def _integer_distributions(capacities: list[int], total: int):
    """Yield all tuples (k0,...,kN) with 0<=ki<=capacities[i] and sum==total."""
    if not capacities:
        if total == 0:
            yield ()
        return
    cap = capacities[0]
    for k in range(min(cap, total) + 1):
        for rest in _integer_distributions(capacities[1:], total - k):
            yield (k,) + rest


def _supercell_permutation_args(atoms_label, original_count: int, substitute_count: int) -> list[str]:
    """Build supercell -p arguments for one or more simultaneously substituted sites."""
    if isinstance(atoms_label, dict):
        original_labels = atoms_label["original"]
        substitute_labels = atoms_label["substitution"]
        args = []
        for label in original_labels:
            args.extend(["-p", f"{label}:p={original_count}"])
        for label in substitute_labels:
            args.extend(["-p", f"{label}:p={substitute_count}"])
        return args

    return [
        "-p", f"{atoms_label[0]}:p={original_count}",
        "-p", f"{atoms_label[1]}:p={substitute_count}",
    ]


def _supercell_runs(permutable_sites: int, atoms_label):
    output_multiplier = 1
    # substitute several atoms
    if isinstance(atoms_label, dict):
        sites_per_label = atoms_label["sites_per_label"]
        mode = atoms_label.get("mode", "coupled")
        use_distribution = mode == "independent" or len(set(sites_per_label)) > 1
        if use_distribution:
            # Independent substitution or mixed multiplicities: enumerate all distributions per site
            original_labels = atoms_label["original"]
            substitute_labels = atoms_label["substitution"]
            for k_total in range(sum(sites_per_label) + 1):
                dist_idx = 0
                for dist in _integer_distributions(sites_per_label, k_total):
                    args = []
                    for orig, sub, cap, ki in zip(original_labels, substitute_labels, sites_per_label, dist):
                        args.extend(["-p", f"{orig}:p={cap - ki}", "-p", f"{sub}:p={ki}"])
                    yield k_total, f"S{k_total}_d{dist_idx}", args
                    dist_idx += 1
            return
        # Coupled substitution (AB)→(CC): each step replaces one unit from every original site
        output_multiplier = len(atoms_label["original"])
        permutable_sites = sites_per_label[0]

    # substitute one atom
    for k in range(permutable_sites+1):
        k_rev = permutable_sites - k
        substituted_count = output_multiplier * k
        yield substituted_count, f"S{substituted_count}", _supercell_permutation_args(atoms_label, k_rev, k)


_SUPERCELL_NAME = re.compile(r"^(S\d+)(?:_d(\d+))?_i(\d+)_w(\d+)$")


def _continuous_cluster_names(stems) -> dict[str, str]:
    """Map the supercell output names of one ``S*`` to names without distribution index.

    In distribution mode the supercell program runs once per distribution with the
    prefix ``S<k>_d<m>`` and numbers its clusters from 0 in every run. The clusters
    are renumbered continuously over all distributions, ordered numerically by
    (distribution, index), and the index is zero-padded to one width per ``S*``:

        S3_d0_i0_w4, S3_d0_i1_w8, S3_d1_i0_w2  ->  S3_i0_w4, S3_i1_w8, S3_i2_w2

    Returns ``{old stem: new stem}`` for the names that change; empty if no name
    carries a distribution index.
    """
    parsed = []
    for stem in stems:
        match = _SUPERCELL_NAME.match(stem)
        if match is None:
            raise ValueError(f"Unexpected supercell output name: {stem!r}")
        parsed.append((stem, match))
    if not any(match.group(2) is not None for _, match in parsed):
        return {}

    prefixes = {match.group(1) for _, match in parsed}
    if len(prefixes) != 1:
        raise ValueError(f"Cluster names of several S* folders mixed: {sorted(prefixes)}")
    prefix = prefixes.pop()

    parsed.sort(key=lambda item: (int(item[1].group(2) or 0), int(item[1].group(3))))
    width = len(str(len(parsed) - 1))
    mapping = {}
    for index, (stem, match) in enumerate(parsed):
        new = f"{prefix}_i{index:0{width}d}_w{match.group(4)}"
        if new != stem:
            mapping[stem] = new
    return mapping


def _tail(path: str | Path, n: int = 15) -> str:
    """Last ``n`` lines of a text file, for error messages."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return "(log file not readable)"
    return "\n".join(lines[-n:])


def run_supercell(path_to_supercell:str, path_to_cif:str, supercell_size:str, permutable_sites:int, atoms_label, out_dir:str, cif_only:bool = False, create_dft_folders:bool = True):
    """Run the supercell program to create all non-equivalent clusters of a given size and set up the folder structure.

    Parameters
    ----------
    cif_only : bool
        If True, write all CIF files directly into ``out_dir`` and skip the folder
        organisation and the creation of the DFT subdirectories.
    create_dft_folders : bool
        Create the ``vc-relax``, ``bands`` and ``phonons`` subfolder of every
        cluster folder. Set it to False for an ML-only run, which reads none of
        them; :func:`create_DFT_in` makes its own when input files are written.
    """
    if shutil.which(str(path_to_supercell)) is None:
        raise FileNotFoundError(
            f"supercell_program {path_to_supercell!r} is not an executable file. "
            "Download it from https://orex.github.io/supercell/ and pass its path."
        )
    out_dir_path = Path(out_dir)
    started_logs = set()
    uses_distributions = False      # only distribution mode names carry _d<m>
    for substituted_count, output_prefix, permutation_args in _supercell_runs(permutable_sites, atoms_label):
        uses_distributions |= "_d" in output_prefix
        if cif_only:
            cif_output_dir = out_dir_path
            log_path = out_dir_path / "logs"
            log_path.mkdir(parents=True, exist_ok=True)
            log_path = log_path/"log.out"
            log_mode = "a"
        else:
            out_dir_temp = out_dir_path / f"S{substituted_count}"
            out_dir_temp.mkdir(parents=True, exist_ok=True)
            cif_output_dir = out_dir_temp
            log_path = out_dir_temp / "log.out"
            # one log per S*: the first run starts it, further distributions append
            log_mode = "a" if log_path in started_logs else "w"
            started_logs.add(log_path)

        specs = [(a.split(":p=")[0], int(a.split(":p=")[1])) for a in permutation_args if ":" in a]
        sites_str = "  ".join(f"{lbl}:{cnt}" for lbl, cnt in specs)
        print(f"Running {output_prefix}  [{sites_str}]")

        with open(log_path, log_mode) as logfile:
            result = subprocess.run(
                [
                    path_to_supercell,
                    "-i", path_to_cif,
                    "-s", supercell_size,
                    *permutation_args,
                    "-m",
                    "-o", str(cif_output_dir / output_prefix),
                    "-v", "2"
                ],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"The supercell program failed for {output_prefix} (exit code {result.returncode}).\n"
                f"Last lines of {log_path}:\n{_tail(log_path)}\n"
                f"Delete {out_dir_path} before running again."
            )
        if not any(cif_output_dir.glob(f"{output_prefix}_i*.cif")):
            raise RuntimeError(
                f"The supercell program wrote no CIF files for {output_prefix}.\n"
                f"Last lines of {log_path}:\n{_tail(log_path)}\n"
                f"Delete {out_dir_path} before running again."
            )

    print("Supercell program done!\n")

    if cif_only:
        if not uses_distributions:
            return
        # rename in place to names without distribution index
        by_S = {}
        for cif in out_dir_path.glob("S*.cif"):
            match = _SUPERCELL_NAME.match(cif.stem)
            if match is not None:
                by_S.setdefault(match.group(1), []).append(cif.stem)
        for stems in by_S.values():
            for old, new in _continuous_cluster_names(stems).items():
                (out_dir_path / f"{old}.cif").rename(out_dir_path / f"{new}.cif")
        return

    # all folders S*
    folders = [p for p in Path(out_dir).iterdir() if p.is_dir() and p.name.startswith("S")]

    # move S*.cif into folder S*, dropping the distribution index from the name
    all_cif_files = []
    for S_dir in folders:
        cifs = list(S_dir.glob("S*.cif"))
        names = _continuous_cluster_names([cif.stem for cif in cifs]) if uses_distributions else {}
        all_cif_files += [(S_dir, cif, names.get(cif.stem, cif.stem)) for cif in cifs]
    n_total = len(all_cif_files)
    for i, (S_dir, cif_file, folder_name) in enumerate(all_cif_files, 1):
        w = len(str(n_total))
        print(f"Organising clusters: {i:{w}d}/{n_total} ({folder_name})", end="\r", flush=True)
        target_dir = S_dir / folder_name
        target_dir.mkdir(exist_ok=True)
        cif_file.rename(target_dir / f"{folder_name}.cif")
        if create_dft_folders:
            for dft_dir in ["vc-relax", "bands", "phonons"]:
                sub_dir = target_dir /dft_dir
                sub_dir.mkdir()
    print("Organising clusters done!"+" " * 30 + "\n")


def get_directories(base_dir: str | Path):
    """Return all ``S*/S*_i*_w*`` cluster folders of a cluster tree.

    The list is sorted by the number of substituted atoms and, within one ``S*``, by
    folder name. Every workflow function that loops over clusters uses this order.

    Parameters
    ----------
    base_dir : str or Path
        Folder containing the ``S*`` subdirectories.

    Returns
    -------
    list of pathlib.Path
        One entry per cluster folder.
    """
    base_dir = Path(base_dir)
    directories = []
    # ------------ Loop over S* ------------ #
    for S_dir in sorted((p for p in base_dir.glob("S*") if p.is_dir() and re.search(r"S(\d+)", p.name)), key=extract_Snumber):
        # ------- Loop over S*_i*_w* ------- #
        for sub_dir in sorted(S_dir.glob("S*_i*_w*")):
            if not sub_dir.is_dir():
                continue
            directories.append(sub_dir)
    return directories


def _write_ml_clusters_header(
    path: str | Path,
    *,
    permutable_sites: int,
    sites: int,
    compound: str,
    ML_model: str | None,
) -> None:
    with open(path, "w") as f:
        f.write(f"# EGQCA input for compound={compound}\n")
        if ML_model is not None:
            f.write(f"# uMLIP model for relaxation: {ML_model}\n")
        f.write("# total number of sites="+str(sites)+"\n")
        f.write("# permutable sites n="+str(permutable_sites)+"\n")
        f.write("#ID\t\tn_j\tNat\tx\tg_j\tE_j (Ry)\n")


def _write_ml_volume_header(path: str | Path, *, ML_model: str | None) -> None:
    with open(path, "w") as f:
        f.write("# Lattice parameters required for including the PV term in EGQCA\n")
        if ML_model is not None:
            f.write(f"# uMLIP model for relaxation: {ML_model}\n")
        f.write("#ID\t\ta(ang)\tb(ang)\tc(ang)\talpha(deg)\tbeta(degree)\tgamma(degree)\n")


def set_up_clusters_dat(clusters_folder: str | Path, permutable_sites: int, sites: int, Nat: int, compound: str, filename: str = CLUSTERS_TEMPLATE_NAME) -> Path:
    """Write the clusters template ``<clusters_folder>/clusters_template.dat``.

    One row per cluster folder ``S*/S*_i*_w*``: the ID, the number of substituted
    atoms n_j, the total number of atoms Nat, the composition x = n_j / n, the
    degeneracy g_j and the energy ``missing``. The header holds the compound, the
    total number of sites and the number of permutable sites n. Every run folder
    (DFT, ML, EDDP) takes its cluster setup from this file; the DFT run folder
    starts from a copy of it.

    Parameters
    ----------
    clusters_folder : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories; the template
        is written into it.
    permutable_sites : int
        Number of permutable sites, written to the header and used for x = n_j / n.
    sites : int
        Total number of sites, written to the header.
    Nat : int
        Total number of atoms of a cluster.
    compound : str
        Compound label written to the header, e.g. ``A$_{1-x}$B$_x$C``.
    filename : str
        Name of the template file.

    Returns
    -------
    pathlib.Path
        Path to the written template.
    """

    clusters_folder = Path(clusters_folder)
    clusters_info = []

    # Loop over all subfolders S*_i*_w* and extract the information for the template
    for sub_dir in get_directories(clusters_folder):
        cluster_id = sub_dir.name
        match = re.search(r"S(\d+)_", str(sub_dir.name))
        if match:
            n_j = int(match.group(1))
            x = n_j / permutable_sites
        else:
            n_j = None
            x = None
        match = re.search(r"_w(\d+)", str(sub_dir.name))
        g_j = int(match.group(1)) if match else None

        clusters_info.append((cluster_id, n_j, Nat, x, g_j))

    clusters_file = clusters_folder / filename
    _write_ml_clusters_header(
        clusters_file,
        permutable_sites=permutable_sites,
        sites=sites,
        compound=compound,
        ML_model=None,
    )
    with open(clusters_file, "a") as f:
        for cluster_info in clusters_info:
            f.write("\t".join(map(str, cluster_info)) + "\tmissing\n")
    return clusters_file


def _format_variable_multiplicity(multiplicity: int, variable: str) -> str:
    if multiplicity == 1:
        return variable
    return f"{multiplicity}{variable}"


def _formula_count(symbol: str, count, latex: bool) -> str:
    """``symbol`` with its count as subscript; a count of 1 is not written."""
    if count == 1 or count == "1":
        return symbol
    return f"{symbol}$_{{{count}}}$" if latex else f"{symbol}{count}"


def _recognized_structural_formula(atom_sites: list[dict], permutable_pairs: list[list[dict]], latex: bool = False) -> str:
    """Compound label with x in [0, 1], as in the clusters files.

    One permutable site of multiplicity m: ``(A$_{1-x}$B$_{x}$)$_{m}$`` (no outer
    subscript for m = 1), as a reduced formula: m and the counts of the other
    elements are divided by their common factor (Ga4N4 -> ``Ga$_{1-x}$In$_{x}$N``).
    Several permutable sites: ``(AB)$_{1-x}$C$_{2x}$``, not reduced. The other
    sites follow, merged per element in the order of the CIF, e.g. two Se sites of
    multiplicity 2 and 1 give ``Se$_{3}$``.
    """
    original_sites = [pair[0] for pair in permutable_pairs]
    substitution_sites = [pair[1] for pair in permutable_pairs]
    permutable_labels = {site["label"] for pair in permutable_pairs for site in pair}

    substitution_by_symbol = {}
    for site in substitution_sites:
        substitution_by_symbol[site["symbol"]] = substitution_by_symbol.get(site["symbol"], 0) + 1
    substitution_parts = [
        _formula_count(symbol, _format_variable_multiplicity(multiplicity, "x"), latex)
        for symbol, multiplicity in substitution_by_symbol.items()
    ]

    # the other sites, merged per element
    other_counts = {}
    for site in atom_sites:
        if site["label"] in permutable_labels or np.isclose(site["occupancy"], 0.0):
            continue
        other_counts[site["symbol"]] = other_counts.get(site["symbol"], 0) + site["multi"]

    formula_parts = []
    if len(permutable_pairs) > 1:
        original_group = "".join(site["symbol"] for site in original_sites)
        formula_parts.append(f"({original_group})$_{{1-x}}$" if latex else f"({original_group})1-x")
        formula_parts.extend(substitution_parts)
    else:
        # reduced formula: all counts divided by their common factor, e.g. Ga4N4 -> GaN
        m = original_sites[0]["multi"]
        common = gcd(m, *other_counts.values())
        m //= common
        other_counts = {symbol: count // common for symbol, count in other_counts.items()}
        group = _formula_count(original_sites[0]["symbol"], "1-x", latex) + "".join(substitution_parts)
        formula_parts.append(group if m == 1 else _formula_count(f"({group})", m, latex))

    formula_parts.extend(_formula_count(symbol, count, latex) for symbol, count in other_counts.items())

    return "".join(formula_parts)


def _nat_from_cif(path_to_cif: str | Path, supercell_size: str) -> int:
    """Return total number of physical atom sites in the CIF scaled by the supercell factor."""
    capture = False
    idx = 0
    multi_col = None
    coord_cols: list[int] = []
    seen_coords: set = set()
    total = 0
    with Path(path_to_cif).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "_atom_site_" in line:
                capture = True
                if "_atom_site_symmetry_multiplicity" in line:
                    multi_col = idx
                elif "_atom_site_fract_" in line:
                    coord_cols.append(idx)
                idx += 1
                continue
            if capture:
                if line.startswith("loop_"):
                    break
                parts = line.split()
                if not parts:
                    continue
                coords = tuple(parts[i] for i in coord_cols) if coord_cols else (parts[0],)
                if coords not in seen_coords:
                    seen_coords.add(coords)
                    total += int(float(parts[multi_col])) if multi_col is not None else 1
    sc_factor = int(np.prod([int(x) for x in supercell_size.split("x")]))
    return sc_factor * total


def get_elements(path_to_cif:str | Path, supercell_size:str, substitution_mode: str = "auto"):
    """ Get all elements from a cif-file """
    cif_file = Path(path_to_cif)
    elements = []
    elements_label = []
    permutable_elements = []
    permutable_elements_label = []
    permutable_pairs = []
    multiplicity = []
    atom_sites = []
    capture = False
    periodicTable = ATOMIC_MASSES.keys()

    idx_atom_site = 0
    relevant_idx = dict(coord= [])
    with cif_file.open() as f:
        for line in f:
            line = line.strip()
            # Look for atom site lines (skip comments and headers)
            if not line or line.startswith("#"):
                continue

            # Start capturing after _atom_site_occupancy
            if "_atom_site_" in line:
                capture = True
                if "_atom_site_label" in line:
                    relevant_idx["label"] = idx_atom_site
                elif "_atom_site_type_symbol" in line:
                    relevant_idx["type_symbol"] = idx_atom_site
                elif "_atom_site_symmetry_multiplicity" in line:
                    relevant_idx["multi"] = idx_atom_site
                elif "_atom_site_fract_" in line:
                    relevant_idx["coord"].append(idx_atom_site)
                elif "_atom_site_occupancy" in line:
                    relevant_idx["occupancy"] = idx_atom_site
                idx_atom_site +=1
                continue
            if capture:
                # Stop where the atom_site loop ends: another loop, data item or data block
                if line.startswith(("loop_", "_", "data_")):
                    break
                # Split line and get first column (element symbol)
                parts = line.split()
                if parts:
                    if "type_symbol" not in relevant_idx:
                        raise ValueError(
                            f"The CIF file {cif_file} has no '_atom_site_type_symbol' column. The "
                            "element of every site is read from it (the site labels, e.g. 'O1a', "
                            "do not name the element reliably). Export the CIF with this column, "
                            "e.g. from VESTA."
                        )
                    symbol_label = parts[relevant_idx["label"]]  # e.g., 'C1', 'O2', 'Mg'
                    # element from the type symbol, without oxidation state: 'O2-' -> 'O', 'Fe3+' -> 'Fe'
                    type_symbol = re.match(r"[A-Z][a-z]?", parts[relevant_idx["type_symbol"]])
                    symbol = type_symbol.group(0) if type_symbol else ""
                    if symbol not in periodicTable:
                        raise ValueError(
                            f"Unknown element {parts[relevant_idx['type_symbol']]!r} in the "
                            f"'_atom_site_type_symbol' column of site {symbol_label!r} in {cif_file}."
                        )
                    coords = tuple(parts[i] for i in relevant_idx["coord"])
                    elements.append(symbol)
                    elements_label.append(symbol_label)
                    if "multi" in relevant_idx:
                        multi = int(float(parts[relevant_idx["multi"]]))
                    else: 
                        multi = 1
                    occupancy = float(parts[relevant_idx["occupancy"]]) if "occupancy" in relevant_idx else 1.0
                    atom_sites.append(
                        {
                            "symbol": symbol,
                            "label": symbol_label,
                            "multi": multi,
                            "coords": coords,
                            "occupancy": occupancy,
                        }
                    )
                    multiplicity.append(multi)

    if "multi" not in relevant_idx:
        raise ValueError(
            "The CIF file is missing the '_atom_site_symmetry_multiplicity' column. "
            "The supercell program needs the multiplicity of each site to determine the clusters correctly."
            "You can obtain the multiplicity via VESTA:"
            "Utilities -> Standardization of Crystal Data -> see Output in bottom panel."
        )

    sites_by_coord = {}
    for site in atom_sites:
        sites_by_coord.setdefault(site["coords"], []).append(site)

    num_sites_uc = sum(sites[0]["multi"] for sites in sites_by_coord.values())
    permutable_sites_uc = None
    for sites in sites_by_coord.values():
        unique_symbols = {site["symbol"] for site in sites}
        if len(unique_symbols) < 2:
            continue

        occupied_sites = [site for site in sites if site["occupancy"] > 0.0]
        zero_occupancy_sites = [site for site in sites if np.isclose(site["occupancy"], 0.0)]
        if occupied_sites and zero_occupancy_sites:
            permutable_pair = [occupied_sites[0], zero_occupancy_sites[0]]
        else:
            permutable_pair = sites[:2]

        permutable_pairs.append(permutable_pair)

    if not permutable_pairs:
        raise ValueError(
            "Could not identify the permutable site in the CIF file. "
            "Add two atom-site rows with different elements at the same fractional "
            "coordinates, using occupancy 1 for the original element and occupancy 0 "
            "for the substituting element."
        )

    multiplicities_per_pair = [pair[0]["multi"] for pair in permutable_pairs]
    _all_multiplicities_equal = len(set(multiplicities_per_pair)) == 1

    if len(permutable_pairs) == 1:
        permutable_elements = [site["symbol"] for site in permutable_pairs[0]]
        permutable_elements_label = [site["label"] for site in permutable_pairs[0]]
    else:
        original_sites = [pair[0] for pair in permutable_pairs]
        substitution_sites = [pair[1] for pair in permutable_pairs]
        substitution_symbols = {site["symbol"] for site in substitution_sites}
        if len(substitution_symbols) != 1:
            raise ValueError(
                "Only one concentration variable is currently supported. "
                "Use original species coupled to one substituting species."
            )

        permutable_elements = [site["symbol"] for site in original_sites + substitution_sites]
        original_symbols = {site["symbol"] for site in original_sites}
        auto_mode = "independent" if len(original_symbols) == 1 else "coupled"
        mode = auto_mode if substitution_mode == "auto" else substitution_mode
        permutable_elements_label = {
            "mode": mode,
            "original": [site["label"] for site in original_sites],
            "substitution": [site["label"] for site in substitution_sites],
            "sites_per_label": None,
        }

    # -------- Permutable sites & total sites ------------- #
    num_supercells = [int(x) for x in supercell_size.split("x")]
    sc_factor = int(np.prod(num_supercells))
    sites = sc_factor * num_sites_uc
    permutable_sites = sc_factor * sum(multiplicities_per_pair)
    if isinstance(permutable_elements_label, dict):
        permutable_elements_label["sites_per_label"] = [sc_factor * m for m in multiplicities_per_pair]
    structural_formula = _recognized_structural_formula(atom_sites, permutable_pairs)

    # -------- Console -----------
    print("CIF Structure Summary")
    print("---------------------")
    print(f"Elements             : {', '.join(elements)}")
    #print(f"Multiplicity         : {', '.join(f'{x:g}' for x in multiplicity)}")
    print(f"Permutable elements  : {', '.join(permutable_elements)}")
    print(f"Structural formula   : {structural_formula}")
    print(f"Sites / permutable   : {sites} / {permutable_sites}\n")

    compound = _recognized_structural_formula(atom_sites, permutable_pairs, latex=True)
        

    return elements, permutable_elements_label, sites, permutable_sites, compound


def eval_cosine(angle:list[float]):
    """ Evaluate cosine(angle) and ignore round off error"""
    eval_angle = []
    for ang in angle:
        x = np.cos(ang)
        if np.isclose(x, 0, atol=1e-15):
            x = 0.0
        eval_angle.append(x)
    return eval_angle
   

# Deprecated #
def get_lattice_parameters(path_to_cif:str | Path):
    """ Extract the lattice vectors from a cif-file """

    # Initialize dictionary to store the values
    lat_par = {}

    # Regular expression to match floats (int, float, scientific)
    float_re = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

    # Open file and parse lines
    with open(path_to_cif, "r") as f:
        for line in f:
            for axis in ["a", "b", "c"]:
                key = f"_cell_length_{axis}"
                if line.startswith(key):
                    match = re.search(float_re, line)
                    if match:
                        lat_par[axis] = float(match.group(1))
            for angle in ["alpha", "beta", "gamma"]:
                key = f"_cell_angle_{angle}"
                if line.startswith(key):
                    match = re.search(float_re, line)
                    if match:
                        lat_par[angle] = float(match.group(1))
    
    return lat_par


#####################################################
# -------------- Set up DFT in files -------------- #
#####################################################
def update_DFT_in(
    out_dir: str | Path,
    qe_par: dict | None = None,
    aims_par: dict | None = None,
    vasp_par: dict | None = None,
    kpts = None,
    cif_file_suffix: str | None = None,
    dft_folder_name: str | None = None,
) -> None:
    """Update the vc-relax input files of an existing cluster tree.

    Use this to change solver parameters after the supercell program has been run and
    the folder structure is set up.

    Parameters
    ----------
    out_dir : str or Path
        Cluster folder containing the ``S*/S*_i*_w*`` subdirectories.
    qe_par, aims_par, vasp_par : dict or None
        Solver parameters, as in :func:`set_up_clusters_and_dft`.
    kpts : tuple or None
        k-point grid for the DFT calculations.
    cif_file_suffix : str or None
        Read ``<cluster_id>_<cif_file_suffix>.cif`` instead of ``<cluster_id>.cif``,
        e.g. to build the inputs from an ML-relaxed structure.
    dft_folder_name : str or None
        DFT run folder inside ``out_dir`` (default ``DFT``). Its ``clusters.dat`` is
        created as a copy of ``clusters_template.dat`` if it does not exist yet; an
        existing one is never changed. Use another name for a second DFT run, or
        the new name of a renamed DFT folder.

    Warnings
    --------
    Input files of the solvers given here are overwritten in every ``vc-relax``
    folder. Input files of other solvers are left untouched.
    """
    qe_par = {} if qe_par is None else dict(qe_par)
    aims_par = {} if aims_par is None else dict(aims_par)
    vasp_par = {} if vasp_par is None else dict(vasp_par)

    create_DFT_in(
        base_path=out_dir,
        qe_par=qe_par,
        aims_par=aims_par,
        vasp_par=vasp_par,
        kpts=kpts,
        cif_file_suffix=cif_file_suffix,
    )
    _copy_template_to_run_folder(out_dir, Path(out_dir) / (dft_folder_name or "DFT"))

def create_DFT_in(
    base_path: str | Path,
    qe_par: dict,
    aims_par: dict,
    vasp_par: dict,
    kpts,
    cif_file_suffix: str | None = None,
):
    """ Create the DFT input files for:
        - vc-relax
        and the folder structure for
        - bands
        - phonons
        for QE, FHI-aims, and VASP if the corresponding paths are given.
    """
    base = Path(base_path)

    def cif_of(sub_dir):
        if cif_file_suffix:
            return sub_dir/f"{sub_dir.name}_{cif_file_suffix}.cif"
        return sub_dir/f"{sub_dir.name}.cif"

    # QE parameters common to all clusters (defaults, cutoffs); every cluster gets a copy
    if "pseudo_dir" in qe_par:
        qe_common = _merge_qe_defaults(qe_par)
        _set_common_qe_cutoffs(qe_common, [cif_of(d) for d in get_directories(base)])
    # FHI-aims parameters common to all clusters, set up once (a k_grid set in the
    # loop would make every later cluster warn about k_grid and kpts both given)
    if "species_dir" in aims_par:
        aims_common = AIMS_DEFAULTS | aims_par
        _set_aims_k_grid(aims_common, kpts)

    for sub_dir in get_directories(base):
        # ---------- cif-file ---------- #
        cif_file = cif_of(sub_dir)

        # ----- make directories ----- #
        relax_dir = Path(cif_file).parent/"vc-relax/"
        bands_dir = Path(cif_file).parent/"bands/"
        phonon_dir = Path(cif_file).parent/"phonons/"
        relax_dir.mkdir(parents=True, exist_ok=True)
        bands_dir.mkdir(parents=True, exist_ok=True)
        phonon_dir.mkdir(parents=True, exist_ok=True)

        # --------- Atoms -------- #
        atoms = read(cif_file)  # extract basis + cell from CIF

        # ---------- QE in ---------- #
        if "pseudo_dir" in qe_par:
            pseudo_dir = Path(qe_par["pseudo_dir"])
            if not pseudo_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {pseudo_dir}")
            

            write_qe_vc_relax(
                atoms,
                relax_dir,
                sub_dir.name,
                qe_par=copy.deepcopy(qe_common),
                kpts = DEFAULT_KPTS if kpts is None else kpts,
            )
        elif qe_par:
            warnings.warn("qe_par was provided but 'pseudo_dir' is missing. QE input files will not be generated.", UserWarning, stacklevel=2)

        # -------- FHI-aims in -------- #
        if "species_dir" in aims_par:
            aims_species_dir = Path(aims_par["species_dir"])
            if not aims_species_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {aims_species_dir}")
            
            write_aims_vc_relax(
                atoms, 
                relax_dir, 
                aims_par=dict(aims_common))
        elif aims_par:
                warnings.warn("aims_par was provided but 'species_dir' is missing. FHI-aims input files will not be generated.", UserWarning, stacklevel=2)

        # --------- VASP in --------- #
        if "potcar_dir" in vasp_par:
            potcar_dir = Path(vasp_par["potcar_dir"])
            if not potcar_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {potcar_dir}")

            vasp_par = VASP_DEFAULTS | vasp_par

            write_vasp_vc_relax(
                atoms,
                relax_dir,
                vasp_par=vasp_par,
                kpts=DEFAULT_KPTS if kpts is None else kpts,
            )
        elif vasp_par:
            warnings.warn(
                "vasp_par was provided but 'potcar_dir' is missing. VASP input files will not be generated.", UserWarning, stacklevel=2
            )

    print("DFT inputs done!")
        

# Called separately after relaxation is done
def create_bands_in(outdir:Path | str,
                    qe_par: dict | None = None, espresso_out_file: str | None = None,
                    aims_par: dict | None = None, aims_out_file: str | None = None,
                    vasp_par: dict | None = None, vasp_out_file: str | None = None,
                    kpts = None, bands: dict | None = None):
    """Create the band-structure input files of every cluster in ``bands/``.

    Called after the relaxation: the structure is read from the relaxation output of
    the corresponding solver, so the band path follows the relaxed cell. Quantum
    ESPRESSO, FHI-aims, and VASP are supported and can be combined in one call.

    Parameters
    ----------
    outdir : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories.
    qe_par, aims_par, vasp_par : dict or None
        Solver parameters. A solver is only processed if its resource directory
        (``pseudo_dir``, ``species_dir``, ``potcar_dir``) is present.
    espresso_out_file, aims_out_file, vasp_out_file : str or None
        Name of the relaxation output file inside ``vc-relax/``. Required for the
        corresponding solver.
    kpts : tuple or None
        k-point grid used for the self-consistent part of the FHI-aims input.
    bands : dict or None
        Band path settings, ``{"path": "", "kpoint_density": 20}`` by default. An
        empty ``path`` lets ASE choose the default path of the Bravais lattice. The
        path is set up once on the unrelaxed S0 cluster and used for every cluster
        (same scaled special points and number of points), see
        :func:`egqca.create_dft_inputs.reference_bandpath`.
    """

    outdir = Path(outdir)
    qe_par = {} if qe_par is None else dict(qe_par)
    aims_par = {} if aims_par is None else dict(aims_par)
    vasp_par = {} if vasp_par is None else dict(vasp_par)
    bands = {"path": "", "kpoint_density": 20} if bands is None else dict(bands)

    # One band path for all clusters, from the unrelaxed S0 cluster (full parent
    # symmetry): same special points (scaled) and number of points for every cluster.
    s0_dirs = [d for d in get_directories(outdir) if d.parent.name == "S0"]
    if not s0_dirs or not (s0_dirs[0]/f"{s0_dirs[0].name}.cif").is_file():
        raise FileNotFoundError(f"No S0 cluster CIF in {outdir}: needed as reference for the band path.")
    bands = reference_bandpath(read(s0_dirs[0]/f"{s0_dirs[0].name}.cif").cell,
                               bands.get("path", ""), bands.get("kpoint_density", 20))

    # QE parameters common to all clusters, with the same cutoffs as create_DFT_in
    if "pseudo_dir" in qe_par:
        qe_common = _merge_qe_defaults(qe_par)
        _set_common_qe_cutoffs(qe_common, [d/f"{d.name}.cif" for d in get_directories(outdir)])
    # FHI-aims parameters common to all clusters, set up once
    if "species_dir" in aims_par:
        aims_common = AIMS_DEFAULTS | aims_par
        _set_aims_k_grid(aims_common, kpts)

    # one pass per cluster folder: the structure comes from the vc-relax output, so
    # further CIFs in the folder (e.g. <id>_<suffix>.cif) must not repeat the inputs
    for sub_dir in get_directories(outdir):
        relax_dir = sub_dir/"vc-relax/"
        bands_dir = sub_dir/"bands/"

        # -------- Espresso -------- #
        if "pseudo_dir" in qe_par:
            pseudo_dir = Path(qe_par["pseudo_dir"])
            if not pseudo_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {pseudo_dir}")
            if espresso_out_file is None:
                raise ValueError("qe_par was given but espresso_out_file is None.")
                
            atoms = read(relax_dir/espresso_out_file)
            write_qe_bands(
                atoms,
                bands_dir,
                sub_dir.name,
                qe_par=copy.deepcopy(qe_common),
                bands = bands
                )
        elif qe_par:
            warnings.warn("qe_par was provided but 'pseudo_dir' is missing. bands.in will not be generated.", UserWarning, stacklevel=2)

        # -------- FHI-aims -------- #
        if "species_dir" in aims_par:
            aims_species_dir = Path(aims_par["species_dir"])
            if not aims_species_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {aims_species_dir}")
            if aims_out_file is None:
                raise ValueError("aims_par was given but aims_out_file is None.")
                
            atoms = read(relax_dir/aims_out_file)
            write_aims_bands(
                atoms,
                bands_dir,
                dict(aims_common),
                bands= bands,
                )
        elif aims_par:
            print(aims_par)
            warnings.warn(
                "aims_par was provided but 'species_dir' is missing. bands/control.in will not be generated.", UserWarning, stacklevel=2
                )

        # -------- VASP bands -------- #
        if "potcar_dir" in vasp_par:
            potcar_dir = Path(vasp_par["potcar_dir"])
            if not potcar_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {potcar_dir}")
            if vasp_out_file is None:
                raise ValueError("vasp_par was given but vasp_out_file is None.")

            vasp_par = VASP_DEFAULTS | vasp_par

            atoms = read(relax_dir / vasp_out_file, format="vasp-out", index=-1)
            write_vasp_bands(
                atoms,
                bands_dir,
                vasp_par=vasp_par,
                bands=bands,
            )
        elif vasp_par:
            warnings.warn(
                "vasp_par was provided but 'potcar_dir' is missing. VASP bands inputs will not be generated.", UserWarning, stacklevel=2
            )

def create_phonons_in(outdir:Path | str,
                    qe_par: dict | None = None, espresso_out_file: str | None = None,
                    aims_par: dict | None = None, aims_out_file: str | None = None,
                    vasp_par: dict | None = None, vasp_out_file: str | None = None,
                    kpts = None, phonon_supercell: tuple[int, int, int] = (1,1,1),
                    aims_calculate_friction: bool = False):
    """Create the phonon input files of every cluster in ``phonons/``.

    Quantum ESPRESSO gets a ``ph.in`` built from ``qe_par`` merged with
    ``QE_PHONON_DEFAULTS``; FHI-aims gets ``geometry.in`` of the relaxed structure
    for the phonon supercell plus ``control.in`` from ``aims_par``. VASP is not
    implemented.

    Parameters
    ----------
    outdir : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories.
    qe_par, aims_par, vasp_par : dict or None
        Solver parameters. For Quantum ESPRESSO this is a ``ph.x`` dictionary, but it
        must still contain ``pseudo_dir``, which is what selects the solver.
    espresso_out_file, aims_out_file, vasp_out_file : str or None
        Name of the relaxation output file inside ``vc-relax/``. Required for Quantum
        ESPRESSO. For FHI-aims the relaxed structure is the last one in
        ``aims_out_file``; if it is None, ``vc-relax/geometry.in.next_step`` is used.
    kpts : tuple or None
        k-point grid written to the FHI-aims ``control.in``.
    phonon_supercell : tuple of int
        Supercell of the FHI-aims phonon geometry, ``atoms.repeat(phonon_supercell)``.
    aims_calculate_friction : bool
        Write ``calculate_friction .true.`` after every atom of the FHI-aims
        ``geometry.in``.
    """

    outdir = Path(outdir)
    qe_par = {} if qe_par is None else dict(qe_par)
    aims_par = {} if aims_par is None else dict(aims_par)
    if vasp_par:
        warnings.warn("VASP phonon inputs are not yet implemented.", UserWarning, stacklevel=2)

    # FHI-aims parameters common to all clusters, set up once
    if "species_dir" in aims_par:
        aims_common = dict(aims_par)
        _set_aims_k_grid(aims_common, kpts)

    # one pass per cluster folder (further CIFs in the folder must not repeat the inputs)
    for sub_dir in get_directories(outdir):
        relax_dir = sub_dir/"vc-relax/"
        phonon_dir = sub_dir/"phonons/"

        # -------- Espresso -------- #
        if "pseudo_dir" in qe_par:
            pseudo_dir = Path(qe_par["pseudo_dir"])
            if not pseudo_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {pseudo_dir}")
            if espresso_out_file is None:
                raise ValueError("qe_par was given but espresso_out_file is None.")
                

            if not (relax_dir/espresso_out_file).is_file():
                raise FileNotFoundError(f"No relaxation output {relax_dir/espresso_out_file}.")
            write_qe_phonons(phonon_dir, sub_dir.name, _qe_phonon_parameters(qe_par))
        elif qe_par:
            warnings.warn("qe_par was provided but 'pseudo_dir' is missing. ph.in will not be generated.", UserWarning, stacklevel=2)

        # -------- FHI-aims -------- #
        if "species_dir" in aims_par:
            aims_species_dir = Path(aims_par["species_dir"])
            if not aims_species_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {aims_species_dir}")

            write_aims_ph(
                _read_aims_relaxed_structure(relax_dir, aims_out_file),
                phonon_dir,
                aims_ph_param=dict(aims_common),
                phonon_supercell=phonon_supercell,
                calculate_friction=aims_calculate_friction,
                )
        elif aims_par:
            print(aims_par)
            warnings.warn(
                "aims_par was provided but 'species_dir' is missing. phonons/control.in will not be generated.", UserWarning, stacklevel=2
                )

###############################################
# ---------- After DFT calculations ----------#
###############################################
_DFT_NOT_FINISHED = "# WARNING: DFT run did not finish or its relaxation did not converge"
# warning written by earlier versions; replaced like the current one
_DFT_NOT_FINISHED_OLD = "# WARNING: DFT calculation did not finish successfully"


def _read_dft_energy(filepath: Path, aims_use_zero_broadening_corrected_energy: bool = False) -> tuple[float | None, bool, str | None]:
    """Total energy in Ry from a QE, FHI-aims or VASP output, whether the run is okay, and the code.

    A run is okay if it finished normally and, for a relaxation, the relaxation
    converged. The energy is None if the file holds none; the code is None if it
    cannot be determined (then the file holds no energy either).
    """
    with open(filepath) as f:
        lines = f.readlines()
    try:
        solver = determine_dft_solver(lines)
    except ValueError:
        return None, False, None

    energy = None
    if solver == "espresso":
        # last occurrence of the total energy line (final scf at the relaxed structure)
        for line in reversed(lines):
            if re.search(r'!\s*total\s+energy', line):
                energy = float(line.split()[4])
                unit = line.split()[5]
                break
        # pw.x writes 'JOB DONE.' only on a normal exit; a run that stops early ends
        # with a '%%%%' error block instead. A BFGS relaxation that hits nstep also
        # ends with 'JOB DONE.', so its convergence is checked separately.
        finished = any("JOB DONE." in line for line in lines)
        relaxation = any("BFGS Geometry Optimization" in line for line in lines)
        relaxation_done = any("bfgs converged" in line for line in lines)
    elif solver == "aims":
        marker = ('| Final zero-broadening corrected energy (caution - metals only)'
                  if aims_use_zero_broadening_corrected_energy
                  else '| Total energy of the DFT / Hartree-Fock s.c.f. calculation')
        finished = relaxation = relaxation_done = False
        for line in lines:
            if marker in line:
                energy = float(line.split()[-2])
                unit = line.split()[-1]
            elif "Have a nice day." in line:
                finished = True
            elif "Geometry relaxation:" in line:
                relaxation = True
            elif "Present geometry is converged." in line:
                relaxation_done = True
    else:   # VASP OUTCAR: last TOTEN line (final ionic step)
        nsw = 0
        finished = relaxation_done = False
        for line in lines:
            if re.match(r'\s*NSW\s*=', line):
                nsw = int(line.split()[2])
            elif re.search(r'free\s+energy\s+TOTEN', line):
                energy = float(line.split()[4])
                unit = line.split()[-1]
            elif 'reached required accuracy - stopping structural energy minimisation' in line:
                relaxation_done = True
            elif 'General timing and accounting informations' in line:
                finished = True
        relaxation = nsw > 0
    dft_calculation_okay = finished and (not relaxation or relaxation_done)

    if energy is None:
        return None, dft_calculation_okay, solver

    # convert to Ry
    if unit == "Ha":
        energy *= 2      # 1 Ha = 2 Ry
    elif unit == "eV":
        energy /= (Ry_to_mev/1000)
    return energy, dft_calculation_okay, solver


def _set_energy_column(line: str, energy: float, dft_calculation_okay: bool) -> str:
    """Row of clusters.dat with the 6th column set to ``energy``, other formatting kept."""
    data, _, comment = line.rstrip("\n").partition("#")
    a, b = [m.span() for m in re.finditer(r"\S+", data)][5]
    row = data[:a] + str(energy) + data[b:]
    comment = comment.strip()
    # an old DFT warning is replaced by the current one
    if comment and comment not in (_DFT_NOT_FINISHED.lstrip("# "), _DFT_NOT_FINISHED_OLD.lstrip("# ")):
        row = row.rstrip() + "    # " + comment
    if not dft_calculation_okay:
        # keep the comment on the same line, otherwise it swallows the next row
        row = row.rstrip() + "    " + _DFT_NOT_FINISHED
    return row + "\n"


def extract_total_energies(datafile: str | Path, clusters_folder:str | Path, out_file_name: str, dft_out_file_subfolder: str = "vc-relax", aims_use_zero_broadening_corrected_energy: bool = False):
    """ 
    Extract the total energy from the DFT output files and write it to clusters.dat .

    Quantum ESPRESSO, FHI-aims, and VASP outputs are recognized automatically and the
    energies are converted to Ry. A run that did not finish successfully, or whose
    relaxation did not converge, keeps its energy but gets a ``# WARNING:`` comment
    appended to its row in clusters.dat.
    The code is determined for every output file; outputs of different codes raise a
    ValueError, since their total energies are not comparable.

    The energy of every cluster in clusters.dat is (re-)extracted from
    ``<clusters_folder>/S<n>/<cluster_id>/<dft_out_file_subfolder>/<out_file_name>``
    and written to clusters.dat. A cluster without folder, output file or energy in
    its output keeps its entry: ``missing`` stays ``missing``, an existing energy is
    kept with a warning. If any cluster is still ``missing`` afterwards, a ValueError
    lists all of them. Cluster folders that are not in clusters.dat only give a
    warning.

    Parameters
    ----------
    datafile : str or Path
        clusters.dat file whose energy column is filled.
    clusters_folder : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories.
    out_file_name : str
        Name of the DFT output file inside each cluster folder.
    dft_out_file_subfolder : str
        Subfolder of each cluster folder holding the DFT output file.
    aims_use_zero_broadening_corrected_energy : bool
        Use the zero-broadening corrected energy of the FHI-aims output instead of
        the total energy.
    """
    datafile = Path(datafile)
    base = Path(clusters_folder)
    with open(datafile) as f:
        lines = f.readlines()
    rows = {
        l.split()[0]: l.split()[5]
        for l in lines if l.strip() and not l.lstrip().startswith("#")
    }

    results = {}
    failed = {}     # cluster_id -> (reason, path)
    solvers = {}    # DFT code -> cluster IDs
    for cluster_id in rows:
        sub_dir = base / cluster_id.split("_")[0] / cluster_id
        filepath = sub_dir / dft_out_file_subfolder / out_file_name
        if not sub_dir.is_dir():
            failed[cluster_id] = ("no cluster folder", sub_dir)
        elif not filepath.is_file():
            failed[cluster_id] = ("no output file", filepath)
        else:
            energy, dft_calculation_okay, solver = _read_dft_energy(filepath, aims_use_zero_broadening_corrected_energy)
            if solver is not None:
                solvers.setdefault(solver, []).append(cluster_id)
            if energy is None:
                failed[cluster_id] = ("no energy in the output file", filepath)
            else:
                results[cluster_id] = (energy, dft_calculation_okay)

    if len(solvers) > 1:
        # total energies of different codes (pseudopotentials, all-electron) are not comparable
        raise ValueError(
            f"The DFT outputs of the clusters in {datafile} come from different codes; their "
            "total energies cannot be compared. Nothing was written.\n"
            + "\n".join(f"  {solver} ({len(ids)}): {', '.join(ids)}" for solver, ids in solvers.items())
        )

    extra = sorted({d.name for d in get_directories(base)} - set(rows))
    if extra:
        warnings.warn(f"Cluster folders not listed in {datafile} are ignored: {', '.join(extra)}")

    new_lines = []
    for line in lines:
        cluster_id = line.split()[0] if line.strip() and not line.lstrip().startswith("#") else None
        if cluster_id not in results:
            new_lines.append(line)
            continue
        energy, dft_calculation_okay = results[cluster_id]
        new_lines.append(_set_energy_column(line, energy, dft_calculation_okay))
        if not dft_calculation_okay:
            warnings.warn(f"DFT run did not finish or its relaxation did not converge for cluster {cluster_id}.")

    # replace the file in one step, so an interrupted run leaves it unchanged
    with _atomic_write(datafile) as f:
        f.write("".join(new_lines))

    kept = [cid for cid in failed if rows[cid] != "missing"]
    if kept:
        warnings.warn(
            f"No DFT energy found for {', '.join(kept)}; the energies already in {datafile.name} are kept."
        )
    still_missing = {cid: failed[cid] for cid in failed if rows[cid] == "missing"}
    if still_missing:
        reasons = {}
        for reason, path in still_missing.values():
            reasons.setdefault(reason, []).append(str(path))
        report = "\n".join(
            f"  {reason} ({len(paths)}):\n" + "\n".join(f"    {p}" for p in paths)
            for reason, paths in reasons.items()
        )
        raise ValueError(
            f"Could not extract the DFT energies of {len(still_missing)} of the {len(rows)} "
            f"clusters in {datafile}:\n{report}\n"
            f"The {len(results)} extracted energies were written to {datafile.name}; the other "
            "clusters stay 'missing'. Every cluster needs its energy before EGQCA can run."
        )


@entry_point("EDDP energy collection", lambda b: [
    ("Clusters folder", b.arguments["out_dir"]),
    ("Compound", b.arguments.get("compound") or "from clusters_template.dat"),
    ("Sites per cluster", b.arguments.get("sites") or "from clusters_template.dat"),
    ("Permutable sites n", b.arguments.get("permutable_sites") or "from clusters_template.dat"),
    ("repose log per cluster", f"eddp/{b.arguments['energy_filename']}"),
    ("Output file", f"eddp/{b.arguments['filename']}"),
])
def extract_eddp_energies(
    out_dir: str | Path,
    permutable_sites: int | None = None,
    sites: int | None = None,
    Nat: int | None = None,
    compound: str | None = None,
    filename: str = "clusters_eddp.dat",
    not_converged_filename: str = "not_converged_clusters.dat",
    energy_filename: str = "energy.dat",
) -> Path:
    """
    Read each cluster's relaxed energy from its eddp/energy.dat file (an
    AIRSS `repose` log, see collect_eddp_clusters.sh) and write
    clusters_eddp.dat into a new "eddp" subfolder of out_dir.

    Clusters whose relaxation did not converge are additionally listed (one
    cluster id per line, with the reason in a second column) in
    not_converged_clusters.dat next to clusters_eddp.dat. A cluster counts as
    converged only if repose logged its "converged in <n> steps" marker.

    Parameters
    ----------
    out_dir : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories.
    permutable_sites, sites, Nat, compound : optional
        Number of permutable sites (x = n_j / n), total number of sites, atoms per
        cluster and compound label for the header. By default read from
        ``<out_dir>/clusters_template.dat`` (else ``DFT/clusters.dat`` of an older
        tree, else a scan of the cluster CIFs, which writes the template). A value
        that differs from the template raises a ValueError.
    filename : str
        Name of the cluster file written into the ``eddp`` subfolder.
    not_converged_filename : str
        Name of the summary file listing the clusters without a converged relaxation.
    energy_filename : str
        Name of the repose output file inside each cluster's ``eddp`` subfolder.
        Defaults to ``energy.dat``. Use this to read an alternative relaxation of
        the same cluster set, for example ``RHO.energy.dat`` from a run with the
        cell constrained to the rhombohedral Bravais lattice.

    Returns
    -------
    pathlib.Path
        Path to the written cluster file.
    """
    out_dir = Path(out_dir)
    permutable_sites, sites, Nat, compound, _ = _resolve_cluster_setup(
        out_dir, permutable_sites=permutable_sites, sites=sites, Nat=Nat, compound=compound,
    )
    eddp_dir = out_dir / "eddp"
    eddp_dir.mkdir(exist_ok=True)
    clusters_file = eddp_dir / filename
    not_converged_file = eddp_dir / not_converged_filename

    directories = get_directories(out_dir)
    clusters_info = []
    not_converged = []
    # per-S counts for the header of not_converged_clusters.dat, keyed by n_j
    n_total_per_S = {}
    n_not_converged_per_S = {}
    n_missing = 0
    n_incomplete = 0

    for sub_dir in directories:
        cluster_id = sub_dir.name
        energy_file = sub_dir / "eddp" / energy_filename

        match = re.search(r"S(\d+)_", cluster_id)
        n_j = int(match.group(1)) if match else None
        x = n_j / permutable_sites if n_j is not None else None
        match = re.search(r"_w(\d+)", cluster_id)
        g_j = int(match.group(1)) if match else None

        n_total_per_S[n_j] = n_total_per_S.get(n_j, 0) + 1

        if not energy_file.exists():
            n_missing += 1
            not_converged.append((cluster_id, f"missing eddp/{energy_filename}"))
            n_not_converged_per_S[n_j] = n_not_converged_per_S.get(n_j, 0) + 1
            clusters_info.append((cluster_id, n_j, Nat, x, g_j, "missing"))
            continue

        with open(energy_file) as f:
            lines = f.readlines()

        # repose ends the optimisation block with either "converged in <n> steps"
        # or "did not converge after <n> steps" (followed by "STOP repose: failed
        # to converge"). A truncated log has neither -- treat that as unconverged
        # too, since convergence cannot be confirmed.
        converged = any("converged in" in line for line in lines)
        if not converged:
            if any("did not converge" in line or "failed to converge" in line for line in lines):
                reason = "repose did not converge"
            else:
                reason = f"no convergence marker in eddp/{energy_filename}"
            not_converged.append((cluster_id, reason))
            n_not_converged_per_S[n_j] = n_not_converged_per_S.get(n_j, 0) + 1

        energy_eV = None
        for line in reversed(lines):
            if line.strip().startswith("Energy:"):
                energy_eV = float(line.split(":")[1].strip())
                break
        if energy_eV is None:
            n_incomplete += 1
            warnings.warn(f"No 'Energy:' line in {energy_file}, skipping cluster {cluster_id}.")
            clusters_info.append((cluster_id, n_j, Nat, x, g_j, "missing"))
            continue

        energy_Ry = energy_eV / (Ry_to_mev / 1000)

        # an energy from an unconverged relaxation is kept but flagged on its row, as
        # in the ML route; read_clusters_dat repeats the warning when the file is read
        clusters_info.append((cluster_id, n_j, Nat, x, g_j, energy_Ry) if converged
                             else (cluster_id, n_j, Nat, x, g_j, energy_Ry, "\t  # WARNING: Relaxation did not converge"))

    _write_ml_clusters_header(
        clusters_file,
        permutable_sites=permutable_sites,
        sites=sites,
        compound=compound,
        ML_model="eddp",
    )
    with open(clusters_file, "a") as f:
        for cluster_info in clusters_info:
            row, comment = cluster_info[:6], "".join(cluster_info[6:])
            f.write("\t".join(map(str, row)) + comment + "\n")
    if any(len(cluster_info) > 6 for cluster_info in clusters_info):
        # same header warning as the ML route
        text = clusters_file.read_text()
        clusters_file.write_text(
            "# WARNING: Relaxation did not converge for some of the clusters. "
            f"They are listed in {not_converged_file.name}.\n" + text
        )

    with open(not_converged_file, "w") as f:
        f.write(f"# Clusters without a converged eddp relaxation ({len(not_converged)} of {len(directories)})\n")
        # per-S breakdown; S* groups in which everything converged are not listed
        for n_j in sorted(n_not_converged_per_S, key=lambda k: (k is None, k)):
            S_label = f"S{n_j}" if n_j is not None else "unknown"
            f.write(f"# {S_label}: {n_not_converged_per_S[n_j]}/{n_total_per_S[n_j]} not converged\n")
        f.write("#ID\treason\n")
        for cluster_id, reason in not_converged:
            f.write(f"{cluster_id}\t{reason}\n")

    print(
        f"#####################################################################################\n"
        f"Summary:\n"
        f"({n_missing} missing eddp/{energy_filename}, {n_incomplete} without a completed relaxation).\n"
        f"{len(not_converged)} of {len(directories)} clusters did not converge, listed in {not_converged_file}.\n"
        f"#####################################################################################\n"
    )
    return clusters_file


def set_up_volume_dat(clusters_folder: str | Path, dft_out_file: str | None = None, dft_out_file_subfolder: str = "vc-relax", filename: str = "volume.dat", run_folder: str | Path | None = None):
    """Write volume.dat with the lattice parameters of every cluster from its DFT output.

    The file holds a, b, c in Angstrom and alpha, beta, gamma in degree, read from
    the relaxed structure in ``<cluster>/<dft_out_file_subfolder>/<dft_out_file>``.
    It is written into the DFT run folder ``run_folder`` (default
    ``<clusters_folder>/DFT``). The ML route writes its own ``volume_<model>.dat``.

    Parameters
    ----------
    clusters_folder : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories.
    dft_out_file : str
        Name of the DFT output file to read the relaxed structure from.
    dft_out_file_subfolder : str
        Subfolder of each cluster folder holding the DFT output file.
    filename : str
        Name of the output file.
    run_folder : str or Path or None
        DFT run folder the file is written into; default ``<clusters_folder>/DFT``.

    Returns
    -------
    tuple
        ``(volume_file, cluster_id, a, b, c, alpha, beta, gamma)``.
    """

    clusters_folder = Path(clusters_folder)
    if dft_out_file is None:
        raise ValueError("dft_out_file is required: volume.dat is read from the DFT output files.")
    run_folder = clusters_folder / "DFT" if run_folder is None else Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    volume_file = run_folder / filename

    volume_info = []
    cluster_id = []
    for sub_dir in get_directories(clusters_folder):
        cluster_id.append(sub_dir.name)

        path_dft_out = sub_dir/dft_out_file_subfolder/dft_out_file
        dft_solver = determine_dft_solver(path_dft_out)

        # ASE
        if dft_solver == "vasp":
            atom = read(path_dft_out, format="vasp-out", index=-1)
        else:
            with open(path_dft_out, "r") as f:
                if dft_solver == "espresso":
                    atom = list(read_espresso_out(f))[-1]
                elif dft_solver == "aims":
                    atom = read_aims_output(f)

        a,b,c,alpha,beta,gamma = atom.cell.cellpar()

        volume_info.append((a, b, c, alpha, beta, gamma))

    # Write the extracted information to the volume.dat file
    _write_ml_volume_header(volume_file, ML_model=None)
    with open(volume_file, "a") as f:
        for cl, vol in zip(cluster_id, volume_info):
            f.write(cl+"\t"+"\t".join(map(str, vol)) + "\n")

    a,b,c,alpha,beta, gamma = np.array(volume_info, dtype=float).T
    return volume_file, np.array(cluster_id).T, a,b,c,alpha,beta, gamma


def write_properties_file(clusters_folder: str | Path, properties_extraction: dict, filename: str = "properties.dat"):
    """Extract per-cluster properties and write them to a properties.dat file.

    The file is written into ``clusters_folder`` and has the format expected by
    :meth:`egqca.EGQCA.average_properties`: the cluster ID in the first column and
    one column per property.

    Parameters
    ----------
    clusters_folder : str or Path
        Folder containing the ``S*/S*_i*_w*`` cluster subdirectories; the file is
        written into it. A ValueError is raised if it holds no cluster folders.
    properties_extraction : dict
        Mapping of property name to a callable that accepts a cluster directory and
        returns the value of that property for the cluster.
    filename : str
        Name of the output file.

    Returns
    -------
    pathlib.Path
        Path to the written properties file.

    Examples
    --------
    >>> egqca.write_properties_file(
    ...     out_dir,
    ...     {"N(e_F)": lambda cluster_dir: read_dos_at_ef(cluster_dir / "bands")},
    ... )
    """
    clusters_folder = Path(clusters_folder)
    directories = get_directories(clusters_folder)
    if not directories:
        raise ValueError(
            f"No S*/S*_i*_w* cluster folders in {clusters_folder}. write_properties_file "
            "takes the clusters folder, the folder that contains the S* folders."
        )
    prop_file = clusters_folder / filename

    prop_names = list(properties_extraction.keys())

    with open(prop_file, "w") as f:
        f.write("# Properties for EGQCA\n")
        f.write("#ID\t" + "\t".join(prop_names) + "\n")
        for sub_dir in directories:
            values = [extractor(sub_dir) for extractor in properties_extraction.values()]
            f.write(sub_dir.name + "\t" + "\t".join(str(v) for v in values) + "\n")

    return prop_file




##############################################################
# -------------------- Clusters template ------------------- #
##############################################################

def _compound_from_cif(cif_file: str | Path) -> str:
    symbols = read(cif_file).get_chemical_symbols()
    counts = Counter(symbols)
    ordered_symbols = list(dict.fromkeys(symbols))
    return "".join(
        symbol if counts[symbol] == 1 else f"{symbol}$_{{{counts[symbol]}}}$"
        for symbol in ordered_symbols
    )


def _format_latex_variable_term(symbol: str, coefficient: int) -> str:
    if coefficient == 1:
        return f"{symbol}$_{{x}}$"
    return f"{symbol}$_{{{coefficient}x}}$"


def _compound_from_cif_pair(base_cif: str | Path, substituted_cif: str | Path) -> str:
    base_symbols = read(base_cif).get_chemical_symbols()
    substituted_symbols = read(substituted_cif).get_chemical_symbols()
    base_counts = Counter(base_symbols)
    substituted_counts = Counter(substituted_symbols)
    ordered_symbols = list(dict.fromkeys(base_symbols + substituted_symbols))

    deltas = {
        symbol: substituted_counts.get(symbol, 0) - base_counts.get(symbol, 0)
        for symbol in ordered_symbols
    }
    original_symbols = [symbol for symbol in ordered_symbols if deltas[symbol] < 0]
    substituting_symbols = [symbol for symbol in ordered_symbols if deltas[symbol] > 0]
    if not original_symbols or not substituting_symbols:
        return _compound_from_cif(base_cif)

    scale = 0
    for value in list(base_counts.values()) + [abs(delta) for delta in deltas.values() if delta != 0]:
        scale = gcd(scale, value)
    scale = max(scale, 1)

    scaled_base = {symbol: base_counts.get(symbol, 0) // scale for symbol in ordered_symbols}
    scaled_delta = {symbol: deltas[symbol] // scale for symbol in ordered_symbols}
    x_max = min(scaled_base[symbol] for symbol in original_symbols)

    # x in [0, 1] as in the clusters files, permutable group first, as in
    # _recognized_structural_formula: (A$_{1-x}$B$_{x}$)$_{m}$C$_{k}$
    substituting_terms = "".join(
        _format_latex_variable_term(symbol, abs(scaled_delta[symbol]) // x_max)
        for symbol in substituting_symbols
    )
    if len(original_symbols) > 1:
        original_group = "".join(
            symbol if scaled_base[symbol] == 1 else f"{symbol}{scaled_base[symbol]}"
            for symbol in original_symbols
        )
        formula_parts = [f"({original_group})$_{{1-x}}$", substituting_terms]
    else:
        group = f"{original_symbols[0]}$_{{1-x}}$" + substituting_terms
        formula_parts = [group if x_max == 1 else f"({group})$_{{{x_max}}}$"]

    unchanged_symbols = [
        symbol
        for symbol in ordered_symbols
        if symbol not in original_symbols and symbol not in substituting_symbols and scaled_base[symbol] > 0
    ]
    for symbol in unchanged_symbols:
        count = scaled_base[symbol]
        formula_parts.append(symbol if count == 1 else f"{symbol}$_{{{count}}}$")

    return "".join(formula_parts)


def _header_permutable_sites(path: str | Path) -> int | None:
    """n from the ``# permutable sites n=`` header line of a clusters file, None without one."""
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"permutable sites n=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def _clusters_template_path(clusters_folder: str | Path) -> Path:
    return Path(clusters_folder) / CLUSTERS_TEMPLATE_NAME


def _template_from_clusters_file(source: Path, template: Path) -> None:
    """Write ``template`` from a clusters.dat: same header and rows, energies ``missing``.

    Comments on the rows and header warnings of the source are dropped.
    """
    lines = []
    for line in source.read_text().splitlines():
        if line.startswith("#"):
            if not line.startswith("# WARNING") and not line.startswith("# uMLIP model"):
                lines.append(line)
            continue
        fields = line.partition("#")[0].split()
        if len(fields) >= 6:
            fields[5] = "missing"
        if fields:
            lines.append("\t".join(fields))
    with _atomic_write(template) as f:
        f.write("\n".join(lines) + "\n")


def _find_clusters_template(clusters_folder: str | Path, clusters_template: str | Path | None = None) -> Path | None:
    """The clusters template to read the cluster setup from, or None.

    ``clusters_template`` if given, else ``<clusters_folder>/clusters_template.dat``,
    else ``<clusters_folder>/DFT/clusters.dat`` of an older tree, from which
    clusters_template.dat is then written once (energies set to ``missing``).
    """
    if clusters_template is not None:
        return Path(clusters_template)
    template = _clusters_template_path(clusters_folder)
    if template.is_file():
        return template
    legacy = Path(clusters_folder) / LEGACY_DFT_CLUSTERS
    if legacy.is_file():
        _template_from_clusters_file(legacy, template)
        print(f"Wrote {template} from {legacy}.")
        return template
    return None


def _check_explicit(name: str, given, template_value, template: Path) -> None:
    if given is not None and given != template_value:
        raise ValueError(
            f"{name}={given!r} differs from {template_value!r} in {template}. Leave {name} "
            "out to use the template, or fix the template."
        )


def _resolve_permutable_sites(
    clusters_folder: str | Path,
    permutable_sites: int | None = None,
    clusters_template: str | Path | None = None,
) -> tuple[int, Path | None]:
    """Number of permutable sites n and the clusters template it was read from.

    The template is found as in :func:`_find_clusters_template`. Without one, n is
    ``permutable_sites`` or, by default, the highest S* folder number; the latter is
    wrong if the S* folders hold only a sub-range of compositions. A
    ``permutable_sites`` that differs from the template raises a ValueError. The
    returned path is None without a template.
    """
    template = _find_clusters_template(clusters_folder, clusters_template)
    if template is None:
        if permutable_sites is not None:
            return int(permutable_sites), None
        s_numbers = [
            int(m.group(1)) for d in Path(clusters_folder).glob("S*")
            if d.is_dir() and (m := re.fullmatch(r"S(\d+)", d.name))
        ]
        if not s_numbers:
            raise FileNotFoundError(
                f"No S* cluster folders in {clusters_folder} and no clusters template to "
                "read the number of permutable sites n from. Pass permutable_sites."
            )
        return max(s_numbers), None
    header_n = _parse_clusters_template(template)[0]
    _check_explicit("permutable_sites", None if permutable_sites is None else int(permutable_sites),
                    header_n, template)
    return header_n, template


def _resolve_cluster_setup(
    clusters_folder: str | Path,
    *,
    permutable_sites: int | None = None,
    sites: int | None = None,
    Nat: int | None = None,
    compound: str | None = None,
    clusters_template: str | Path | None = None,
) -> tuple[int, int, int, str, list[str]]:
    """``(permutable_sites, sites, Nat, compound, cluster_ids)`` of a cluster tree.

    Read from the clusters template (see :func:`_find_clusters_template`); a value
    given here that differs from it raises a ValueError. Without a template, sites,
    Nat and compound come from a scan of the cluster CIFs (values given here take
    precedence), n as in :func:`_resolve_permutable_sites`, and
    ``<clusters_folder>/clusters_template.dat`` is written from the result.
    """
    n, template = _resolve_permutable_sites(clusters_folder, permutable_sites, clusters_template)
    if template is not None:
        _, t_sites, t_Nat, t_compound = _parse_clusters_template(template)
        _check_explicit("sites", sites, t_sites, template)
        _check_explicit("Nat", Nat, t_Nat, template)
        _check_explicit("compound", compound, t_compound, template)
        return n, t_sites, t_Nat, t_compound, _cluster_ids_from_template(template)
    _, s_sites, s_Nat, s_compound, cluster_ids = _infer_cluster_setup(clusters_folder, n)
    sites = s_sites if sites is None else sites
    Nat = s_Nat if Nat is None else Nat
    compound = s_compound if compound is None else compound
    set_up_clusters_dat(clusters_folder, n, sites, Nat, compound)
    print(f"Wrote {_clusters_template_path(clusters_folder)} from the cluster CIFs (n={n}).")
    return n, sites, Nat, compound, cluster_ids


def _copy_template_to_run_folder(clusters_folder: str | Path, run_folder: str | Path) -> Path:
    """``<run_folder>/clusters.dat`` as a copy of the clusters template; never overwritten.

    The template is created first if the tree has none (see :func:`_resolve_cluster_setup`).
    """
    run_file = Path(run_folder) / "clusters.dat"
    if run_file.exists():
        return run_file
    template = _find_clusters_template(clusters_folder)
    if template is None:
        _resolve_cluster_setup(clusters_folder)
        template = _clusters_template_path(clusters_folder)
    run_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, run_file)
    return run_file


def _infer_cluster_setup(clusters_folder: str | Path, permutable_sites: int):
    """``(permutable_sites, sites, Nat, compound, cluster_ids)`` from the cluster CIFs.

    ``permutable_sites`` is passed through (see :func:`_resolve_permutable_sites`).
    """
    clusters_folder = Path(clusters_folder)
    cluster_dirs = [p for p in clusters_folder.glob("S*") if p.is_dir()]
    if not cluster_dirs:
        raise FileNotFoundError(f"No S* cluster directories found in {clusters_folder}.")

    Nat = 0
    first_cif = None
    cifs_by_S = {}
    cluster_ids: list[str] = []
    sorted_cluster_dirs = sorted(cluster_dirs, key=lambda p: _extract_snumber_from_name(p.name))
    n_s_dirs = len(sorted_cluster_dirs)
    for i, S_dir in enumerate(sorted_cluster_dirs, 1):
        print(f"\rInferring cluster setup: {S_dir.name} ({i}/{n_s_dirs})", end="", flush=True)
        for sub_dir in sorted(S_dir.glob("S*_i*_w*")):
            if not sub_dir.is_dir():
                continue
            cluster_ids.append(sub_dir.name)
            cif_file = _original_cluster_cif(sub_dir)
            if cif_file is None:
                continue
            Nat = max(Nat, len(read(cif_file)))
            first_cif = cif_file if first_cif is None else first_cif
            cifs_by_S.setdefault(_extract_snumber_from_name(S_dir.name), cif_file)
    print()

    if Nat == 0:
        raise ValueError(f"Could not infer atom count from CIF files in {clusters_folder}.")

    S_values = sorted(cifs_by_S)
    if len(S_values) > 1:
        compound = _compound_from_cif_pair(cifs_by_S[S_values[0]], cifs_by_S[S_values[-1]])
    else:
        compound = _compound_from_cif(first_cif)

    return permutable_sites, Nat, Nat, compound, cluster_ids


def _parse_clusters_template(path: str | Path) -> tuple[int, int, int, str]:
    """Read permutable_sites, sites, Nat, and compound from a clusters.dat header.

    Data rows are only scanned to obtain the first valid Nat value.
    """
    path = Path(path)
    permutable_sites: int | None = None
    sites: int | None = None
    compound: str | None = None
    Nat: int | None = None

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                m = re.search(r"compound=(.+)", stripped)
                if m:
                    compound = m.group(1).strip()
                m = re.search(r"total number of sites=\s*(\d+)", stripped)
                if m:
                    sites = int(m.group(1))
                m = re.search(r"permutable sites n=\s*(\d+)", stripped)
                if m:
                    permutable_sites = int(m.group(1))
                continue
            if Nat is None:
                fields = stripped.split()   # tabs or spaces
                if len(fields) >= 3 and fields[2] not in ("", "None"):
                    try:
                        Nat = int(fields[2])
                    except ValueError:
                        pass

    missing_fields = [
        name for name, val in [
            ("permutable_sites", permutable_sites),
            ("sites", sites),
            ("compound", compound),
            ("Nat", Nat),
        ] if val is None
    ]
    if missing_fields:
        raise ValueError(
            f"Could not parse {', '.join(missing_fields)} from template {path}."
        )
    return permutable_sites, sites, Nat, compound


def _cluster_ids_from_template(path: str | Path) -> list[str]:
    """Return all cluster IDs (first column) from a clusters.dat file."""
    ids = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cluster_id = stripped.split()[0]   # tabs or spaces
            if cluster_id:
                ids.append(cluster_id)
    return ids


##############################################################
# -------------------- Helper functions -------------------- #
##############################################################

# Lines that only the output of one code contains: the program header first, the
# energy lines as a fallback for outputs without a header. Matched case-sensitively,
# so a path or pseudopotential name containing e.g. "aims" does not count.
_DFT_SOLVER_MARKERS = {
    "espresso": (re.compile(r"Program PWSCF|Quantum ESPRESSO"), re.compile(r"!\s*total\s+energy")),
    "aims": (re.compile(r"FHI-aims"),
             re.compile(r"\| Total energy of the DFT / Hartree-Fock s\.c\.f\. calculation"
                        r"|\| Final zero-broadening corrected energy")),
    "vasp": (re.compile(r"^\s*vasp\.\d|POTCAR:"), re.compile(r"free\s+energy\s+TOTEN")),
}


def determine_dft_solver(dft_out: str | Path | list[str]) -> str:
    """DFT code that wrote an output file: ``"espresso"``, ``"aims"`` or ``"vasp"``.

    ``dft_out`` is the path of the output file or its lines. A program header
    decides; without one, the first energy line of one of the codes does.
    Raises a ValueError if neither is found.
    """
    if isinstance(dft_out, (str, Path)):
        with open(dft_out) as f:
            lines = f.readlines()
        name = str(dft_out)
    else:
        lines, name = dft_out, "the DFT output"
    for kind in (0, 1):     # 0: header, 1: energy line
        for line in lines:
            for solver, markers in _DFT_SOLVER_MARKERS.items():
                if markers[kind].search(line):
                    return solver
    raise ValueError(f"Could not determine the DFT code (espresso, aims or vasp) of {name}.")


def extract_Snumber(path):
    return int(re.search(r"S(\d+)", path.name).group(1))


def _merge_qe_defaults(qe_par: dict) -> dict:
    """New QE parameter dict: ``QE_DEFAULTS`` merged with ``qe_par`` per namelist.

    A namelist given in ``qe_par`` (e.g. ``"system"``) only overrides the keys it
    contains. Neither ``QE_DEFAULTS`` nor ``qe_par`` is modified.
    """
    merged = copy.deepcopy(QE_DEFAULTS)
    for key, value in qe_par.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(copy.deepcopy(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _qe_phonon_parameters(qe_par: dict) -> dict:
    """New ph.x parameter dict: ``QE_PHONON_DEFAULTS`` updated with ``qe_par``.

    ``qe_par`` holds &INPUTPH variables, flat or under ``"inputph"``. ``pseudo_dir``
    only selects the QE solver and is not a ph.x variable, so it is left out.
    """
    user = {k: v for k, v in qe_par.items() if k not in ("pseudo_dir", "inputph")}
    user.update(qe_par.get("inputph", {}))
    return copy.deepcopy(QE_PHONON_DEFAULTS) | copy.deepcopy(user)


def _set_aims_k_grid(aims_par: dict, kpts) -> None:
    """Set ``aims_par["k_grid"]`` in place.

    ``k_grid_density`` in ``aims_par`` wins and no ``k_grid`` is set. Otherwise
    ``kpts`` is used if given, then a ``k_grid`` already in ``aims_par``, then
    ``DEFAULT_KPTS``.
    """
    if "k_grid_density" in aims_par:
        if kpts is not None:
            warnings.warn("Both 'k_grid_density' in aims_par and kpts are provided. 'k_grid_density' will take precedence.", UserWarning, stacklevel=3)
    elif kpts is not None:
        if "k_grid" in aims_par:
            warnings.warn("Both 'k_grid' in aims_par and kpts are provided. kpts will take precedence.", UserWarning, stacklevel=3)
        aims_par["k_grid"] = kpts
    elif "k_grid" not in aims_par:
        aims_par["k_grid"] = DEFAULT_KPTS


def _read_aims_relaxed_structure(relax_dir: Path, aims_out_file: str | None = None):
    """Relaxed structure of an FHI-aims run in ``relax_dir``.

    Read from the last structure in ``aims_out_file``, or from
    ``geometry.in.next_step`` if ``aims_out_file`` is None. Warns if the output
    shows an unfinished run or an unconverged relaxation.
    """
    if aims_out_file is None:
        next_step = relax_dir/"geometry.in.next_step"
        if not next_step.is_file():
            raise FileNotFoundError(f"aims_out_file is None and there is no {next_step}.")
        return read(next_step, format="aims")

    aims_out = relax_dir/aims_out_file
    if not aims_out.is_file():
        raise FileNotFoundError(f"No relaxation output {aims_out}.")
    text = aims_out.read_text()
    if "Have a nice day." not in text:
        warnings.warn(f"FHI-aims run {aims_out} did not finish; using its last structure.", UserWarning, stacklevel=3)
    elif "Geometry relaxation:" in text and "Present geometry is converged." not in text:
        warnings.warn(f"Relaxation in {aims_out} did not converge; using its last structure.", UserWarning, stacklevel=3)
    return read(aims_out)


def _set_common_qe_cutoffs(qe_par: dict, structure_files) -> None:
    """Set the default ``ecutwfc``/``ecutrho`` of ``qe_par`` from all clusters.

    All clusters must use the same cutoffs, otherwise their total energies cannot be
    compared. The default ``ecutwfc`` is the PseudoDojo value of the hardest element in
    any of ``structure_files``, ``ecutrho`` is 8 * ``ecutwfc``. Cutoffs given by the
    user (in ``qe_par["system"]`` or at the top level) are kept.
    """
    system = qe_par.setdefault("system", {})
    if "ecutwfc" not in qe_par and "ecutwfc" not in system:
        elements = sorted(set().union(*(read(f).get_chemical_symbols() for f in structure_files)))
        system["ecutwfc"] = 2 * _default_ecutwfc_Ha(elements)
    if "ecutrho" not in qe_par and "ecutrho" not in system:
        system["ecutrho"] = 8 * qe_par.get("ecutwfc", system.get("ecutwfc"))


def _read_cluster_setup_metadata(clusters_file: str | Path):
    clusters_file = Path(clusters_file)
    compound = None
    sites = None
    permutable_sites = None

    with open(clusters_file) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "compound=" in line and compound is None:
                compound = line.split("compound=")[1].split()[0]
            if "total number of sites=" in line and sites is None:
                sites = int(line.split("total number of sites=")[1].split()[0])
            if "permutable sites n=" in line and permutable_sites is None:
                permutable_sites = int(line.split("permutable sites n=")[1].split()[0])

    if compound is None or sites is None or permutable_sites is None:
        raise ValueError(
            f"Could not extract compound/sites/permutable_sites from {clusters_file}."
        )

    Nat = np.genfromtxt(clusters_file, usecols=2, comments="#", ndmin=1)
    if Nat.size == 0:
        raise ValueError(f"No cluster entries found in {clusters_file}.")

    return permutable_sites, sites, int(np.max(Nat)), compound
