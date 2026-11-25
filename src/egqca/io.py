from pathlib import Path
import warnings
import numpy as np
from .subclasses import lattice
from .constants import GPA_TO_PA, J_to_meV, cm_minus1_to_Hz, meV_to_Hz, hbar
import re
from .workflow import set_up_volume_dat, extract_total_energies
from ._ml_utils import _atomic_write

# Frequency units of a phonon DOS header and their factor to Hz. Checked in this order;
# the lookbehinds keep meV from matching eV and THz/GHz from matching Hz.
_PHONON_UNITS = (
    (r"cm\^-1|1/cm", cm_minus1_to_Hz),
    (r"THz", 1.0e12),
    (r"GHz", 1.0e9),
    (r"(?<![A-Za-z])Hz|s\^-1", 1.0),
    (r"meV", meV_to_Hz),
    (r"(?<![A-Za-z])eV", meV_to_Hz * 1000),
)


#############################################
# ---------------- Helpers ---------------- #
#############################################
def sort_by_id(cluster_id, *arrays):
    """Sort arrays according to cluster_id."""

    # ensure order 1,2,...,10,11,... instead of 1,10,11,...,2,3,...
    S = np.array([int(re.search(r"S(\d+)", s).group(1)) for s in cluster_id])
    i = np.array([int(re.search(r"i(\d+)", s).group(1)) for s in cluster_id])
    order = np.lexsort((i, S))
    
    cluster_id_sorted = cluster_id[order]
    sorted_arrays = tuple(arr[order] for arr in arrays)

    return (cluster_id_sorted, *sorted_arrays)


def reorder_by_reference_id(reference_id, cluster_id, *arrays):
    """Reorder arrays to match the order of ``reference_id``."""

    cluster_id = np.asarray(cluster_id)
    reference_id = np.asarray(reference_id)
    order_map = {cluster: idx for idx, cluster in enumerate(cluster_id)}

    missing = [cluster for cluster in reference_id if cluster not in order_map]
    extra = [cluster for cluster in cluster_id if cluster not in set(reference_id)]
    if missing or extra:
        message = []
        if missing:
            message.append(f"missing IDs: {missing}")
        if extra:
            message.append(f"extra IDs: {extra}")
        raise ValueError("Cluster IDs do not match reference order (" + "; ".join(message) + ").")

    order = np.array([order_map[cluster] for cluster in reference_id], dtype=int)
    reordered_arrays = tuple(np.asarray(arr)[order] for arr in arrays)

    return (cluster_id[order], *reordered_arrays)


def convert_volume_file(
    volume_file: str | Path,
    conversion_function,
    volume_file_name: str | None = None,
) -> Path:
    """
    Convert lattice parameters in a volume.dat-style file on demand.

    The volume-file convention is:
    ID, a, b, c, alpha, beta, gamma, with lengths in Angstrom and angles
    in degrees. The conversion function is called once for each data row as
    ``conversion_function(a, b, c, alpha, beta, gamma)``. It must return either
    a sequence ``(a, b, c, alpha, beta, gamma)`` or a dictionary with these
    six keys.

    Parameters
    ----------
    volume_file
        Input volume file.
    conversion_function
        Callable that converts one row of lattice parameters.
    volume_file_name
        Optional output filename. If omitted, the output is written next to the
        input as ``<input filename>_converted.dat``.

    Returns
    -------
    pathlib.Path
        Path to the converted volume file.
    """

    volume_file = Path(volume_file)
    if volume_file_name is None:
        volume_name = volume_file.name
        if volume_name.endswith(".dat"):
            volume_name = volume_name[:-4]
        converted_volume_file = volume_file.with_name(f"{volume_name}_converted.dat")
    else:
        converted_volume_file = Path(volume_file_name)
        if not converted_volume_file.is_absolute():
            converted_volume_file = volume_file.parent / converted_volume_file

    if converted_volume_file.exists():
        print(f"Converted volume file already exists: {converted_volume_file}")
        return converted_volume_file

    if not callable(conversion_function):
        raise TypeError("conversion_function must be callable.")

    lattice_keys = ("a", "b", "c", "alpha", "beta", "gamma")

    # written to a temporary file and moved into place only when every row converted:
    # a failing row raises and leaves no truncated file that a later call would return
    with open(volume_file) as fin, _atomic_write(converted_volume_file) as fout:
        for line_number, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                fout.write(line)
                continue

            columns = stripped.split()
            if len(columns) != 7:
                raise ValueError(
                    f"Expected 7 columns in {volume_file} line {line_number}: "
                    "ID, a, b, c, alpha, beta, gamma."
                )

            cluster_id = columns[0]
            lattice_parameters = tuple(float(value) for value in columns[1:])
            converted = conversion_function(*lattice_parameters)

            if isinstance(converted, dict):
                try:
                    converted_values = tuple(converted[key] for key in lattice_keys)
                except KeyError as exc:
                    raise ValueError(
                        "conversion_function returned a dictionary, but one of "
                        "the required keys is missing: a, b, c, alpha, beta, gamma."
                    ) from exc
            else:
                converted_values = tuple(converted)

            if len(converted_values) != 6:
                raise ValueError(
                    "conversion_function must return six values: "
                    "a, b, c, alpha, beta, gamma."
                )

            fout.write(
                cluster_id
                + "\t"
                + "\t".join(str(float(value)) for value in converted_values)
                + "\n"
            )

    return converted_volume_file


###################################################################
# --------------------------- Read-in --------------------------- #
###################################################################
def _warn_row_comments(lines: list[str], datafile: str | Path) -> None:
    """One warning per distinct ``# WARNING:`` comment on a data row, listing its clusters."""
    flagged = {}    # warning text -> cluster IDs
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        data, _, comment = line.partition("#")
        comment = comment.strip()
        if comment.startswith("WARNING:") and data.split():
            flagged.setdefault(comment[len("WARNING:"):].strip(), []).append(data.split()[0])
    for text, ids in flagged.items():
        warnings.warn(f"{Path(datafile).name}: {text.rstrip('.')} for {len(ids)} cluster(s): {', '.join(ids)}",
                      UserWarning, stacklevel=3)


def read_clusters_dat(self,datafile: str | Path, dft_out_file:str, sites:int, n:int, compound:str, dft_out_file_subfolder:str, aims_use_zero_broadening_corrected_energy:bool=False):
    """
    Read the clusters.dat file and extract the relevant information for the EGQCA calculation.
    
    Parameters
    ----------
    datafile : str or Path
        Path to the clusters.dat file.
    dft_out_file : str
        Name of the DFT output file.
    sites : int or None
        Number of total sites. Extracted from clusters.dat if None.
    n : int or None
        Number of permutable sites. Extracted from clusters.dat if None.
    compound : str or None
        Chemical formula of the compound. Extracted from clusters.dat if None.
    dft_out_file_subfolder : str
        Subfolder of each cluster folder holding the DFT output file.
    aims_use_zero_broadening_corrected_energy : bool
        Whether to use the zero-broadening corrected energy from the FHI-aims
        output instead of the total energy.

    Returns
    -------
    tuple
        ``(cluster_id, nj, Nat, xj, gj, Ej_Ry, sites, n, compound)``, sorted by
        cluster ID. ``Nat`` is the atom-count column of the file, ``sites`` the
        total number of sites from the header (or the argument).
    """

    # --- Check if Ej is missing in clusters.dat --- #
    ids_and_energies = np.atleast_2d(np.genfromtxt(datafile, dtype=str, usecols=(0, 5), comments="#"))
    missing_ids = [cid for cid, e in ids_and_energies if e == "missing"]
    if missing_ids and not dft_out_file:
        if len(missing_ids) == len(ids_and_energies):
            raise ValueError("The energies in clusters.dat are missing, and 'dft_out_file_name' is not specified. " 
                            "Please provide a DFT output file to extract the energies or set them manually.")
        raise ValueError("The 6th column (energies) in 'clusters.dat' contains both, numeric values and 'missing' "
                         f"(missing: {', '.join(missing_ids)}). Either provide all energies manually, or set "
                         "'dft_out_file_name' to extract all of them from the DFT outputs.")
    if missing_ids:
        # some or all energies missing: (re-)extract all of them from the DFT outputs
        extract_total_energies(datafile, self.clusters_folder, dft_out_file, dft_out_file_subfolder, aims_use_zero_broadening_corrected_energy)
        # Write DFT/volume.dat from the same outputs, so the lattice parameters are at
        # hand for a later run at finite pressure. Not needed at zero pressure, so a
        # failure only warns.
        try:
            set_up_volume_dat(self.clusters_folder, dft_out_file, dft_out_file_subfolder,
                              run_folder=Path(datafile).parent)
        except Exception as exc:
            warnings.warn(f"Could not write volume.dat from the DFT output files: {exc}")
    
    # --- read header --- #
    with open(datafile) as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("#"):
            break
        if line.startswith("# WARNING:"):
            print(line[1:].rstrip())
        if "compound=" in line and compound is None:
            compound = line.split("compound=")[1].split()[0]
        if "n=" in line and n is None:
            n = int(line.split("n=")[1].split()[0])
        if "sites=" in line and sites is None:
            sites = int(line.split("sites=")[1].split()[0])

    # --- repeat the warnings on data rows (unfinished DFT run, unconverged relaxation) --- #
    # Skipped right after an extraction, which already warned for every flagged cluster.
    if not missing_ids:
        _warn_row_comments(lines, datafile)

    if sites is None or n is None:
        raise NameError("The number of sites or the permutable sites (n) are not defined. " 
                        "Please specify them either in the header of clusters.dat (sites=/n=) or via the input variables (sites/n).")

    # --- read data from clusters.dat --- #
    # columns (1..5): nj, Nat, xj, g, E_Ry
    # nj ... number of atoms B in cluster j
    # sites ... total number of atoms
    # xj ... composition of cluster j
    # gj ... degeneracy of cluster j
    # Ej_Ry ... Energy of cluster j in Rydberg
    cluster_id = np.genfromtxt(datafile, dtype=str, usecols=0)
    nj, Nat, xj, gj, Ej_Ry = np.genfromtxt(datafile, usecols=(1,2,3,4,5), unpack=True)
   
    # unreadable entries (e.g. 'nan' or a typo) become NaN in genfromtxt
    cluster_id = np.atleast_1d(cluster_id)
    columns = {"nj": nj, "xj": xj, "gj": gj, "Ej": Ej_Ry}
    bad = {name: list(cluster_id[~np.isfinite(np.atleast_1d(col))]) for name, col in columns.items()}
    bad = {name: ids for name, ids in bad.items() if ids}
    if bad:
        raise ValueError(
            f"{datafile} contains entries that are not finite numbers: "
            + "; ".join(f"{name} of {', '.join(ids)}" for name, ids in bad.items())
        )

    # sort according to id, implicitly sorts x = 0 to x = 1
    cluster_id, nj, Nat, xj, gj, Ej_Ry = sort_by_id(cluster_id, nj, Nat, xj, gj, Ej_Ry)

    return cluster_id, nj, Nat, xj, gj, Ej_Ry, sites, n, compound


def read_volume_dat(self, volume_file: str | Path, dft_out_file:str, dft_out_file_subfolder: str,  E_elec: np.ndarray):
    """
    Read the volume.dat file and extract the lattice parameters for each cluster.

    The lattice parameters are stored on the model as ``self.lattice``.

    Parameters
    ----------
    volume_file : str or Path or None
        Path to volume.dat (ID, a, b, c, alpha, beta, gamma). When None,
        ``volume.dat`` next to the clusters file (the DFT run folder) is used, which
        is written together with its energies, and created from the DFT output
        files if it does not exist.
    dft_out_file : str
        Name of the DFT output file used to create volume.dat when needed.
    dft_out_file_subfolder : str
        Subfolder of each cluster folder holding the DFT output file.
    E_elec : numpy.ndarray
        Electronic energies of the clusters, used to check consistency with volume.dat.

    Returns
    -------
    tuple
        ``(Ep, volume_file)`` with the PV term ``Ep`` in meV per site.
    """

    # volume.dat in the DFT run folder (next to the clusters file) is written by
    # read_clusters_dat together with the energies
    if volume_file is None:
        run_folder = Path(self.datafile).parent
        dft_volume_file = run_folder / "volume.dat"
        if dft_volume_file.is_file():
            volume_file = dft_volume_file

    # create volume.dat if non existing
    if volume_file is None:
        volume_file, cluster_id, a, b, c, alpha_deg, beta_deg, gamma_deg = set_up_volume_dat(
            self.clusters_folder, dft_out_file, dft_out_file_subfolder, run_folder=run_folder)
    else:
        # Read lattice parameters from volume.dat (numeric columns only)
        # Columns: ID, a (Angstrom), b (Angstrom), c (Angstrom),
        #          alpha (deg), beta (deg), gamma (deg)
        cluster_id = np.genfromtxt(volume_file, dtype=str, usecols=0)
        a, b, c, alpha_deg, beta_deg, gamma_deg = np.genfromtxt(
            volume_file, usecols=(1, 2, 3, 4, 5, 6), unpack=True
        )

    cluster_id, a, b, c, alpha_deg, beta_deg, gamma_deg = reorder_by_reference_id(
        self.cluster_id,
        cluster_id,
        a,
        b,
        c,
        alpha_deg,
        beta_deg,
        gamma_deg,
    )

    if a.size != E_elec.size:
        raise ValueError(
            "clusters.dat and lattice.dat have different number of entries"
        )
            
    self.lattice = lattice(a, b, c, alpha_deg, beta_deg, gamma_deg)
            

    # Pressure in Pa
    P_Pa = self.pressure_GPa * GPA_TO_PA
    # PV per site (meV/site)
    Ep = P_Pa * self.lattice.V_cell * J_to_meV / float(self.sites)

    return Ep, volume_file


# Relative deviation from 3 modes per site above which read_phonons warns.
PHONON_NORM_TOLERANCE = 0.05


def read_phonons(self, phonon_file_name:str, sites: int):
    """
    Read the phonon files and extract the phonon dos for each cluster.

    Two layouts are supported, selected automatically:

    * **Folder mode** -- if ``phonon_file_name`` points to a directory (resolved
      relative to ``self.clusters_folder`` when not absolute), the DOS of each
      cluster is read from ``<folder>/<cluster_id>.dos``. This matches the layout
      written by :func:`egqca.run_ml_relaxation` into ``<output_dir>/phonons/``.
    * **Per-subfolder mode** -- otherwise ``phonon_file_name`` is treated as a
      file name and read from each ``S*/S*_i*_w*/phonons/<phonon_file_name>``.

    Parameters
    ----------
    phonon_file_name : str
        Directory of ``<cluster_id>.dos`` files, or the DOS file name expected in
        each cluster folder under ``phonons/``.
    sites : int
        Total number of sites of a cluster. Each DOS is rescaled to 3 modes per
        site, including imaginary modes, before those are dropped; a warning
        lists the files whose normalization was off by more than
        ``PHONON_NORM_TOLERANCE``.

    Returns
    -------
    list of numpy.ndarray
        One array per cluster, in the order of ``self.cluster_id``, holding the
        angular frequency in rad/s and the DOS in states per rad/s per site.
    """
    base = Path(self.clusters_folder)

    phonon_folder = Path(phonon_file_name)
    if not phonon_folder.is_absolute():
        phonon_folder = base / phonon_folder
    folder_mode = phonon_folder.is_dir()

    phonon_data = []
    # clusters with imaginary (negative) frequencies: (id, dropped modes/site, min hbar*omega in meV)
    imaginary_modes = []
    # clusters whose DOS was not normalized to 3 modes per site: (id, modes per site in the file)
    misnormalized = []

    # Loop over the clusters of clusters.dat (not the cluster folders), so that
    # phonon_data[k] belongs to cluster_id[k]: the folders are sorted as text
    # (i10 before i2) and may contain clusters that are not in clusters.dat.
    for cluster_id in self.cluster_id:
        if folder_mode:
            phpath = phonon_folder/f"{cluster_id}.dos"
        else:
            phpath = base/cluster_id.split("_")[0]/cluster_id/"phonons"/phonon_file_name
        if not phpath.is_file():
            raise FileNotFoundError(f"No phonon DOS file for cluster {cluster_id}: {phpath} does not exist.")
        with open(phpath, "r") as f:
            # reset for every cluster, so that an unrecognized unit cannot silently
            # inherit the conversion factor of the previously read DOS file
            conversion_factor = None
            for line in f:
                for unit_pattern, factor in _PHONON_UNITS:
                    if re.search(unit_pattern, line):
                        conversion_factor = factor
                        break
                if conversion_factor is not None:
                    break

            if conversion_factor is None:
                raise ValueError(
                    f"Could not determine the frequency unit of the phonon DOS file {phpath}. "
                    "Expected a header line containing 'cm^-1'/'1/cm', 'THz', 'GHz', 'Hz'/'s^-1', 'meV' or 'eV'."
                )

            # The branches above give the ordinary frequency nu (in Hz), but the
            # thermodynamic integrands use hbar*omega. 
            # Convert from frequency to angular frequency
            conversion_factor *= 2.0 * np.pi

            wg = np.genfromtxt(phpath, usecols=(0, 1))
            # Column 0: frequency, convert to angular frequency in rad/s
            wg[:, 0] *= conversion_factor
            # Column 1: DOS, convert to states per (rad/s) per site
            wg[:, 1] /= (conversion_factor * sites)

            # Rescale to 3 modes per site. DOS files differ in their normalization
            # (per cell, per formula unit, ASE <= 3.24 without the q-point weight),
            # and F_vib scales linearly with it. Done before dropping the imaginary
            # modes, so that their weight is not moved to the real ones.
            modes_per_site = float(np.trapezoid(wg[:, 1], wg[:, 0]))
            if not modes_per_site > 0.0:
                raise ValueError(f"The phonon DOS in {phpath} does not integrate to a positive number of modes.")
            if abs(modes_per_site / 3.0 - 1.0) > PHONON_NORM_TOLERANCE:
                misnormalized.append((cluster_id, modes_per_site))
            wg[:, 1] *= 3.0 / modes_per_site

            # Exclude omega <= 0 explicitly to avoid log(0)
            mask_pos = wg[:,0] > 0.0
            if not np.all(mask_pos):
                # imaginary modes are dropped here, so the vibrational free energy of this
                # cluster misses them; record how much weight that is
                dropped_modes = float(np.trapezoid(wg[~mask_pos, 1], wg[~mask_pos, 0]))
                if dropped_modes > 0.0:
                    imaginary_modes.append(
                        (cluster_id, dropped_modes, hbar * float(wg[:, 0].min()))
                    )
            wg = wg[mask_pos,:]

            phonon_data.append(wg)

    if misnormalized:
        shown = ", ".join(f"{cluster_id} ({modes:.4g})" for cluster_id, modes in misnormalized[:10])
        if len(misnormalized) > 10:
            shown += f", ... ({len(misnormalized)} in total)"
        warnings.warn(
            f"{len(misnormalized)} of {len(phonon_data)} phonon DOS files do not integrate to "
            f"3 modes per site (with sites={sites}) and were rescaled: {shown}. "
            "Check that the files belong to these clusters and that 'total number of sites' is right."
        )

    if imaginary_modes:
        worst = min(imaginary_modes, key=lambda entry: entry[2])
        warnings.warn(
            f"{len(imaginary_modes)} of {len(phonon_data)} phonon DOS files contain imaginary "
            "(negative) frequencies, which are excluded from the vibrational free energy. "
            f"Most negative mode: {worst[2]:.2f} meV in {worst[0]}; "
            f"largest excluded weight: {max(e[1] for e in imaginary_modes):.4f} of 3 modes per site. "
            "Check the dynamic stability of these clusters."
        )

    return phonon_data


def _shorten_list(items: list, limit: int = 20) -> str:
    shown = ", ".join(map(str, items[:limit]))
    return shown + (f", ... ({len(items)} in total)" if len(items) > limit else "")


def check_cluster_ids(cluster_id, reference_id, source: str | Path, reference_name: str = "clusters.dat") -> None:
    """Raise a ValueError if ``cluster_id`` holds an ID twice or one not in ``reference_id``.

    ``reference_name`` names the cluster file ``reference_id`` comes from, for the message.
    """
    cluster_id = [str(c) for c in np.atleast_1d(cluster_id)]
    counts = {}
    for c in cluster_id:
        counts[c] = counts.get(c, 0) + 1
    duplicates = [f"{c} ({n}x)" for c, n in counts.items() if n > 1]
    if duplicates:
        raise ValueError(f"{source} lists these cluster IDs more than once: {_shorten_list(duplicates)}")
    known = {str(c) for c in np.atleast_1d(reference_id)}
    unknown = [c for c in cluster_id if c not in known]
    if unknown:
        raise ValueError(
            f"{len(unknown)} cluster IDs in {source} are not in {reference_name}: {_shorten_list(unknown)}"
        )


def read_properties(self, properties_file: str | Path, ncols:int | None = None):
    """
    Read the properties.dat file and extract the relevant information for the EGQCA calculation.

    Parameters
    ----------
    properties_file : str or Path
        Path to the properties.dat file.
    ncols : int or None
        Number of columns to read. Taken from the first data row when None.

    Returns
    -------
    tuple
        ``(Prop_j, property names, cluster_id)``. Rows are reordered to the cluster
        order of the model when the file contains every cluster, otherwise the file
        order is kept.
    """
    # --- read data from properties.dat ---
    # Column 0: ID
    # Columns (1..n): any quantity of interest for each cluster, e.g. N(ef)
    # The order of the clusters should be the same as in clusters.dat

    # read header
    header_lines = 0
    header = None

    with open(properties_file) as f:
        for line in f:
            if re.match(r"#\s*ID", line):
                header = line[1:].split()
            if line.startswith("#"):
                header_lines += 1
            else:
                if ncols is None:
                    ncols = len(line.split())
                else:
                    # neglect first column id
                    ncols += 1
                break

    # If no header
    if header is None:
        header = ["ID"] + [f"Property_{i}" for i in range(1, ncols)]
        
    cluster_id = np.genfromtxt(properties_file, usecols=0, skip_header=header_lines, dtype=str)
    Prop_j = np.genfromtxt(properties_file, usecols=range(1, ncols), skip_header=header_lines, dtype=float, ndmin=2)

    # header names (split on whitespace) and data columns must agree
    if len(header) - 1 != Prop_j.shape[1]:
        raise ValueError(
            f"{properties_file}: the header names {len(header) - 1} properties ({', '.join(header[1:])}), "
            f"but the data has {Prop_j.shape[1]} property columns. Header names are split on "
            "whitespace: combine a name of several words, e.g. 'Band-gap' instead of 'Band gap'."
        )
    # a missing value would turn every average of its column into NaN
    bad = ~np.isfinite(Prop_j)
    if bad.any():
        ids = np.atleast_1d(cluster_id)
        details = "; ".join(
            f"{header[1 + col]}: {', '.join(ids[bad[:, col]])}" for col in np.where(bad.any(axis=0))[0]
        )
        raise ValueError(f"{properties_file} contains non-numeric or NaN values ({details}).")
    datafile = getattr(self, "datafile", None)
    check_cluster_ids(cluster_id, self.cluster_id, properties_file,
                      Path(datafile).name if datafile is not None else "clusters.dat")

    if len(np.atleast_1d(cluster_id)) == len(self.cluster_id):
        cluster_id, Prop_j = reorder_by_reference_id(self.cluster_id, cluster_id, Prop_j)

    return Prop_j, header[1:], cluster_id
