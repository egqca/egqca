import re
import shutil
import warnings
import numpy as np
import scipy.optimize as spo
import scipy.spatial as spatial
import scipy.integrate as spi
import scipy.interpolate as spinterp
from scipy.special import xlogy
from matplotlib import pyplot as plt
import matplotlib as mpl
from labellines import labelLine, labelLines
from .colors import Palette
from pathlib import Path
# import files
from .constants import Ry_to_mev, kB, hbar, angstrom_to_meter
from .io import read_clusters_dat, read_volume_dat, read_phonons
from . import properties
from ._banner import entry_point, describe_grid, header_lines, __version__


def _latex_available() -> bool:
    """Whether matplotlib can render text with LaTeX (``text.usetex``).

    usetex needs ``latex`` for the layout and ``dvipng`` for raster output; without
    them every figure fails with "latex could not be found" (e.g. on CI runners).
    """
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None



_NUMBER_BOX = dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none")


def cluster_number_markers(ax, x, y, color, fontsize: str | float = "small",
                           background: bool = False, zorder: float = 3) -> None:
    """Draw the numbers 1, 2, 3, ... (clusters.dat order) as markers at (x[i], y[i]).

    ``background`` puts a small white box behind each number, e.g. on top of a line.
    The numbers are explained by ``cluster_numbers.dat``.
    """
    for number, (xi, yi) in enumerate(zip(np.ravel(x), np.ravel(y)), start=1):
        ax.text(xi, yi, str(number), color=color, fontsize=fontsize, ha="center", va="center",
                bbox=_NUMBER_BOX if background else None, zorder=zorder, clip_on=True)


def label_cluster_curves(ax, lines, x_positions, fontsize: str | float = "x-small") -> None:
    """Write the numbers 1, 2, 3, ... on the curves ``lines`` (one per cluster) with labellines.

    Each number has the color of its curve, sits on the curve at ``x_positions[i]`` and has a
    small white background. Positions outside a curve's data range are moved inside it.
    """
    for number, (line, x_pos) in enumerate(zip(lines, np.ravel(x_positions)), start=1):
        xdata = np.asarray(line.get_xdata(), dtype=float)
        if xdata.size < 2:
            cluster_number_markers(ax, xdata, line.get_ydata(), line.get_color(), fontsize, background=True)
            continue
        x_pos = float(np.clip(x_pos, xdata.min(), xdata.max()))
        labelLine(line, x_pos, label=str(number), align=False, outline_color=None,
                  fontsize=fontsize, bbox=_NUMBER_BOX, zorder=3)


class EGQCA:
    """
    Generalized quasichemical approximation (GQCA) for a binary alloy, with optional
    vibrational contributions from phonon DOS and optional PV (enthalpy) correction
    as described in https://doi.org/10.1016/j.mtphys.2024.101547

    All energies are mixing energies in meV per site, where "site" is one of the
    ``sites`` sites of a cluster (not one of the ``permutable_sites``).
    If volume_file and pressure_GPa are given, we add a PV term
    (enthalpy per site) based on the cell volume from volume.dat.

    Public usage:
      m = EGQCA(
          datafile="clusters.dat",            # sites, permutable sites and compound
          phonon_file_name="phonons",         # are read from its header
          volume_file="volume.dat", pressure_GPa=0.0,
      )
      x = np.arange(0.0, 1.0, 1e-3)
      Tlist = [600, 800, 1000, 1400]
      m.thermodynamic_assessment(x, Tlist)     # p_j(x,T), Delta H/G/S, miscibility gap
      m.average_properties("volume.dat")       # any per-cluster property, averaged
      m.cluster_importance(x, Tlist)           # smallest sufficient cluster set
    """

    def __init__(self, datafile: str | Path, out_folder_name: str | None = None,
                 dft_out_file_name:str | None = None, dft_out_file_subfolder: str = "vc-relax",  
                 permutable_sites: int | None = None, sites: int | None = None,    
                 compound: str | None = None,             
                 volume_file: str | None = None, pressure_GPa: float=0.0, 
                 phonon_file_name: str | None=None,                         
                 aims_use_zero_broadening_corrected_energy: bool = False,
                 entropy_approximation: str = "overlapping"):
        """
        Parameters
        ----------
        datafile : str
            Path to clusters.dat (columns: ID, n_j, Nat, x_j, g_j, E_j in Ry).
        permutable_sites : int
            Number of permutable sites in the cluster. Read from the clusters.dat
            header (n=) when None.
        sites : int
            Total number of sites in the cluster; all energies are reported per
            such site. Read from the clusters.dat header (sites=) when None.
        compound: str
            Name of the compound in the format A$_{1-x}$B$_x$ for plotting
        volume_file : str or None
            Path to volume.dat with the lattice parameters of every cluster
            (ID, a, b, c, alpha, beta, gamma). Only used if pressure_GPa != 0.
        pressure_GPa : float
            External pressure in GPa. If 0.0, no PV term.
        phonon_file_name: str | None
            Either a directory of ``<cluster_id>.dos`` files (resolved relative to
            the clusters folder when not absolute, e.g. an ML run's
            ``<output_dir>/phonons``), or the name of the phonon DOS file expected
            in each cluster folder under ``phonons/``.
        aims_use_zero_broadening_corrected_energy: bool
            Whether to use the zero-broadening corrected energy from FHI-aims output or the total energy.
        entropy_approximation: str
            How the clusters tile the crystal, which sets the weight of the
            Kullback-Leibler term in the entropy and, consistently, the
            Boltzmann factor of the cluster probabilities:

            - ``"overlapping"``: S/kB = -(n/sites) [x ln x + (1-x) ln(1-x) + D],
              p_j ~ p0_j exp(-E_j / (n kB T)).
            - ``"non-overlapping"``: S/kB = -(n/sites) [x ln x + (1-x) ln(1-x)] - D/sites,
              p_j ~ p0_j exp(-E_j / (kB T)).

            E_j is the mixing energy of the whole cluster (delta_j * sites).

        """

        # Plotting style
        self.style = {
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 22,
            # fall back to mathtext when no LaTeX is installed
            "text.usetex": _latex_available(),
            "axes.linewidth": 3.5,
            "xtick.major.width": 3.5,
            "ytick.major.width": 3.5,
        }

        # Parameters
        self.pressure_GPa = float(pressure_GPa)
        self.pj = {}
        # vibrational free energies per cluster and T: probability inputs, kept across
        # init_thermo (they depend only on T and the phonon data)
        self.Fvib_j = {}
        self.deltaFvib_j = {}
        self.grid_compare_rtol = 1e-8
        self.grid_compare_atol = 1e-10
        self.current_output_dir = None
        self.thermodynamic_assessment_dir = None
        self.averaging_dir = None
        self.cluster_reduction_dir = None
        self.entropy_approximation = entropy_approximation

        self.set_up(
            datafile=datafile,
            out_folder_name=out_folder_name,
            dft_out_file_name=dft_out_file_name,
            dft_out_file_subfolder=dft_out_file_subfolder,
            permutable_sites=permutable_sites,
            sites=sites,
            compound=compound,
            volume_file=volume_file,
            phonon_file_name=phonon_file_name,
            aims_use_zero_broadening_corrected_energy=aims_use_zero_broadening_corrected_energy
        )

    ENTROPY_APPROXIMATIONS = ("overlapping", "non-overlapping")

    @property
    def entropy_approximation(self) -> str:
        """``"overlapping"`` or ``"non-overlapping"``, see the class docstring."""
        return self._entropy_approximation

    @entropy_approximation.setter
    def entropy_approximation(self, mode: str):
        mode = str(mode).lower().replace("_", "-")
        if mode not in self.ENTROPY_APPROXIMATIONS:
            raise ValueError(f"entropy_approximation must be one of {self.ENTROPY_APPROXIMATIONS}, got {mode!r}")
        if getattr(self, "_entropy_approximation", mode) != mode:
            # probabilities, thermodynamics and the gap depend on the mode
            self._reset_results()
        self._entropy_approximation = mode

    @property
    def D_weight(self) -> float:
        """Weight w of the KL divergence in the entropy, S/kB = ... - (w/sites) D.

        w = n for overlapping clusters, w = 1 for non-overlapping clusters. The
        Boltzmann factor that minimizes F with this entropy is
        exp(-beta * delta_j * sites / w).
        """
        return float(self.n) if self.entropy_approximation == "overlapping" else 1.0

    def set_up(
        self,
        datafile: str | Path,
        out_folder_name: str | None = None,
        dft_out_file_name: str | None = None,
        dft_out_file_subfolder: str = "vc-relax",
        permutable_sites: int | None = None,
        sites: int | None = None,
        compound: str | None = None,
        volume_file: str | None = None,
        phonon_file_name: str | None = None,
        aims_use_zero_broadening_corrected_energy: bool = False,
    ):
        """Read the EGQCA input files and initialize the model state."""

        self.datafile = Path(datafile)
        self.clusters_folder = self.datafile.parent.parent  # folder containing S* cluster dirs
        if out_folder_name is None:
            self.out_dir = self.datafile.parent
        else:
            self.out_dir = self.datafile.parent / out_folder_name
        self.dft_out_file_name = dft_out_file_name
        self.dft_out_file_subfolder = dft_out_file_subfolder

        # ------------------- Clusters -------------------- #
        cluster_id, nj, Nat, xj, gj, Ej_Ry, sites, permutable_sites, compound = read_clusters_dat(
            self,
            self.datafile,
            dft_out_file_name,
            sites,
            permutable_sites,
            compound,
            dft_out_file_subfolder,
            aims_use_zero_broadening_corrected_energy,
        )

        Nat = np.asarray(Nat, dtype=int)
        self.nj = np.asarray(nj, dtype=int)
        self.xj = np.asarray(xj, dtype=float)
        self.gj = np.asarray(gj, dtype=float)
        self.n = permutable_sites
        self.cluster_id = cluster_id
        self.sites = sites
        self.compound = compound

        # meV / total sites
        E_elec = Ej_Ry * Ry_to_mev / float(self.sites)

        # ------------------- Volume -------------------- #
        if volume_file is None and dft_out_file_name is None and self.pressure_GPa != 0.0:
            warnings.warn(
                f"pressure_GPa={self.pressure_GPa:g} was given, but including the pressure (PV term) "
                "requires cell volumes (volume_file or dft_out_file_name). No volume was given, "
                "so the pressure is set to 0 GPa.",
                stacklevel=3,
            )
            self.pressure_GPa = 0.0

        if (volume_file is not None or dft_out_file_name is not None) and (self.pressure_GPa != 0.0):
            Ep, volume_file = read_volume_dat(
                self,
                volume_file,
                dft_out_file_name,
                dft_out_file_subfolder,
                E_elec,
            )
            E_elec += Ep

        self.volume_file = volume_file
        self.Hj = E_elec

        # ------------------- Phonons -------------------- #
        self.phonon_file_name = phonon_file_name
        if phonon_file_name is not None:
            self.phonon_data = read_phonons(self, phonon_file_name, sites)
        else:
            self.phonon_data = None

        # new data: no earlier result is valid any more
        self._reset_results(keep_vibrational=False)



    
    ########################################################
    # ------------------ Run overview -------------------- #
    ########################################################
    OVERVIEW_FILE = "egqca_overview.txt"

    def _run_summary(self, task: str, bound) -> list[tuple[str, object]]:
        """Most important inputs of a run, printed at its start and appended to the overview file."""
        args = bound.arguments
        x, Tlist = args.get("x"), args.get("Tlist")
        energies = "clusters.dat"
        if self.dft_out_file_name is not None:
            energies = f"S*/S*_i*_w*/{self.dft_out_file_subfolder}/{self.dft_out_file_name}"
        items = [
            ("Cluster file", Path(self.datafile).resolve()),
            ("Output folder", Path(self.out_dir).resolve()),
            ("Compound", self.compound),
            ("Sites per cluster", self.sites),
            ("Permutable sites n", self.n),
            ("Number of clusters", len(self.cluster_id)),
            ("Energies from", energies),
            ("Entropy approximation", self.entropy_approximation),
            ("Pressure (GPa)", self.pressure_GPa),
            ("Volume file", None if self.volume_file is None else
             f"{Path(self.volume_file).resolve()}" + ("  (not used at 0 GPa)" if self.pressure_GPa == 0.0 else "")),
            ("Phonon files", self.phonon_file_name if self.phonon_file_name is not None else "none (no vibrational free energy)"),
            ("Composition x", describe_grid(self._sanitize_x_grid(x)) if x is not None else "from previous run"),
            ("Temperatures", describe_grid(Tlist, "K") if Tlist is not None else "from previous run"),
        ]
        if task == "Thermodynamic assessment":
            gap = "no"
            if args["run_miscibility_gap"]:
                gap = f"yes (T_min = {args['T_min_misc_gap']:g} K, max_iter = {args['misc_gap_max_iter']}, dx = {args['dx_plot']:g})"
            items.append(("Miscibility gap", gap))
        elif task == "Property averaging":
            files = args["properties_file"]
            items.append(("Properties file(s)", files if isinstance(files, (str, Path)) else ", ".join(map(str, files))))
            items.append(("Surrogate model(s)", args["surrogate_model"]))
        elif task == "Cluster importance":
            items.append(("Top percentages", args["top_percentage"]))
        self._write_overview(task, items)
        return items

    # output subfolder of each entry point, see _prepare_output_dir
    TASK_SUBFOLDERS = {
        "Thermodynamic assessment": "thermodynamic_assessment",
        "Property averaging": "results",
        "Cluster importance": "cluster_reduction",
    }

    def _write_overview(self, task: str, items: list[tuple[str, object]]):
        """Write the summary of this run to ``egqca_overview.txt`` in the task's output folder.

        The file is overwritten, like the results next to it, so it always describes the last run.
        """
        from ._banner import format_summary
        path = self._output_dir_path(self.TASK_SUBFOLDERS[task]) / self.OVERVIEW_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write("\n".join(header_lines()) + "\n\n")
                f.write("\n".join(format_summary(f"{task}  [egqca {__version__}]", items)) + "\n\n")
        except OSError as err:
            warnings.warn(f"Could not write the run overview to {path}: {err}", stacklevel=3)

    ########################################################
    # ------------------ Main functions ------------------ #
    ########################################################
    @entry_point("Thermodynamic assessment", lambda b: b.arguments["self"]._run_summary("Thermodynamic assessment", b))
    def thermodynamic_assessment(self, x:np.ndarray, Tlist:list[float], run_miscibility_gap:bool = True, T_min_misc_gap:float=30, misc_gap_max_iter:int=200, dx_plot:float= 0.01, plot_flag:bool = True, figsize:tuple=(6,6), enumerate_clusters:bool = False):
        """Run the thermodynamic assessment workflow and save outputs in ``thermodynamic_assessment``.

        ``enumerate_clusters`` writes the cluster numbers (1, 2, ... in clusters.dat order)
        into the excess energy and p_j plots, and the key ``cluster_numbers.dat``.
        """

        if not self._grid_matches_cached(x, Tlist):
            self.pj = {}
        x = self._prepare_main(x, Tlist, output_subfolder="thermodynamic_assessment")

        # plot excess energies
        self.plot_excess_enthalpy(figsize=figsize, enumerate_clusters=enumerate_clusters)

        self.init_thermo()
        self._ensure_probabilities(x, Tlist)
        for T in Tlist:
            print(f'Calculating the mixing free energy at T = {T}...')
            self.thermo(T)
            print('Done!')

        if run_miscibility_gap:
            self.calc_miscibility_gap(x, T_min=T_min_misc_gap, dx_plot=dx_plot, max_iter=misc_gap_max_iter)

        print("Saving thermodynamic assessment...")
        self.plot_thermo(addLineLabel=True, figsize=figsize)
        if run_miscibility_gap:
            self.plot_miscibility_gap(figsize=figsize)
        if plot_flag and len(Tlist) > 1:
            self.plot_probabilities_vs_T(x[np.unique([len(x)//4, len(x)//2, 3*len(x)//4])], figsize=figsize, enumerate_clusters=enumerate_clusters)
            self.plot_KL_divergence_vs_T(x[np.unique([len(x)//4, len(x)//2, 3*len(x)//4])], figsize=figsize)
        if self.volume_file is not None and (self.pressure_GPa != 0.0):    # plot flag?
            self.plot_lat_pars(self.x_plot, Tlist, figsize=figsize)


    @entry_point("Property averaging", lambda b: b.arguments["self"]._run_summary("Property averaging", b))
    def average_properties(
        self,
        properties_file: str | Path | list[str | Path] | tuple[str | Path, ...],
        Tlist:list[float] | None = None,
        x:np.ndarray | None = None,
        idx_clusters_surrogate: dict[float, dict[float | str, list[int]]] | None = None,
        surrogate_model: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        ylabels: list[str] | None = None,
        columns: list[int] | list[list[int]] | None = None,
        legend_labels: list[str] | None = None,
        figsize: tuple = (6, 6),
    ):
        """Average properties and save outputs.

        ``properties_file`` may be a single file or a list of files; when several are
        given their curves for the same property are overlaid in one figure. ``columns``
        optionally selects a subset of property columns by index (excluding the ID
        column), either as a single list applied to every file or one list per file.

        If no new ``x`` or ``Tlist`` is provided, cached occurrence probabilities
        from a previous thermodynamic assessment are reused. If a different grid
        is requested, the probabilities are recalculated before averaging.
        """

        if isinstance(properties_file, (str, Path)):
            properties_files = [Path(properties_file)]
        else:
            properties_files = [Path(pf) for pf in properties_file]
        if not properties_files:
            raise ValueError("At least one properties file must be provided.")

        cached_x = getattr(self, "x", None)
        cached_Tlist = getattr(self, "Tlist", None)
        cached_pj = getattr(self, "pj", None)
        has_cached_probabilities = bool(cached_pj)

        if x is None and Tlist is None:
            if not has_cached_probabilities or cached_x is None or cached_Tlist is None:
                raise ValueError(
                    "No cached probabilities available. Run thermodynamic_assessment first or provide both x and Tlist."
                )
            x = np.asarray(cached_x, dtype=float)
            Tlist = list(np.asarray(cached_Tlist, dtype=float))
            reuse_probabilities = True
        else:
            if x is None:
                if cached_x is None:
                    raise ValueError(
                        "No cached composition grid available. Provide x or run thermodynamic_assessment first."
                    )
                x = np.asarray(cached_x, dtype=float)
            if Tlist is None:
                if cached_Tlist is None:
                    raise ValueError(
                        "No cached temperature grid available. Provide Tlist or run thermodynamic_assessment first."
                    )
                Tlist = list(np.asarray(cached_Tlist, dtype=float))

            x = self._sanitize_x_grid(x)
            Tlist = list(self._sanitize_temperature_grid(Tlist))
            reuse_probabilities = has_cached_probabilities and self._grid_matches_cached(x, Tlist)

        if surrogate_model is not None and idx_clusters_surrogate is not None:
            raise ValueError("Provide either surrogate_model or idx_clusters_surrogate, not both.")

        if surrogate_model is not None:
            if isinstance(surrogate_model, (str, Path)):
                surrogate_models = [Path(surrogate_model)]
            else:
                surrogate_models = [Path(model_path) for model_path in surrogate_model]
            if not surrogate_models:
                raise ValueError("At least one surrogate model path must be provided.")

            surrogate_indices_by_name = {
                surrogate_path.stem: self._read_surrogate_model_indices(surrogate_path)
                for surrogate_path in surrogate_models
            }
            idx_clusters_surrogate = {
                T: surrogate_indices_by_name
                for T in Tlist
            }

        x = self._prepare_main(x, Tlist, output_subfolder="results")
        if not reuse_probabilities:
            self.pj = {}
            self._ensure_probabilities(x, Tlist)
        return properties.calculate_averages(self, properties_file=properties_files, Tlist=Tlist, x_grid=self.x_plot, idx_clusters_surrogate=idx_clusters_surrogate, ylabels=ylabels, columns=columns, legend_labels=legend_labels, figsize=figsize)



    @entry_point("Cluster importance", lambda b: b.arguments["self"]._run_summary("Cluster importance", b))
    def cluster_importance(self, x:np.ndarray, Tlist:list[float], top_percentage:list[float]=[0.02, 0.05, 0.1], plot_flag:bool = True, figsize:tuple=(6,6), enumerate_clusters:bool = False):
        """ 
            Find a subset with top_percentage% of cluster which are sufficient to describe the properties of the solid solution.

            ``enumerate_clusters`` writes the cluster numbers into the p_j plots, see
            :meth:`thermodynamic_assessment`.
        """
        
        x = self._prepare_main(x, Tlist, output_subfolder="cluster_reduction")

        # probabilities of earlier calls on the same x grid are reused
        self._ensure_probabilities(x, Tlist)
        
        # --- find most important clusters --- #
        self.rank_clusters(x, Tlist, top_percentage)

        # ---------------- Plot -------------- #
        # plot cluster occurence probabilities
        if plot_flag:
            self.plot_probabilities_vs_x(Tlist, figsize=figsize, enumerate_clusters=enumerate_clusters)

        return self.S_clusters


    #########################################################
    # ---------------------- Helpers ---------------------- #
    #########################################################
    def _output_dir_path(self, subfolder: str) -> Path:
        """Output folder of a task; ``_phonons`` is appended when phonon files are used."""
        folder_name = f"{subfolder}_phonons" if self.phonon_file_name is not None else subfolder
        return Path(self.out_dir) / folder_name

    def _prepare_output_dir(self, subfolder: str | None):
        if subfolder is None:
            self.current_output_dir = None
            return None
        output_dir = self._output_dir_path(subfolder)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_output_dir = output_dir
        if subfolder == "thermodynamic_assessment":
            self.thermodynamic_assessment_dir = output_dir
        elif subfolder == "results":
            self.averaging_dir = output_dir
        elif subfolder == "cluster_reduction":
            self.cluster_reduction_dir = output_dir
        return output_dir

    def _add_phonon_suffix(self, filename: str) -> str:
        if self.phonon_file_name is None:
            return filename
        path = Path(filename)
        return str(path.with_stem(path.stem + "_phonons"))

    def _save_dat_file(self, filename: str, data: np.ndarray, header: str = ""):
        if self.current_output_dir is None:
            return
        path = self.current_output_dir / self._add_phonon_suffix(filename)
        np.savetxt(path, np.asarray(data, dtype=float), header=header, fmt="%.8e")

    def _matching_dat_filename(self, savename: str) -> str:
        path = Path(savename)
        return str(path.with_suffix(".dat"))

    def _prepare_main(self, x:np.ndarray, Tlist:list[float], output_subfolder: str | None = None):
        """Prepare composition grid, temperature list, output folder, and common derived quantities."""
        Tlist = self._sanitize_temperature_grid(Tlist)
        self.Tlist = Tlist
  
        x = self._sanitize_x_grid(x)
        if not self._arrays_are_similar(x, getattr(self, "x", None)):
            # results on the old composition grid are not valid on the new one
            self._reset_results()
        self.x_plot = x*self.extract_composition_number(self.compound) 
        self.x = x

        # excess energy and hull in meV
        self.delta_j = self.excess_energy()
        data_hull = np.vstack((self.xj, self.delta_j)).T
        self.data_hull = data_hull[self.delta_j<=0]         # only negative values

        self._prepare_output_dir(output_subfolder)

        return x

    def _read_surrogate_model_indices(self, surrogate_model: str | Path) -> list[int]:
        """Read cluster IDs from a surrogate-model file and map them to cluster indices."""
        surrogate_model = Path(surrogate_model)
        selected_cluster_ids = []
        with open(surrogate_model) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                selected_cluster_ids.append(stripped.split()[0])

        if not selected_cluster_ids:
            raise ValueError(f"No cluster IDs found in surrogate model file {surrogate_model}")

        duplicates = sorted({c for c in selected_cluster_ids if selected_cluster_ids.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"The surrogate model {surrogate_model} lists these cluster IDs more than once: "
                + ", ".join(duplicates)
            )

        cluster_index_by_id = {cluster_id: idx for idx, cluster_id in enumerate(self.cluster_id)}
        missing_cluster_ids = [cluster_id for cluster_id in selected_cluster_ids if cluster_id not in cluster_index_by_id]
        if len(missing_cluster_ids) == len(selected_cluster_ids):
            # not a single ID matched: the file most likely is not a surrogate model file at all,
            # e.g. occurence_probabilities_T<T>.dat, whose first column is the composition grid
            raise ValueError(
                f"{surrogate_model} does not look like a surrogate model file: none of its "
                f"{len(selected_cluster_ids)} first-column entries are cluster IDs from clusters.dat "
                f"(first entries: {', '.join(selected_cluster_ids[:5])}). "
                "Surrogate models are the 'surrogate_T<T>_<N>clusters.dat' files written by "
                "cluster_importance(); other .dat files in the same folder list the composition "
                "grid in their first column."
            )
        if missing_cluster_ids:
            shown_cluster_ids = ", ".join(missing_cluster_ids[:20])
            if len(missing_cluster_ids) > 20:
                shown_cluster_ids += f", ... ({len(missing_cluster_ids)} in total)"
            raise ValueError(
                f"The following cluster IDs from the surrogate model {surrogate_model} were not "
                f"found in clusters.dat: {shown_cluster_ids}"
            )

        return [cluster_index_by_id[cluster_id] for cluster_id in selected_cluster_ids]


    def _ensure_probabilities(self, x: np.ndarray, Tlist: list[float]):
        """ Calculate pj's if non-existing for given grid """
        if not hasattr(self, "pj") or self.pj is None:
            self.pj = {}
        if not hasattr(self, "deltaFvib_j") or self.deltaFvib_j is None:
            self.deltaFvib_j = {}
        self.apriori(x)
        for T in np.sort(Tlist):
            if T not in self.pj or T not in self.deltaFvib_j:
                print(f"Calculating the occurrence probabilities at T = {T}...")
                self.equilibrate(T, x)

    def write_cluster_numbers(self, folder: str | Path | None = None) -> Path:
        """Write ``cluster_numbers.dat``: the number shown in plots with ``enumerate_clusters``,
        the cluster ID and its x, into ``folder`` (default: the current output folder)."""
        folder = Path(folder) if folder is not None else self.current_output_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "cluster_numbers.dat"
        xj = self.xj * self.extract_composition_number(self.compound)
        with open(path, "w") as f:
            f.write("# number\tcluster_ID\tx\n")
            for number, (cluster_id, x) in enumerate(zip(self.cluster_id, xj), start=1):
                f.write(f"{number}\t{cluster_id}\t{x:.6g}\n")
        return path

    def _reset_results(self, keep_vibrational: bool = True):
        """Drop the results that depend on the data, the x grid or the entropy approximation.

        Clears the probabilities, F/U/S, the miscibility gap and the cluster ranking.
        The vibrational free energies per cluster (``Fvib_j``, ``deltaFvib_j``) depend
        only on T and the data; they are kept unless ``keep_vibrational`` is False.
        """
        self.pj = {}
        self.init_thermo()
        self.binodal_line = np.array([])
        self.spinodal_line = np.array([])
        self.T_c_miscibility_gap = None
        self.x_c_miscibility_gap = None
        self.S_clusters = {}
        if not keep_vibrational:
            self.Fvib_j = {}
            self.deltaFvib_j = {}

    def _ensure_thermo_storage(self):
        if not hasattr(self, "F"):
            self.init_thermo()

    def _sanitize_x_grid(self, x: np.ndarray | list[float]) -> np.ndarray:
        x_array = np.sort(np.asarray(x, dtype=float))
        return x_array[(x_array > 0) & (x_array < 1)]

    def _sanitize_temperature_grid(self, Tlist: list[float] | np.ndarray) -> np.ndarray:
        return np.sort(np.asarray(Tlist, dtype=float))

    def _arrays_are_similar(self, values: np.ndarray | None, cached_values: np.ndarray | None) -> bool:
        if values is None or cached_values is None:
            return False
        values = np.asarray(values, dtype=float)
        cached_values = np.asarray(cached_values, dtype=float)
        if values.shape != cached_values.shape:
            return False
        return np.allclose(
            values,
            cached_values,
            rtol=self.grid_compare_rtol,
            atol=self.grid_compare_atol,
        )

    def _grid_matches_cached(self, x: np.ndarray | list[float], Tlist: list[float] | np.ndarray) -> bool:
        cached_x = getattr(self, "x", None)
        cached_Tlist = getattr(self, "Tlist", None)
        if cached_x is None or cached_Tlist is None:
            return False
        x_array = self._sanitize_x_grid(x)
        Tlist_array = self._sanitize_temperature_grid(Tlist)
        return self._arrays_are_similar(x_array, cached_x) and self._arrays_are_similar(Tlist_array, cached_Tlist)

    def extract_composition_number(self, s):
        if s == None:
            return 1
        match = re.search(r'(\d+)-x', s)
        return int(match.group(1)) if match else 1

    def rank_clusters(self, x:np.ndarray, Tlist:list[float], top_percentage:list[float]):
        """ 
            Consider only top_percentage of clusters with highest probability at each composition xj.
        """ 

        save_dir = self.cluster_reduction_dir or self.out_dir/"cluster_reduction"
        save_dir.mkdir(parents=True, exist_ok=True)


        # ------------------ max coverage ------------------ #
        self.S_clusters = {}
        S_clusters_idx = {}
        for T in Tlist:
            print("Calculating the most important clusters at T="+str(T)+" K.")
            S_clusters_idx[T] = {}
            probjs = self.pj[T].copy()
            probjs = probjs.T

            # --- 1) max coverage: greedy algorithm wrt total probability weight --- #
            prob_weight = np.zeros(len(self.cluster_id))
            for idx, probj in enumerate(probjs):
                prob_weight[idx] = np.trapezoid(probj, x)

            # --------- Subset of clusters per composition xj --------- #
            S_clusters = {}  
            #norm_probs = []
            for percentage in top_percentage:
                S_clusters_idx[T][percentage] = []
                S_prob_weight = []
                for i in np.unique(self.nj):
                    idx_nj = np.where(self.nj == i)[0]
                    probs = prob_weight[idx_nj]
                    n_top = int(len(probs) * percentage)+1
                    idx_relevant = np.argsort(probs)[::-1] 
                    idx_relevant = idx_relevant[:n_top]
                    S_clusters_idx[T][percentage].extend(idx_nj[idx_relevant].tolist())
                    S_prob_weight.extend(probs[idx_relevant].tolist())
                    
                S_clusters[percentage] = self.cluster_id[S_clusters_idx[T][percentage]]

                # save important clusters to file
                with open(save_dir/self._add_phonon_suffix(f"surrogate_T{T}_{len(S_clusters[percentage])}clusters.dat"), "w") as f:
                    f.write(f"# Selected clusters: {len(S_clusters[percentage])}/{len(self.cluster_id)}\n")
                    f.write("# Cluster ID\t Probability weight\n")
                    for cluster, prob in zip(S_clusters[percentage], S_prob_weight):
                        f.write(f"{cluster}\t {prob}\n")
                
            self.S_clusters[T] = S_clusters  
       
        return S_clusters_idx   #, norm_probs.T
   
    

    ######################################################################
    # ------------------------ Thermodynamics -------------------------- #
    ######################################################################          
    def init_thermo(self):
        """Initialize thermodynamic quantities, e.g. F(x,T)"""
        # configurational part
        self.Uc = dict()
        self.Fc = dict()
        self.Sc = dict()
        # vibrational part
        self.Uv = dict()
        # Fvib_j / deltaFvib_j (per cluster and T) are inputs of the probabilities, not
        # results: they depend only on T and the phonon data and are kept, so that
        # _ensure_probabilities can reuse the cached temperatures
        self.Fv = dict()
        self.Sv = dict()
        # total thermodynamics
        self.U = dict()
        self.F = dict()
        self.S = dict()

    def thermo(self, T):
        """
            Calculates and stores thermodynamic quantities at temperature T:
            - Configurational part from GQCA.
            - Vibrational free energy from phonon DOS.
        """
        # configurational part
        self.Uc[T] = self.Us(self.pj[T])
        self.Sc[T] = self.Ss(self.pj[T])*T 
        self.Fc[T] = (self.Uc[T] - self.Sc[T]) 
        # vibrational part
        self.Fv[T] = self.Fvib(self.deltaFvib_j[T], self.pj[T])
        self.Uv[T] = self.Uvib(T, self.pj[T]) 
        self.Sv[T] = self.Svib(T, self.pj[T])
        # total thermodynamics
        self.U[T] = self.Uc[T] + self.Uv[T]
        self.S[T] = self.Sc[T] + self.Sv[T]
        self.F[T] = self.Fc[T] + self.Fv[T]



    def calc_F(self, T, pj, deltaFvib_j, x=None, pj0=None):
        """
            Calculates free energy F(x,T) at temperature T for the miscibility gaps:
            - Configurational part from GQCA.
            - Vibrational free energy from phonon DOS.

            Nothing is stored on the model: ``x`` and ``pj0`` describe the grid the
            probabilities ``pj`` were computed on and default to the cached one.
        """
        # configurational part
        Uc = self.Us(pj)
        Sc = self.Ss(pj, x=x, pj0=pj0)*T
        Fc = (Uc - Sc)
        # vibrational part
        Fv = self.Fvib(self._check_deltaFvib_j(deltaFvib_j), pj)
        # total free energy
        return (Fc + Fv)

    def excess_energy(self):
        """0 K mixing energy (or enthalpy) per site in meV. WARNING: 0 K only."""
        excess_energy = (
            self.Hj - (1.0 - self.xj) * self.Hj[0] - self.xj * self.Hj[-1]
        )
        return excess_energy
    
    def calc_pj0(self, x:np.ndarray):
        '''Return the apriori cluster probabilities p0(x) without storing them.

        p0 is the ideal (random) solid solution on the n permutable sites. For a single
        permutable sublattice sum(p0) = 1, because the degeneracies g_j add up to the
        binomial coefficients.
        '''
        return self.gj * (1.0 - x[:,np.newaxis]) ** (self.n - self.nj) * x[:,np.newaxis]**self.nj

    def apriori(self, x:np.ndarray):
        '''Compute the apriori cluster probabilities p0(x) and save them in self.pj0'''
        self.pj0 = self.calc_pj0(x)
    
    def _check_deltaFvib_j(self, deltaFvib_j) -> np.ndarray:
        """Return ``deltaFvib_j`` as an array with one entry per cluster."""
        deltaFvib_j = np.asarray(deltaFvib_j, dtype=float)
        if deltaFvib_j.shape != (len(self.delta_j),):
            raise ValueError(
                f"deltaFvib_j needs one value per cluster, shape ({len(self.delta_j)},); "
                f"got shape {deltaFvib_j.shape}."
            )
        return deltaFvib_j

    def calc_pj(self, T:float, x:np.ndarray, deltaFvib_j: np.ndarray | None = None):
        """Equilibrium cluster probabilities at T for every x.

        ``deltaFvib_j`` is the vibrational mixing free energy per cluster (one value
        per cluster); None means no vibrational contribution.
        """
        if deltaFvib_j is None:
            delta = self.delta_j
        else:
            delta = self.delta_j + self._check_deltaFvib_j(deltaFvib_j)

        with np.errstate(divide="ignore"):
            log_gj = np.log(self.gj)
        nj = self.nj.astype(float)

        def equilib_scalar(T_scalar, x_scalar):
            # p_j ~ g_j exp(-beta delta_j sites / w) eta^n_j, with eta fixed by
            # sum_j p_j n_j = n x. Solved for t = ln(eta) with log-sum-exp
            # normalization, so large beta*delta (low T, non-overlapping
            # clusters) cannot overflow.
            beta_val = 1.0 / (kB * T_scalar)
            log_w0 = log_gj - beta_val * delta * self.sites / self.D_weight
            n_target = self.n * x_scalar

            def probabilities(t):
                log_w = log_w0 + nj * t
                w = np.exp(log_w - log_w.max())
                return w / w.sum()

            def find_t(t):
                # <n_j>(t) - n x, monotonically increasing in t
                return probabilities(t) @ nj - n_target

            t_lo, t_hi = -1.0, 1.0
            for _ in range(200):
                if find_t(t_lo) < 0.0:
                    break
                t_lo *= 2.0
            for _ in range(200):
                if find_t(t_hi) > 0.0:
                    break
                t_hi *= 2.0
            if not (find_t(t_lo) < 0.0 < find_t(t_hi)):
                raise RuntimeError(
                    f"Failed to bracket ln(eta) for T = {T_scalar}, x = {x_scalar}: "
                    f"[{t_lo}, {t_hi}] -> [{find_t(t_lo)}, {find_t(t_hi)}]"
                )

            t = spo.brentq(find_t, t_lo, t_hi, xtol=1e-14, maxiter=1000)
            return probabilities(t)

        # Vectorize over x
        equilib_vec = np.vectorize(equilib_scalar, signature="(),()->(n)")
        return equilib_vec(T, x)

    def equilibrate(self, T:float, x:np.ndarray):
        """
        Compute equilibrium cluster probabilities pj(x, T) for all compositions x
        and store them in self.pj[T].
        """
        self.deltaFvib_j[T] = self.delta_Fvib_j(T, x)
        self.pj[T] = self.calc_pj(T, x, self.deltaFvib_j[T])

    def probability(self, T:float):
        return self.pj[T]

    def gqca_mean(self, prop, T:float):
        return self.pj[T] @ prop

    def gqca_std(self, prop, T):
        A = self.pj[T] @ prop**2
        B = self.gqca_mean(prop, T) ** 2
        diff = A - B
        # set small negative diff to 0 (happens due to numerical errror when A and B are large but very similar)
        return np.sqrt( np.where(diff<0, 0, diff) )
    
    def Ds(self, pj, pj0=None):
        """Kullback-Leibler divergence D(p || p0) for each x.

        Measured against the ideal solid solution of the n permutable sites, see
        :meth:`calc_pj0`. For cluster sets restricted by a stoichiometric constraint
        p0 is not normalized, so the result carries the constant offset -ln(sum(p0))
        and is not zero in the ideal limit. That offset is the entropy cost of the
        constraint and belongs in ``Ss``; only keep it in mind when reading the
        plotted D_KL of such a system.

        ``pj0`` defaults to the stored ``self.pj0``; pass it explicitly to work
        on a composition grid that is not the cached one.
        """
        if pj0 is None:
            pj0 = self.pj0
        # xlogy(p, q) = p ln q with 0 ln 0 = 0: probabilities that underflow to
        # exactly 0 at low T contribute nothing instead of 0 * (-inf) = NaN.
        DKL = (xlogy(pj, pj) - xlogy(pj, pj0)).sum(axis=1)
        return DKL

    def Us(self, pj):
        """Configurational mixing energy (or enthalpy) per site."""
        U0 = pj @ self.delta_j
        return U0

    def Ss(self, pj, x=None, pj0=None):
        """
        Configurational mixing entropy per site,

            S / kB = -[ (n / sites) (x ln x + (1 - x) ln(1 - x)) + (w / sites) D(p || p0) ],

        with w = n for ``entropy_approximation="overlapping"`` and w = 1 for
        ``"non-overlapping"`` (see :attr:`D_weight`).

        ``x`` and ``pj0`` default to the stored grid (``self.x``, ``self.pj0``).
        """
        if x is None:
            x = self.x
        Sid = xlogy(x, x) + xlogy(1.0 - x, 1.0 - x)
        S0 = -kB / float(self.sites) * (self.n * Sid + self.D_weight * self.Ds(pj, pj0))

        return S0
    
    ######################################################################
    # --------------------- Phase diagram analysis --------------------- #
    ######################################################################
    # helper functions for phase diagram analysis: convex hull, tielines and spinodals   
    def calc_hull(self, x, G_interp, n_x_interp=1000):
        x_fine = np.linspace(x.min(), x.max(), n_x_interp)
        Gmin = np.column_stack((x_fine, G_interp(x_fine)))
        Gmin = Gmin[np.argsort(Gmin[:, 0])]  # ordering array by composition
        hull = spatial.ConvexHull(Gmin)  # getting convex hull for solution phases
        vertices = np.roll(hull.vertices, -1)  # shifting first position to end
        return Gmin, vertices

    def tielines(self, vertices, Gdata, dx, TOL=1e-8):
        t = np.diff(vertices)  # getting differences in vertices array
        t = np.where(t > 1)[0]  # filtering only diffs > 1
        tielines = []  # initializing tielines list
        for i in t:
            # checking for tielines based on dx + TOL
            if Gdata[vertices[i + 1], 0] - Gdata[vertices[i], 0] > dx + TOL:
                tielines.append(vertices[i])
                tielines.append(vertices[i + 1])
        nt = len(tielines)
        return tielines, nt

    def spinodals(self, x, G_interp):
        #find where d2G/dx2 is zero
        d2 = G_interp.derivative(nu=2)
        uvs = spinterp.PPoly.from_spline((d2.t, d2.c, d2.k)).roots(extrapolate=False)
        return uvs
    
    def _valid_spinodal_points(self, x, G_interp, sp, spinodal_len, G_deriv3_TOL=0.2):
        G_interp_deriv2 = G_interp.derivative(nu=2)
        G_interp_deriv3 = G_interp.derivative(nu=3)
        if len(sp) != 2:
            return False
        if G_interp_deriv2(x[0]) < 0 or G_interp_deriv2(x[-1]) < 0:
            return False
        if spinodal_len < 5:
            for sp_i in sp:
                if abs(G_interp_deriv3(sp_i)) < G_deriv3_TOL:
                    return False
        return True

    def calc_miscibility_gap(self, x:np.ndarray, T_min:float=20, dx_plot:float=0.01, max_iter:int=200):
        """ calculate miscibility gap (spinodal and binodal) vs composition x. 
        
            Parameters
            ----------
            x : np.ndarray
                Composition values.
            T_min : float, optional
                Starting temperature for the calculation.
            dx_plot : float, optional
                Approximate difference in x between two calculated spinodal points.
            max_iter : int, optional
                Maximum number of iterations before aborting, by default 200.
        """
        print("Calculating the miscibility gap...")
        x = np.asarray(x, dtype=float)
        dx = x[1] - x[0]
        dT = 10
        T = T_min

        binodal = []
        spinodal = []
        # the critical point is only defined if the scan actually converged onto it
        critical_point_found = False
        T_c = None
        x_c = None

        # all quantities of the temperature scan stay local: the apriori probabilities,
        # the vibrational free energies and the pj's are only used to locate the
        # binodal/spinodal points, so none of the scanned temperatures may end up in
        # the cached model data (self.pj0, self.Fvib_j, self.pj, ...)
        pj0 = self.calc_pj0(x)
        for i in range(max_iter):
            print(f'Calculating the binodal and spinodal points at T = {T:.3f}...', end='')
            deltaFvib_j = self.delta_Fvib_j(T, x, store=False)
            pj = self.calc_pj(T, x, deltaFvib_j)
            F = self.calc_F(T, pj, deltaFvib_j, x=x, pj0=pj0)

            # calculating hull, tielines ans spinodals for each T
            G_interp = spinterp.make_interp_spline(x, F, k=4)
            sp = self.spinodals(x, G_interp)
            if self._valid_spinodal_points(x, G_interp, sp, len(spinodal), G_deriv3_TOL=0.2):     # Prevent unwanted behavior at low temperatures
                spinodal.append([T, sp[0], sp[1]])

                Gdata, vertices = self.calc_hull(x, G_interp)
                tielines, nt = self.tielines(vertices, Gdata, dx)
                # adding tielines to binodal array: tielines holds (left, right) index
                # pairs of every gap on the hull; the binodal of this (single) concave
                # region is the gap that encloses its two spinodal points
                for left, right in zip(tielines[0::2], tielines[1::2]):
                    x_left, x_right = Gdata[left, 0], Gdata[right, 0]
                    if x_left <= sp[0] and sp[1] <= x_right:
                        binodal.append([T, x_left, x_right])
                        break
            else:
                # rejected points must not steer the scan (critical-point test, step size)
                sp = []

            #self._plot_F(Gdata[:,0], Gdata[:,1], hull, tielines, sp, T=T)
            if len(spinodal) > 1:       # calculate new dT based on slope of spinodal points
                if len(sp) != 2 and len(spinodal) > 5: # if after 5 spinodal points no new spinodal points are detected, reduce dT and recalculate
                    step_back = 0.75
                    print(f'   No spinodal points detected, reducing to T = {T - dT*step_back:.3f} K and recalculating...')
                    T -= dT*step_back
                    dT = dT*(1-step_back)
                    if dT < 1e-1:
                        print(f"dT = {dT:.3f} K. Sufficient temperature reslution reached. Stopping.")
                        break
                    continue
                elif len(sp) == 2:
                    diff_spi_1_2 = sp[1] - sp[0]
                    if diff_spi_1_2 < dx_plot:
                        # reported after the scan, once the spinodal line is confirmed
                        critical_point_found = True
                        T_c = T
                        x_c = (sp[0] + sp[1]) / 2
                        break
                    spinodal_dx = (spinodal[-1][1] - spinodal[-2][1])
                    slope_spinodal_1 = (spinodal[-1][0] - spinodal[-2][0]) / spinodal_dx
                    if slope_spinodal_1 >= 0:
                        dT = slope_spinodal_1 * dx_plot
                        if dT > 100:
                            dT = 100
                    elif slope_spinodal_1 < 0:
                        print(f'Warning: Negative slope detected for spinodal points at T = {T:.3f} K.')
                        print(f'   This may indicate numerical issues or a very flat spinodal curve. Consider changing the minimum Temperature for the miscibility gap calculation.')
            elif len(spinodal) == 1:
                dT = 5

            if dT < 1e-1:
                print(f"dT = {dT:.3f} K. Sufficient temperature reslution reached. Stopping.")
                break

            print(f'   dT = {dT:.3f} K')
            T += dT
        if not critical_point_found:
            if i == max_iter-1:
                print(f'\n\nMaximum number of iterations ({max_iter}) reached without finding critical point. Stopping.')
            else:
                print('\n\nStopped without finding a critical point.')
        elif len(spinodal) < 4:
            # too few accepted spinodal points for a miscibility gap (see below)
            print(f'\n\nCritical point candidate at T = {T_c:.3f} K discarded: only {len(spinodal)} spinodal points.')
        else:
            print(f'\n\nCritical point found, stopping after {i} iterations.')
            print(f'Critical point at T = {T_c:.3f} K, x = {x_c:.3f}')

        composition_number = self.extract_composition_number(self.compound)
        binodal = np.array(binodal)
        spinodal = np.array(spinodal)
        if binodal.shape[0] == 0:
            print(f'No binodal points found above {T_min:.1f} K.')
            self.binodal_line = np.array([])
        else:
            # the reversed branch keeps all points, including the lowest temperature one
            self.binodal_line = np.concatenate((binodal[:, [0,1]], binodal[::-1, [0,2]]), axis=0)
            # converting from x to x_plot in case of non-1 composition number in compound formula
            self.binodal_line[:,1] *= composition_number
        if spinodal.shape[0] < 4:
            print(f'No spinodal points found above {T_min:.1f} K, cannot determine miscibility gap.')
            self.spinodal_line = np.array([])
            self.T_c_miscibility_gap = None
            self.x_c_miscibility_gap = None
            return
        else:
            self.spinodal_line = np.concatenate((spinodal[:, [0,1]], spinodal[::-1, [0,2]]), axis=0)
            # converting from x to x_plot in case of non-1 composition number in compound formula
            self.spinodal_line[:,1] *= composition_number
            # only report a critical point when the scan converged onto one
            self.T_c_miscibility_gap = T_c if critical_point_found else None
            self.x_c_miscibility_gap = x_c * composition_number if critical_point_found else None
    

    ######################################################################
    # -------------------- Vibrational contribution -------------------- #
    ######################################################################
    def Fvib_clusters(self, T, store: bool = True):
        """
        Absolute vibrational free energy per site of every cluster in meV/site.
        Uses harmonic approximation:
        F_vib = kT * integral g(omega) ln[2 sinh(beta hbar omega / 2)] d omega.

        The result is cached in ``self.Fvib_j[T]``. Without phonon data all
        entries are zero. Already cached temperatures are always reused, but
        with ``store=False`` a newly computed temperature is not added to the
        cache, so scans over auxiliary temperatures (miscibility gap) leave
        ``self.Fvib_j`` restricted to the assessment grid ``self.Tlist``.
        """
        if not hasattr(self, "Fvib_j") or self.Fvib_j is None:
            self.Fvib_j = dict()
        if T in self.Fvib_j:
            return self.Fvib_j[T]

        kT = kB * T
        Fvib_j = np.zeros_like(self.delta_j)

        # if no phonons are given
        if self.phonon_data is None:
            if store:
                self.Fvib_j[T] = Fvib_j
            return Fvib_j

        for i in range(self.delta_j.size):
            w, phdos = self.phonon_data[i].T    # frequency, phonon dos

            # Exclude omega <= 0 explicitly to avoid log(0)
            mask_pos = w > 0.0
            w = w[mask_pos]
            phdos = phdos[mask_pos]

            if w.size == 0:
                # No positive frequencies, treat as zero contribution.
                Fvib_j[i] = 0.0
                continue

            u = hbar * w / (2.0 * kT)
            y = phdos * np.log(2.0 * np.sinh(u))

            # Remove any non-finite values from numerical noise
            mask_finite = np.isfinite(y)
            if not np.any(mask_finite):
                raise RuntimeError(
                    "No finite DOS values for phonon set {} at T = {}".format(i, T)
                )

            Fvib_j[i] = kT * spi.trapezoid(y[mask_finite], w[mask_finite])

        if store:
            self.Fvib_j[T] = Fvib_j
        return Fvib_j

    def delta_Fvib_j(self, T, x=None, store: bool = True):
        """
        Vibrational mixing free energy contribution per site for each cluster,
        referenced to the end-members. ``store=False`` keeps the temperature out
        of the ``self.Fvib_j`` cache, see :meth:`Fvib_clusters`.
        """
        Fvib_j = self.Fvib_clusters(T, store=store)
        deltaFvib_j = (
            Fvib_j - (1.0 - self.xj) * Fvib_j[0] - self.xj * Fvib_j[-1]
        )
        return deltaFvib_j

    def Fvib(self, deltaFvib_j, pj):
        """Average vibrational mixing free energy per site."""
        return pj @ deltaFvib_j

    def Svib(self, T, pj):
        """Average vibrational mixing entropy times temperature, T*Delta S_vib, in meV per site.

        Harmonic approximation as in https://doi.org/10.1016/j.commatsci.2018.03.057.
        Like the configurational part (``Sc = Ss * T``), the entropy is returned as
        T*S, i.e. in energy units, relative to the end members (x = 0 and x = 1) and
        averaged with the probabilities ``pj``.
        """

        kT = kB * T
        Svib_j = np.zeros_like(self.delta_j)

        if self.phonon_data is None:
            return pj @ Svib_j

        for i, ph_dat in enumerate(self.phonon_data):
            w = ph_dat[:,0]
            phdos = ph_dat[:,1]
            # Exclude omega <= 0 explicitly to avoid log(0)
            mask_pos = w > 0.0
            w = w[mask_pos]
            phdos = phdos[mask_pos]

            if w.size == 0:
                # No positive frequencies, treat as zero contribution.
                continue

            x = hbar * w / (2.0 * kT)
            integrand = (x/np.tanh(x)-np.log(2.0 * np.sinh(x))) * phdos

            # Remove any non-finite values from numerical noise
            mask_finite = np.isfinite(integrand)
            if not np.any(mask_finite):
                raise RuntimeError(
                    "No finite DOS values for phonon set {} at T = {}".format(i, T)
                )

            Svib_j[i] = kT * spi.trapezoid(integrand[mask_finite], w[mask_finite])

        delta_Svib_j = Svib_j - (1-self.xj)*Svib_j[0] - self.xj*Svib_j[-1]
 
        return pj @ delta_Svib_j
    
    def Uvib(self, T, pj):
        """Average vibrational internal energy per site (harmonic approximation)"""

        kT = kB * T
        Uvib_j = np.zeros_like(self.delta_j)

        if self.phonon_data is None:
            return pj @ Uvib_j

        for i, ph_dat in enumerate(self.phonon_data):
            w = ph_dat[:,0]
            phdos = ph_dat[:,1]

            if w.size == 0:
                # No positive frequencies, treat as zero contribution.
                continue

            x = hbar * w / (2.0 * kT)
            integrand = 1/2 *hbar*w/np.tanh(x) * phdos

            # Remove any non-finite values from numerical noise
            mask_finite = np.isfinite(integrand)
            if not np.any(mask_finite):
                raise RuntimeError(
                    "No finite DOS values for phonon set {} at T = {}".format(i, T)
                )

            # vibrational energy per cluster
            Uvib_j[i] = spi.trapezoid(integrand[mask_finite], w[mask_finite])

        # delta compared to pure elements
        delta_Uvib_j = Uvib_j - (1-self.xj)*Uvib_j[0] - self.xj*Uvib_j[-1]
 
        # weighted with cluster probabilities
        return pj @ delta_Uvib_j


    ######################################################################
    #----------------------------- Plotting -----------------------------#
    ######################################################################
    def xlabel_compound(self):
        """ Set up the xlabel"""
        if self.compound is None:
            return "x"
        else:
            return rf"x in {self.compound}"

    def plot_lat_pars(self, x:np.ndarray, Tlist:list[float], lat_pars=["a", "b", "c"], figsize:tuple=(6,6)):
        """ Plot lattice parameters vs composition x at each temperature in Tlist """
        x_label = self.xlabel_compound()

        for lat_par in lat_pars:
            for Ti in Tlist:
                lat_par_val = getattr(self.lattice, lat_par)/angstrom_to_meter
                lat_par_avg = self.gqca_mean(lat_par_val, Ti)     
                lat_par_std = self.gqca_std(lat_par_val, Ti)
                savename = f"lat-par_{lat_par}_{Ti}.pdf"
                self._save_dat_file(
                    self._matching_dat_filename(savename),
                    np.column_stack((x, lat_par_avg, lat_par_std)),
                    header=f"x\tavg_{lat_par}\tstd_{lat_par}",
                )

                with mpl.rc_context(self.style):
                    plt.subplots(figsize=figsize)
                    plt.xlim(x[0], x[-1])
                    plt.xticks((np.arange(np.floor(x[0]),x[-1]+0.1,step=np.ceil(np.max(x))/4)))

                    plt.xlabel(x_label)
                    plt.ylabel(rf"${lat_par}$ [\AA]")
                    plt.plot(x, lat_par_avg, color="blue", linestyle="-", linewidth=3.0, antialiased=True)
                    plt.fill_between(
                        x,
                        lat_par_avg - lat_par_std,
                        lat_par_avg + lat_par_std,
                        color="blue",
                        alpha=0.1,
                    )
                    plt.savefig(self.current_output_dir / self._add_phonon_suffix(savename), bbox_inches="tight", dpi=300)
                    plt.close()

    def plot_gqca_average(self, x:np.ndarray, y_avg:np.ndarray, y_std:np.ndarray, ylabel:str, savename:str="Average", figsize:tuple=(6,6)):
        """ Plot any averaged quantitiy vs composition x"""
        xlabel = self.xlabel_compound()
        self._save_dat_file(
            self._matching_dat_filename(f"{savename}.pdf"),
            np.column_stack((x, y_avg, y_std)),
            header=f"x\tavg\tstd\tlabel={ylabel}",
        )
        
        with mpl.rc_context(self.style):
            plt.subplots(figsize=figsize, constrained_layout=True)
            plt.xlim(x[0], x[-1])
            plt.xticks((np.arange(np.floor(x[0]),x[-1]+0.1,step=np.ceil(np.max(x))/4)))

            plt.xlabel(xlabel)
            plt.ylabel(rf"${ylabel}$")
            plt.plot(x, y_avg, color="blue", linestyle="-", linewidth=3.0, antialiased=True)        # , label="mean"
            plt.fill_between(
                x,
                y_avg - y_std,
                y_avg + y_std,
                color="blue",
                alpha=0.1,
                #label="std dev",
            )
            #plt.legend()
            plt.savefig(self.current_output_dir / self._add_phonon_suffix(f"{savename}.pdf"), bbox_inches="tight", dpi=300)
            plt.close()
    
    
    def plot_excess_enthalpy(self, figsize:tuple=(6,6), enumerate_clusters:bool = False):
        """plotting excess enthalpy in meV; ``enumerate_clusters`` numbers the points (key: cluster_numbers.dat)"""
        xlabel = self.xlabel_compound()

        # excess energies
        y_plot = self.delta_j.T

        aboveHull_flag = len(self.data_hull) > 2

        if aboveHull_flag:
            enthalpy = spatial.ConvexHull(self.data_hull)
            vertices = np.roll(enthalpy.vertices, -1)
            hull_points = self.data_hull[vertices]
        else:
            hull_points = np.empty((0, 2))

        # same x scale as all other plots: x = 0 ... composition number of the compound
        composition_number = self.extract_composition_number(self.compound)
        xj = self.xj * composition_number
        hull_x_plot = hull_points[:, 0] * composition_number
        on_hull = np.zeros_like(self.xj, dtype=int)
        if aboveHull_flag:
            # hull indices
            negative_positions = np.where(self.delta_j <= 0)[0]        # positions of all True entries
            on_hull_positions = negative_positions[vertices]
            on_hull = np.zeros_like(xj, bool)
            on_hull[on_hull_positions] = True
        self._save_dat_file(
            "excess_energy.dat",
            np.column_stack((xj, y_plot, on_hull)),
            header="x_cluster\tdelta_j / meV_per_site\ton_hull",
        )

        with mpl.rc_context(self.style):
            plt.figure(figsize=figsize)
            plt.xlim(xj[0]-1/20*xj[-1], xj[-1]*21/20)
            plt.xticks((np.arange(np.floor(xj[0]),xj[-1]+0.1,step=np.ceil(np.max(xj))/4)))
            plt.tick_params(axis='both',direction='in',bottom=True, top=True, left=True, right=True)
            plt.axhline(y=0, color='black', linestyle = '--')
            plt.xlabel(xlabel)
            plt.ylabel(rf'$\Delta_j$\,[meV/site]')
            if enumerate_clusters:
                # the cluster numbers are the markers: grey, blue (with background) on the hull
                if aboveHull_flag:
                    plt.plot(hull_x_plot, hull_points[:,1], color='blue', linestyle='-', linewidth='5', antialiased=True, label=f'{self.pressure_GPa} GPa', alpha=0.8)
                ax = plt.gca()
                numbers = np.arange(1, len(xj) + 1)
                on_hull_mask = np.asarray(on_hull, dtype=bool)
                for mask, color, background in ((~on_hull_mask, 'dimgrey', False), (on_hull_mask, 'blue', True)):
                    for number, xi, yi in zip(numbers[mask], xj[mask], y_plot[mask]):
                        ax.text(xi, yi, str(number), color=color, fontsize='small', ha='center', va='center',
                                bbox=_NUMBER_BOX if background else None, zorder=3, clip_on=True)
                self.write_cluster_numbers()
            else:
                plt.plot(xj, y_plot, color='black', marker='o', markersize=12, markeredgewidth=2.5, markerfacecolor='grey', linestyle=' ', alpha=0.5, label='higher accuracy')
                if aboveHull_flag:
                    plt.plot(hull_x_plot, hull_points[:,1], color='blue', marker='o', markersize=14, markeredgewidth=3, markerfacecolor='#3399ff', linestyle='-', linewidth='5', antialiased=True, label=f'{self.pressure_GPa} GPa', alpha=0.8)
            plt.savefig(self.current_output_dir / self._add_phonon_suffix('excess_energy.pdf'), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_thermodynamic_property(self, x:np.ndarray, y:np.ndarray, savename:str, ylabel:str, addLineLabel:bool = False, figsize:tuple=(6,6)):
        """Plot thermodynamic potentials in meV. Free energy, enthalpy, entropie."""
        xlabel = self.xlabel_compound()
        columns = [x]
        header = ["x"]
        for iT in self.Tlist:
            columns.append(y[iT])
            header.append(f"{ylabel}(T={iT:g})")
        self._save_dat_file(self._matching_dat_filename(savename), np.column_stack(columns), header="\t".join(header))
        
        # Colors
        colormap = Palette(120, 2, "coolwarm")
        colorplot = colormap.get_colors()
        if len(self.Tlist)*10 > len(colorplot):
            weight_color = int(np.floor(len(colorplot)/len(self.Tlist)))
        else:
            weight_color = 10
        
        with mpl.rc_context(self.style):
            plt.figure(figsize=figsize)
            plt.xlim(x[0], x[-1])
            plt.xticks((np.arange(np.floor(x[0]),x[-1]+0.1,step=np.ceil(np.max(x))/4)))
            plt.tick_params(axis='both',direction='in',bottom=True, top=True, left=True, right=True)
            plt.xlabel(xlabel)
            plt.ylabel(rf'{ylabel} [meV/site]')     # per site of the cluster, N = M*sites
            for idx, iT in enumerate(self.Tlist):       # remove loop when labelLines is working
                line, = plt.plot(x, y[iT], color=colorplot[idx*weight_color], linestyle='-', linewidth='3', antialiased=True)    # plot in meV
                if addLineLabel:      
                    labelLine(line, 0.5, label=str(int(iT)), align=True,backgroundcolor="none", fontsize=12)
                    #labelLines(plt.gca().get_lines(), zorder=2.5) # does not work for matplotlib.__version__ 3.10.1+dfsg1 
                    #plt.legend(loc='best', fontsize='11')

            plt.savefig(self.current_output_dir / self._add_phonon_suffix(savename), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_mixing_free_energy(self, x:np.ndarray, addLineLabel:bool = False, figsize:tuple=(6,6)):
        """plot total mixing free energy"""
        self.plot_thermodynamic_property(x, self.F, "Delta_G.pdf", r'$\Delta G$', addLineLabel=addLineLabel, figsize=figsize)

    def plot_mixing_enthalpy(self, x:np.ndarray, addLineLabel:bool = False, figsize:tuple=(6,6)):
        """plot total mixing enthalpy"""
        self.plot_thermodynamic_property(x, self.U, "Delta_H.pdf", r'$\Delta H$', addLineLabel=addLineLabel, figsize=figsize)

    def plot_mixing_entropy(self, x:np.ndarray, addLineLabel:bool = False, figsize:tuple=(6,6)):
        """plot total mixing entropy times temperature, T*Delta S (configurational + vibrational)"""
        self.plot_thermodynamic_property(x, self.S, "Delta_S.pdf", r'$T\Delta S$', addLineLabel=addLineLabel, figsize=figsize)

    def plot_vibrational_free_energy(self, x:np.ndarray, addLineLabel:bool = False, figsize:tuple=(6,6)):
        """plot vibrational mixing free energy Delta F_vib"""
        self.plot_thermodynamic_property(x, self.Fv, "Delta_Fvib.pdf", r'$\Delta F_{vib}$', addLineLabel=addLineLabel, figsize=figsize)

    def plot_thermo(self, x:np.ndarray | None = None, addLineLabel:bool = False, figsize:tuple=(6,6)):
        """plot all thermodynamic quanities"""

        if x is None:
            x = self.x_plot

        self.plot_mixing_enthalpy(x, addLineLabel=addLineLabel, figsize=figsize)
        self.plot_mixing_free_energy(x, addLineLabel=addLineLabel, figsize=figsize)
        self.plot_mixing_entropy(x, addLineLabel=addLineLabel, figsize=figsize)
        if self.phonon_data is not None:
            self.plot_vibrational_free_energy(x, addLineLabel=addLineLabel, figsize=figsize)

    def plot_miscibility_gap(self, figsize:tuple=(6,6)):
        """Save binodal_line.dat and spinodal_line.dat and plot the miscibility gap (T vs x)."""
        has_binodal = hasattr(self, "binodal_line") and self.binodal_line.size
        has_spinodal = hasattr(self, "spinodal_line") and self.spinodal_line.size
        if not (has_binodal or has_spinodal):
            print("No miscibility gap detected, cannot plot.")
            return

        # save raw binodal and spinodal lines
        if hasattr(self, "binodal_line") and self.binodal_line.size:
            self._save_dat_file(
                "binodal_line.dat",
                self.binodal_line,
                header="T\tx",
            )
        if hasattr(self, "spinodal_line") and self.spinodal_line.size:
            self._save_dat_file(
                "spinodal_line.dat",
                self.spinodal_line,
                header="T\tx",
            )

        # plot
        with mpl.rc_context(self.style):
            plt.figure(figsize=figsize)
            plt.xlim(self.x_plot[0], self.x_plot[-1])
            plt.xticks((np.arange(np.floor(self.x_plot[0]),self.x_plot[-1]+0.1,step=np.ceil(np.max(self.x_plot))/4)))
            plt.tick_params(axis='both',direction='in',bottom=True, top=True, left=True, right=True)
            plt.xlabel(self.xlabel_compound())
            plt.ylabel('$T$ [K]')

            # the critical point is only drawn when the temperature scan converged onto it
            if self.T_c_miscibility_gap is not None:
                plt.hlines(self.T_c_miscibility_gap, xmin=self.x_plot[0], xmax=self.x_plot[-1], colors='gray', linestyles='dashed', label=f'Critical point\nT$_{{cm}}$ = {self.T_c_miscibility_gap:.1f} K')
            if has_binodal:
                plt.plot(self.binodal_line[:, 1], self.binodal_line[:, 0], 'b', linewidth='3', antialiased=True, alpha=0.8, label='Binodal')
            if has_spinodal:
                plt.plot(self.spinodal_line[:, 1], self.spinodal_line[:, 0], color='red', linestyle='-', linewidth='3', antialiased=True, alpha=0.8, label='Spinodal')

            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.current_output_dir / self._add_phonon_suffix("miscibility_gap.pdf"), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_probabilities_vs_T(self, x_plot:np.ndarray | list[float], figsize:tuple=(6,6), enumerate_clusters:bool = False):
        """Plot pj vs T for every composition in ``x_plot``.

        ``x_plot`` holds raw compositions in [0, 1] (looked up in ``self.x``), not the
        scaled ``self.x_plot``. Labels and file names show the compound's x, i.e.
        x times the composition number of the compound, as all other plots do.
        ``enumerate_clusters`` writes the cluster number on each curve, near the
        high-T end (key: cluster_numbers.dat).
        """
        Tlist = self.Tlist
        composition_number = self.extract_composition_number(self.compound)
        if enumerate_clusters:
            self.write_cluster_numbers()

        # Colors
        colormap = plt.get_cmap('plasma')
            
        for x in x_plot:
            x_label = x * composition_number
            columns = [np.asarray(Tlist, dtype=float)]
            header = ["T"]
            x_idx = np.abs(self.x - x).argmin()
            for idx_cluster, cl_id in enumerate(self.cluster_id):
                probj = [self.pj[T][x_idx, idx_cluster] for T in Tlist]
                columns.append(probj)
                header.append(cl_id)
            self._save_dat_file(
                f"occurence_probabilities_x{x_label:.3f}.dat",
                np.column_stack(columns),
                header="\t".join(header),
            )

        with mpl.rc_context(self.style):
            for x in x_plot:
                x_label = x * composition_number
                plt.figure(figsize=figsize)
                for idx_cluster, cl_id in enumerate(self.cluster_id):
                    probj = []
                    for T in Tlist:
                        color = colormap(idx_cluster / (len(self.cluster_id)-1))
                        x_idx = np.abs(self.x - x).argmin()
                        probj.append((self.pj[T][x_idx, idx_cluster]))
                    plt.plot(Tlist, probj, color=color, linestyle='-', linewidth='3', antialiased=True)    
                if enumerate_clusters:
                    # near the high-T end, where the curves are spread out
                    T_label = Tlist[0] + 0.9 * (Tlist[-1] - Tlist[0])
                    label_cluster_curves(plt.gca(), plt.gca().get_lines(), np.full(len(self.cluster_id), T_label))

                plt.xlim(Tlist[0], Tlist[-1])
                plt.xticks((np.arange(np.floor(Tlist[0]),Tlist[-1]+0.1,step=np.ceil(np.max(Tlist))/4)))
                plt.tick_params(axis='both',direction='in',bottom=True, top=True, left=True, right=True)
                plt.xlabel(rf"$T$(K)")
                plt.ylabel(rf'$p_j(x={x_label:.3f}, T)$')     
                plt.savefig(self.current_output_dir / self._add_phonon_suffix(f"occurence_probabilities_x{x_label:.3f}.pdf"), bbox_inches='tight', dpi=300)
                plt.close()


    def plot_KL_divergence_vs_T(self, x_plot:np.ndarray | list[float], figsize:tuple=(6,6)):
        """Plot Kullback-Leibler divergence D(p || p0) vs T for each composition in x_plot.

        ``x_plot`` holds raw compositions in [0, 1] (looked up in ``self.x``), not the
        scaled ``self.x_plot``. The legend and file names show the compound's x, i.e.
        x times the composition number of the compound, as all other plots do.
        """
        Tlist = self.Tlist
        composition_number = self.extract_composition_number(self.compound)

        for x in x_plot:
            x_idx = np.abs(self.x - x).argmin()
            DKL = np.array([self.Ds(self.pj[T])[x_idx] for T in Tlist])
            self._save_dat_file(
                f"KL_divergence_x{x * composition_number:.3f}.dat",
                np.column_stack([np.asarray(Tlist, dtype=float), DKL]),
                header="T\tDKL",
            )

        with mpl.rc_context(self.style):
            colormap = plt.get_cmap('viridis')
            plt.figure(figsize=figsize)
            for i, x in enumerate(x_plot):
                x_idx = np.abs(self.x - x).argmin()
                DKL = np.array([self.Ds(self.pj[T])[x_idx] for T in Tlist])
                color = colormap(i / max(len(x_plot) - 1, 1))
                plt.plot(Tlist, DKL, color=color, linestyle='-', linewidth=3, antialiased=True, label=f"x={x * composition_number:.3f}")
            plt.xlim(Tlist[0], Tlist[-1])
            plt.xticks(np.arange(np.floor(Tlist[0]), Tlist[-1] + 0.1, step=np.ceil(np.max(Tlist)) / 4))
            plt.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
            plt.xlabel(r"$T$(K)")
            plt.ylabel(r"$D_\mathrm{KL}(p\,\|\,p_0)$")
            plt.legend()
            plt.savefig(self.current_output_dir / self._add_phonon_suffix("KL_divergence.pdf"), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_probabilities_vs_x(self, T_plot:np.ndarray | list[float], figsize:tuple=(6,6), enumerate_clusters:bool = False):
        """ plot pj vs composition x for every temperature given in T_plot.

        ``enumerate_clusters`` writes the cluster number at the maximum of each curve
        (key: cluster_numbers.dat).
        """
        Tlist = self.Tlist
        if enumerate_clusters:
            self.write_cluster_numbers()
                
        # Colors
        colormap = plt.get_cmap('plasma')
        # if len(self.Tlist)*10 > 255:
        #     weight_color = int(np.floor(255/len(self.Tlist)))
        # else:
        #     weight_color = 30


        # xlabel
        xlabel = self.xlabel_compound()
            
        for T in T_plot:
            if T not in Tlist:
                continue
            columns = [self.x_plot]
            header = ["x"]
            for idx_cluster, cl_id in enumerate(self.cluster_id):
                columns.append(self.pj[T][:, idx_cluster])
                header.append(cl_id)
            self._save_dat_file(
                f"occurence_probabilities_T{T}.dat",
                np.column_stack(columns),
                header="\t".join(header),
            )

        with mpl.rc_context(self.style):
            for T in T_plot:
                if T not in Tlist:
                    warnings.warn(rf"Probabilities at T={T} K have not been computed.")
                    continue
                plt.figure(figsize=figsize)
                for idx_cluster, cl_id in enumerate(self.cluster_id):
                    color = colormap(0.05 + 0.83 * idx_cluster / max(len(self.cluster_id) - 1, 1))
                    probj = self.pj[T][:, idx_cluster]
                    plt.plot(self.x_plot, probj, color=color, linestyle='-', linewidth='3', antialiased=True)  
                if enumerate_clusters:
                    # at the maximum of each curve
                    label_cluster_curves(plt.gca(), plt.gca().get_lines(), self.x_plot[self.pj[T].argmax(axis=0)])

                plt.xlim(self.x_plot[0], self.x_plot[-1])
                plt.xticks((np.arange(np.floor(self.x_plot[0]),self.x_plot[-1]+0.1,step=np.ceil(np.max(self.x_plot))/4)))
                plt.tick_params(axis='both',direction='in',bottom=True, top=True, left=True, right=True)
                plt.xlabel(xlabel)
                plt.ylabel(rf'$p_j$')     
                plt.savefig(self.current_output_dir / self._add_phonon_suffix(f"occurence_probabilities_T{T}.pdf"), bbox_inches='tight', dpi=300)
                plt.close()

    