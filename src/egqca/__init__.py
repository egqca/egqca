from ._banner import __version__
from .egqca_model import EGQCA
from .io import convert_volume_file
from .workflow import (
    create_bands_in,
    create_phonons_in,
    extract_total_energies,
    set_up_clusters_and_dft,
    set_up_clusters_dat,
    set_up_volume_dat,
    get_directories,
    update_DFT_in,
    write_properties_file,
    extract_eddp_energies,
)
from .comparison import (
    EGQCAInput,
    discover_egqca_inputs,
    make_egqca_inputs,
    plot_average_properties_comparison,
    plot_excess_enthalpy_comparison,
    plot_miscibility_gap_comparison,
    plot_occurrence_probabilities_comparison,
    plot_occurrence_probability_difference_from_ideal,
    plot_thermodynamic_comparison,
    run_egqca_comparison,
)

try:
    from .ml import (
        geometry_relaxation,
        run_ml_relaxation,
        collect_ml_results,
        reset_ml_clusters,
        rerun_nonconverged_ml,
        phonon_dos_from_cache,
    )
except ImportError:  # optional ML stack
    geometry_relaxation = None
    run_ml_relaxation = None
    collect_ml_results = None
    reset_ml_clusters = None
    rerun_nonconverged_ml = None
    phonon_dos_from_cache = None


__all__ = [
    "__version__",
    "EGQCA",
    "convert_volume_file",
    "set_up_clusters_and_dft",
    "run_ml_relaxation",
    "collect_ml_results",
    "reset_ml_clusters",
    "rerun_nonconverged_ml",
    "extract_total_energies",
    "create_bands_in",
    "create_phonons_in",
    "set_up_clusters_dat",
    "set_up_volume_dat",
    "get_directories",
    "update_DFT_in",
    "write_properties_file",
    "extract_eddp_energies",
    "geometry_relaxation",
    "phonon_dos_from_cache",
    "EGQCAInput",
    "discover_egqca_inputs",
    "make_egqca_inputs",
    "run_egqca_comparison",
    "plot_thermodynamic_comparison",
    "plot_excess_enthalpy_comparison",
    "plot_miscibility_gap_comparison",
    "plot_occurrence_probabilities_comparison",
    "plot_occurrence_probability_difference_from_ideal",
    "plot_average_properties_comparison",
]
