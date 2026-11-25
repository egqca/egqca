# API Reference

This page is generated automatically from the docstrings of `egqca`. Everything listed here is re-exported through the package namespace, so it can be imported directly as `egqca.<name>`.

The narrative documentation of the individual workflows is in [Home](index.md), [ML Workflow](ml-workflow.md), [Cluster Reduction](cluster-reduction.md), [Properties Averaging](properties-averaging.md), and [DFT Input Files](dft-files.md).

## Main Class

::: egqca.EGQCA
    options:
      members:
        - set_up
        - entropy_approximation
        - D_weight
        - thermodynamic_assessment
        - average_properties
        - cluster_importance

## Cluster and DFT Setup

::: egqca.set_up_clusters_and_dft

::: egqca.set_up_clusters_dat

::: egqca.set_up_volume_dat

::: egqca.create_bands_in

::: egqca.create_phonons_in

::: egqca.update_DFT_in

::: egqca.get_directories

## Reading Results

::: egqca.extract_total_energies

::: egqca.extract_eddp_energies

::: egqca.write_properties_file

::: egqca.convert_volume_file

## Comparing Several Calculations

::: egqca.EGQCAInput

::: egqca.discover_egqca_inputs

::: egqca.make_egqca_inputs

::: egqca.run_egqca_comparison

::: egqca.plot_thermodynamic_comparison

::: egqca.plot_excess_enthalpy_comparison

::: egqca.plot_miscibility_gap_comparison

::: egqca.plot_occurrence_probabilities_comparison

::: egqca.plot_occurrence_probability_difference_from_ideal

::: egqca.plot_average_properties_comparison

## Optional ML Helpers

These functions are always importable; they only need ASE, which is a core dependency. The
backend of a built-in model is imported when the model is first used, so install the matching
extra (`egqca[upet]`, `egqca[mace]`, `egqca[mattersim]` or `egqca[grace]`) before running it,
otherwise an `ImportError` is raised. A user-supplied calculator needs no extra.

::: egqca.ml.run_ml_relaxation

::: egqca.ml.collect_ml_results

::: egqca.ml.reset_ml_clusters

::: egqca.ml.rerun_nonconverged_ml

::: egqca.ml.geometry_relaxation

::: egqca.ml.phonons

::: egqca.ml.phonon_dos_from_cache
