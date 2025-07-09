# Changelog

All notable changes to `casm-bset` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-08-09

### Added

- Added `--test` option to `__main__` script to test specific configuration flags for clexulator compilation

### Changed

- Make `--autoconfigure` call `--test` as subprocesses and prefer no `-D_GLIBCXX_USE_CXX11_ABI=0`


## [2.0.0] - 2025-05-04

### Changed

- Restrict requires-python to ">=3.9,<3.14"
- Run CI tests using Python 3.13


## [2.0a4] - 2025-02-12

### Fixed

- Fixed site-centric function variable naming (update variables after transforming cluster functions to assign correct neighborhood_site_index and name). C++ functions are unchanged.


## [2.0a3] - 2025-02-10

### Added

- Added formulas using neighbor list indexing

### Fixed

- Fixed cluster function variable naming (update variables after transforming cluster functions to assign correct neighborhood_site_index and name). C++ functions are unchanged.


## [2.0a2] - 2024-12-11

### Added

- Added `ClexulatorWriter.generated_files` attribute, holding a list of the files generated when writing a Clexulator
- Added generation of latex formulas during the Clexulator writing process
- Added `casm.bset.cluster_functions.MakeVariableName` for variable naming
- Added `verbose` and `very_verbose` options to `casm.bset.write_clexulator`
- Added `to_dict` methods to `ClusterFunctionsBuilder`, `OrbitMatrixRepBuilder` and `ClusterMatrixRepBuilder`
- Added `to_dict` and `from_dict` methods to `ExponentSumConstraint`
- Added `casm.bset.json_io` module
- Added option to specify occ_site_basis_functions_specs of type occupation with a choice of reference occupant on each sublattice.
- Added a check that prints an error message and raises an exception if the constant occupation site basis function mixes with other site basis functions.

### Changed

- Updated `ClexulatorWriter.write` to write a "variables.json.gz" file for each Clexulator (including local Clexulator) which contains the variables used by the jinja2 templates as well as information like basis function formulas generated during the write process
- Updated `ClexulatorWriter.write` to write a "cluster_functions.json.gz" file with the generated clusters, matrix reps, and functions.
- Changed the name used for occupation variables. The name "\\phi" is the base, but the number and meaning of the indices can vary depending on the site basis functions. An "occ_site_functions_info" dict in the "variables.json" file contains the string value used as a template ("occ_var_name") and a description of the variable including its indices, if any ("occ_var_desc" and "occ_var_indices").

### Fixed

- Fixed `v1.basic` templates so that the Clexulator compiles for a prim with occupation DoF, but only the constant basis  function is included.
- Fixed tests with anisotropic occupants and occupation site basis functions.


## [2.0a1] - 2024-08-15

This release creates the casm-bset CASM cluster expansion basis set construction module. This includes:

- Methods for generating coupled cluster expansion Hamiltonians of occupation, strain, displacement, and magnetic spin degrees of freedom (DoF) appropriate for the symmetry of any multi-component crystalline solid.
- Methods for generating C++ code for a CASM cluster expansion calculator (Clexulator) which efficiently evaluates the cluster expansion basis function for configuration represented using the CASM `ConfigDoFValues` data structure
- Generalized methods for creating symmetry adapted basis functions of other variables

This package is designed to work with the cluster expansion calculator (Clexulator) evaluation methods which are implemented in [libcasm-clexulator](https://github.com/prisms-center/CASMcode_clexulator). 

This package may be installed via pip install. This release also includes API documentation, built using Sphinx.
