from typing import Any, Callable, Iterable, Optional

import numpy as np

import libcasm.clusterography as casmclust
import libcasm.configuration as casmconfig
import libcasm.configuration.io as config_io
import libcasm.sym_info as sym_info
import libcasm.xtal as xtal
from casm.bset.cluster_functions._discrete_functions import (
    get_occ_site_functions,
    make_occ_site_functions,
    make_occ_site_functions_info,
)
from casm.bset.cluster_functions._matrix_rep import (
    MakeVariableName,
    OrbitMatrixRepBuilder,
    make_variable_name,
)
from casm.bset.cluster_functions._misc import (
    orbits_to_dict,
)
from casm.bset.polynomial_functions import (
    ExponentSumConstraint,
    FunctionRep,
    PolynomialFunction,
    Variable,
    make_symmetry_adapted_polynomials,
)
from libcasm.clexulator import (
    PrimNeighborList,
    make_default_prim_neighbor_list,
)

from ._clex_basis_specs import (
    ClexBasisSpecs,
)


def _get_dof_types(
    xtal_prim: xtal.Prim,
    dofs: Optional[list[str]] = None,
) -> tuple[list[str], list[str], list[str], dict]:
    """Given a list of dofs, make lists of which are global, local continuous, and local
    discrete.

    Parameters
    ----------
    xtal_prim: xtal.Prim
        The Prim
    dofs: Optional[Iterable[str]] = None
        An iterable of string of dof type names that should be used to construct basis
        functions. The default value is all DoF types included in the prim.

    Returns
    -------
    (global_dof, local_continuous_dof, local_discrete_dof):

        global_dof: list[str]
            The types of global degree of freedom (DoF). All global DoF are treated
            as continuous.

        local_continuous_dof: list[str]
            The types of local discrete degree of freedom (DoF).

        local_discrete_dof: list[str]
            The types of local discrete degree of freedom (DoF).

    """
    all_global_dof = set()
    for _global_dof in xtal_prim.global_dof():
        all_global_dof.add(_global_dof.dofname())

    all_local_continuous_dof = set()
    for _sublattice_dof in xtal_prim.local_dof():
        for _local_dof in _sublattice_dof:
            all_local_continuous_dof.add(_local_dof.dofname())

    # TODO: support other local discrete dof
    all_local_discrete_dof = set()
    for _occ_dof in xtal_prim.occ_dof():
        if len(_occ_dof) > 1:
            all_local_discrete_dof.add("occ")

    if dofs is None:
        global_dof = list(all_global_dof)
        local_continuous_dof = list(all_local_continuous_dof)
        local_discrete_dof = list(all_local_discrete_dof)
    else:
        global_dof = []
        local_continuous_dof = []
        local_discrete_dof = []
        for _dofname in dofs:
            if _dofname in all_global_dof:
                global_dof.append(_dofname)
            elif _dofname in all_local_continuous_dof:
                local_continuous_dof.append(_dofname)
            elif _dofname in all_local_discrete_dof:
                local_discrete_dof.append(_dofname)
            else:
                raise Exception(f"Error in _get_dof_types: Unknown dof '{_dofname}'")
    return (global_dof, local_continuous_dof, local_discrete_dof)


def _default_nlist_sublat_indices(
    xtal_prim: xtal.Prim,
) -> list[int]:
    """Default sublattice indices for PrimNeighborList

    Typically, sublattices are included in the neighbor list if they have allowed DoF,
    either >1 occupant DoF allowed, or 1 continuous DoF.

    Parameters
    ----------
    xtal_prim: xtal.Prim
        The Prim.

    Returns
    -------
    sublat_indices: list[int]
        The indices of the sublattices that should be included in the neighbor list.

    """
    sublat_indices = set()
    occ_dof = xtal_prim.occ_dof()
    for b, sublattice_occ_dof in enumerate(occ_dof):
        if len(sublattice_occ_dof) >= 2:
            sublat_indices.add(b)
    local_dof = xtal_prim.local_dof()
    for b, sublattice_local_dof in enumerate(local_dof):
        if len(sublattice_local_dof) > 0:
            sublat_indices.add(b)
    return list(sublat_indices)


def _make_orbits_data(
    prim: casmconfig.Prim,
    clex_basis_specs: ClexBasisSpecs,
    clusters: list[list[casmclust.Cluster]],
    functions: list[list[list[PolynomialFunction]]],
    coordinate_mode: str,
):
    xtal_prim = prim.xtal_prim
    cluster_specs = clex_basis_specs.cluster_specs
    if xtal_prim != cluster_specs.xtal_prim():
        raise Exception(
            "Error in ClusterFunctionsBuilder.basis_dict: "
            "Prim and cluster specs have different xtal_prim."
        )
    if len(clusters[0][0]) != 0:
        raise Exception(
            "Error in ClusterFunctionsBuilder.basis_dict: "
            "This method requires a null cluster orbit."
        )

    def _brief_desc(op: xtal.SymOp) -> str:
        sym_info = xtal.SymInfo(op, xtal_prim.lattice())
        if coordinate_mode == "cart":
            return sym_info.brief_cart()
        elif coordinate_mode == "frac":
            return sym_info.brief_frac()
        else:
            raise Exception(
                "Error in ClusterFunctionsBuilder.basis_dict: "
                f"Unknown coordinate_mode: {coordinate_mode}, "
                "must be 'cart' or 'frac'."
            )

    is_periodic = cluster_specs.phenomenal() is None
    generating_group_site_rep = casmclust.make_integral_site_coordinate_symgroup_rep(
        group_elements=cluster_specs.generating_group().elements,
        xtal_prim=prim.xtal_prim,
    )

    orbits_data = []
    linear_function_index = 0
    for i_orbit, orbit_functions in enumerate(functions):
        orbit = clusters[i_orbit]
        prototype = orbit[0]

        if is_periodic:
            cluster_group = casmclust.make_cluster_group(
                cluster=prototype,
                group=cluster_specs.generating_group(),
                lattice=xtal_prim.lattice(),
                integral_site_coordinate_symgroup_rep=generating_group_site_rep,
            )
        else:
            cluster_group = casmclust.make_local_cluster_group(
                cluster=prototype,
                phenomenal_group=cluster_specs.generating_group(),
                integral_site_coordinate_symgroup_rep=generating_group_site_rep,
            )

        orbit_data = {
            "linear_orbit_index": i_orbit,
            "mult": len(orbit),
            "prototype": orbit[0].to_dict(
                xtal_prim=xtal_prim, phenomenal=cluster_specs.phenomenal()
            ),
        }
        orbit_data["prototype"]["invariant_group"] = cluster_group.head_group_index
        orbit_data["prototype"]["invariant_group_descriptions"] = [
            _brief_desc(op) for op in cluster_group.elements
        ]

        cluster_functions = []
        if len(prototype) == 0:
            # add constant function
            cluster_functions.append(
                {
                    "\\Phi_{0}": "1",
                    "linear_function_index": linear_function_index,
                }
            )
            linear_function_index += 1
        if len(orbit_functions) != 0:
            for i_func, func in enumerate(orbit_functions[0]):
                cluster_functions.append(
                    {
                        f"\\Phi_{{{linear_function_index}}}": func.latex_formula(),
                        "linear_function_index": linear_function_index,
                    }
                )
                linear_function_index += 1
        orbit_data["cluster_functions"] = cluster_functions
        orbits_data.append(orbit_data)
    return orbits_data


def _make_occ_site_function_basis_data(
    prim: casmconfig.Prim,
    occ_site_function: dict,
):
    """Generate basis dict for an occupation site basis function

    Parameters
    ----------
    prim: casmconfig.Prim
        The Prim
    occ_site_function: dict
        Occupation site basis function values, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values,
          as ``value[function_index][occupant_index]``.


    Returns
    -------
    data: dict
        A dict with the format:

        .. code-block:: Python

            # b: sublattice index,
            # r: site basis function index
            # occ1, occ2, etc.: Occupant DoF names, from prim.xtal_prim.occ_dof()[b]
            {
                "basis": {
                    "\\phi_{b,f}": {
                        "occ1": value1,
                        "occ2": value2,
                        ...
                    },
                    ...
                },
                "value": [
                    [value1, value2, ...],
                    ...
                ]
            }
    """
    b = occ_site_function["sublattice_index"]
    phi = np.array(occ_site_function["value"])
    occ_dof = prim.xtal_prim.occ_dof()[b]
    if len(occ_dof) != phi.shape[0] or len(occ_dof) != phi.shape[1]:
        raise Exception(
            "Error in _make_occ_site_function_basis_data: "
            "Inconsistent number of site basis functions or occupants. "
            "Sublattice index: {b}; Phi shape = {phi.shape}; "
        )
    data = {
        "basis": {},
        "value": phi.tolist(),
    }
    for r in range(phi.shape[0]):
        key = f"\\phi_{{{b},{r}}}"
        d = {}
        for c in range(phi.shape[1]):
            d[occ_dof[c]] = phi[r][c]
        data["basis"][key] = d
    return data


def _make_site_functions_data(
    prim: casmconfig.Prim,
    occ_site_functions: Optional[list[dict]],
):
    """Generate basis dict for site functions

    Parameters
    ----------
    prim: casmconfig.Prim
        The Prim
    occ_site_functions: Optional[list[dict]]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values,
          as ``value[function_index][occupant_index]``.

    Returns
    -------
    data: list[dict]
        List of site functions data, with the format:

        .. code-block:: Python

            [
                {
                    "sublat": b,  # sublattice index
                    "asym_unit": a,  # asymmetric unit index
                    "occ": { # if discrete site basis functions are present
                        "basis": {
                            "\\phi_{b,f}": {
                                "occ1": value1,
                                "occ2": value2,
                                ...
                            },
                            ...
                        },
                        "value": [
                            [value1, value2, ...],
                            ...
                        ]
                    }
                },
                ...
            ]

    """

    sublat_to_asym_unit = {}
    for a, sublat_indices in enumerate(xtal.asymmetric_unit_indices(prim.xtal_prim)):
        for b in sublat_indices:
            sublat_to_asym_unit[b] = a

    site_functions_data = [
        {"sublat": b, "asym_unit": a} for b, a in sublat_to_asym_unit.items()
    ]

    if occ_site_functions is not None:
        for occ_site_function in occ_site_functions:
            b = occ_site_function["sublattice_index"]
            site_functions_data[b]["occ"] = _make_occ_site_function_basis_data(
                prim=prim, occ_site_function=occ_site_function
            )
    return site_functions_data


def update_variables(
    prim: casmconfig.Prim,
    cluster: casmclust.Cluster,
    function: PolynomialFunction,
    prim_neighbor_list: PrimNeighborList,
    translation: Optional[np.ndarray] = None,
    make_variable_name_f: Optional[Callable] = None,
    local_discrete_dof: Optional[list[str]] = None,
):
    """Update variables after transforming cluster function to assign
    neighborhood_site_index and name for function variables

    Parameters
    ----------
    prim: libcasm.configuration.Prim
        The prim
    cluster: libcasm.clusterography.Cluster
        The cluster associated with the function.
    function: casm.bset.polynomial_functions.PolynomialFunction
        A PolynomialFunction with variables that have cluster_site_index set
        referring to which site in `cluster` they are associated with.
    prim_neighbor_list: libcasm.clexulator.PrimNeighborList
        The neighbor list
    translation: Optional[np.ndarray],
        Optional translation to apply to cluster.
    make_variable_name_f: Optional[Callable] = None
        Allows specifying a custom class to construct variable names. The default
        class used is :class:`~casm.bset.cluster_functions.MakeVariableName`.
        Custom classes should have the same `__call__` signature as
        :class:`~casm.bset.cluster_functions.MakeVariableName`, and have
        `occ_var_name` and `occ_var_desc` attributes.
    local_discrete_dof: Optional[list[str]] = None
        The types of local discrete degree of freedom (DoF).
    """
    if make_variable_name_f is None:
        make_variable_name_f = make_variable_name
    if local_discrete_dof is None:
        local_discrete_dof = list()

    for var in function.variables:
        if var.cluster_site_index is not None:
            integral_site_coordinate = cluster[var.cluster_site_index]
            if translation is not None:
                integral_site_coordinate = integral_site_coordinate + translation

            # Set neighborhood site index
            var.neighborhood_site_index = prim_neighbor_list.neighbor_index(
                integral_site_coordinate
            )

            # Set variable name:
            var.name = make_variable_name_f(
                xtal_prim=prim.xtal_prim,
                key=var.key,
                site_basis_function_index=var.site_basis_function_index,
                component_index=var.component_index,
                cluster_site_index=var.cluster_site_index,
                sublattice_index=integral_site_coordinate.sublattice(),
                local_discrete_dof=local_discrete_dof,
            )


def make_equivalent_cluster_basis_sets(
    prototype_cluster_basis_set: list[list[PolynomialFunction]],
    equivalence_map_clusters: list[list[casmclust.Cluster]],
    equivalence_map_inv_matrix_rep: list[list[np.ndarray]],
    prim_neighbor_list: PrimNeighborList,
    verbose: bool = False,
    prim: Optional[casmconfig.Prim] = None,
    i_orbit: Optional[int] = None,
    make_variable_name_f: Optional[Callable] = None,
    local_discrete_dof: Optional[list[str]] = None,
) -> tuple[list[casmclust.Cluster], list[list[PolynomialFunction]]]:
    orbit = []
    orbit_basis_sets = []

    # generate basis sets on all equivalent clusters
    if verbose:
        print("Generating equivalents...")
        print()
    # i_equiv: index for equivalent clusters
    # M_list: sym rep matrices (1 for each sym op that prototype cluster to
    #     the equivalent cluster, only the first is needed here)
    for i_equiv, M_list in enumerate(equivalence_map_inv_matrix_rep):
        # add equivalent cluster to orbit of clusters
        equiv_cluster = equivalence_map_clusters[i_equiv][0]
        orbit.append(equiv_cluster)
        if verbose:
            print(f"~~~ i_orbit: {i_orbit}, i_equiv: {i_equiv} ~~~")
            print("Equivalent cluster:")
            print(xtal.pretty_json(equiv_cluster.to_dict(prim.xtal_prim)))
            print()

        # add basis set for equivalent cluster to orbit of basis sets
        equiv_basis_set = []
        for f_prototype in prototype_cluster_basis_set:
            M = M_list[0]
            S = FunctionRep(matrix_rep=M)
            f_equiv = S * f_prototype
            update_variables(
                prim=prim,
                cluster=equiv_cluster,
                function=f_equiv,
                prim_neighbor_list=prim_neighbor_list,
                make_variable_name_f=make_variable_name_f,
                local_discrete_dof=local_discrete_dof,
            )
            assert f_equiv.variables is not f_prototype.variables
            for i in range(len(f_prototype.variables)):
                assert f_equiv.variables[i] is not f_prototype.variables[i]
            equiv_basis_set.append(f_equiv)
        orbit_basis_sets.append(equiv_basis_set)
        if verbose:
            print("Equivalent cluster basis set:")
            if len(equiv_basis_set) == 0:
                print("Empty")
            for i_func, func in enumerate(equiv_basis_set):
                print(f"~~~ order: {func.order()}, function_index: {i_func} ~~~")
                func._basic_print()
                print()
            print()

    return (orbit, orbit_basis_sets)


def make_constraints(
    prototype_cluster: casmclust.Cluster,
    prototype_variables: list[Variable],
    prototype_variable_subsets: list[set[int]],
    occ_site_functions: list[dict],
    local_discrete_dof: list[str],
):
    """Make constraints for constructing symmetry adapted polynomials

    Parameters
    ----------
    prototype_cluster: libcasm.clusterography.Cluster
        The prototype cluster.
    prototype_variables: list[casm.bset.polynomial_functions.Variable]
        The prototype cluster function variables, including global and local variables.
    prototype_variable_subsets: list[set[int]]
        The indices of Variable in `prototype_variables` which mix under application of
        symmetry, or permute as a group. A subset could be strain variables,
        displacement variables on a site, or occupant site basis functions on a site.
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values,
          as ``value[function_index][occupant_index]``.

    local_discrete_dof: list[str]
            The types of local discrete degree of freedom (DoF).

    Returns
    -------
    constraints: list[casm.bset.polynomial_functions.ExponentSumConstraint]
        Constraints that set the exponent for constant discrete site functions to 0,
        and ensure one and only one of mutually exclusive discrete variables
        is included in any monomial.
    """
    constraints = []
    for subset in prototype_variable_subsets:
        var = prototype_variables[subset[0]]
        if var.cluster_site_index is None or var.key not in local_discrete_dof:
            continue

        # Find the constant site function and set constraint to exclude
        found_constant_function = False
        b = None
        for i_var in subset:
            var = prototype_variables[i_var]
            b = prototype_cluster[var.cluster_site_index].sublattice()
            site_function_index = var.site_basis_function_index
            phi = get_occ_site_functions(
                occ_site_functions=occ_site_functions,
                sublattice_index=b,
                site_function_index=site_function_index,
            )

            if (phi == 1.0).all():
                constraints.append(ExponentSumConstraint(variables=[i_var], sum=[0]))
                found_constant_function = True
        if not found_constant_function:
            raise Exception(
                "Error in make_constraints: "
                f"discrete DoF (sublattice={b}) has no constant site basis function"
            )

        # Set mutually exclusive discrete variable subset constraint
        constraints.append(ExponentSumConstraint(variables=subset, sum=[0, 1]))

    return constraints


class ClusterFunctionsBuilder:
    """Constructs cluster functions

    Notes
    -----

    Cluster functions are generated using the following steps:

    1. A cluster from each orbit is input, along with the phenomenal cluster if local
       cluster functions are to be generated.

    2. For each orbit,
       :func:`~casm.bset.polynomial_functions.make_symmetry_adapted_polynomials`
       is used to construct symmetry-adapted polynomial functions
       (:class:`~casm.bset.polynomial_functions.PolynomialFunction`) using matrix
       representations constructed by
       :class:`~casm.bset.cluster_functions.OrbitMatrixRepBuilder`.

       - For functions of discrete degrees of freedom (DoF), the polynomial functions
         include a single occupation site basis function on each site in a cluster.
       - For functions that include continuous DoF, by default, polynomials of order up
         to the cluster size are created. Higher order polynomials are requested either
         according to cluster size using the `orbit_branch_max_poly_order` parameter or
         globally using `global_max_poly_order`. For each orbit, the most specific
         level specified is used.
       - Functions are generated for all clusters in the orbit by default. If only
         functions on the prototype cluster are needed, then the `make_equivalents`
         parameter can be set to `False`.
       - The `prim_neighbor_list` is used to set the `neighborhood_site_index` for the
         :class:`~casm.bset.polynomial_functions.Variable` corresponding to local DoF
         used by each :class:`~casm.bset.polynomial_functions.PolynomialFunction`.

    3. If this is a local cluster expansion (if `phenomenal` is not None), then
       local cluster functions are generated for the orbits associated with each of the
       symmetrically equivalent phenomenal clusters.

       - The symmetrically equivalent, but distinct, local cluster expansions are
         found by using the prim factor group to generate equivalent phenomenal
         clusters and excluding those that are equivalent by `generating_group`
         symmetry.
       - For local cluster expansions, all local cluster expansions in the orbit are
         constructed by default. If only functions in the original orbits are needed,
         then the `make_all_local_basis_sets` parameter can be set to `False`.

    """

    def __init__(
        self,
        prim: casmconfig.Prim,
        generating_group: sym_info.SymGroup,
        clusters: Optional[Iterable[casmclust.Cluster]] = None,
        phenomenal: Optional[casmclust.Cluster] = None,
        dofs: Optional[Iterable[str]] = None,
        global_max_poly_order: Optional[int] = None,
        orbit_branch_max_poly_order: Optional[dict[int, int]] = None,
        occ_site_basis_functions_specs: Any = None,
        prim_neighbor_list: Optional[PrimNeighborList] = None,
        make_equivalents: bool = True,
        make_all_local_basis_sets: bool = True,
        make_variable_name_f: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """

        .. rubric:: Constructor

        Parameters
        ----------
        prim: libcasm.configuration.Prim
            The Prim, with symmetry information
        generating_group: libcasm.sym_info.SymGroup
            The symmetry group for generating cluster functions. For periodic cluster
            functions, this is the prim factor group (usually) or a subgroup. For
            local cluster functions, this is the cluster group of the phenomenal
            cluster or a subgroup (often the subgroup which leaves an
            :class:`~libcasm.occ_events.OccEvent` invariant).
        clusters: Iterable[libcasm.clusterography.Cluster]
            An iterable of :class:`~libcasm.clusterography.Cluster` containing the
            a single cluster from each orbit to generate functions for. If not provided,
            clusters are generated using the `max_length`, phenomenal`, and
            `cutoff_radius` parameters.
        phenomenal: Optional[libcasm.clusterography.Cluster] = None
            For local cluster functions, specifies the sites about which
            local-clusters orbits are generated. The phenomenal cluster must be chosen
            from one of the equivalents that is generated by
            :func:`~libcasm.clusterography.make_periodic_orbit` using the prim factor
            group.
        dofs: Optional[Iterable[str]] = None
            An iterable of string of dof type names that should be used to construct
            basis functions. The default value is all DoF types included in the prim.
        global_max_poly_order: Optional[int] = None
            The maximum order of polynomials of continuous DoF to generate, for any
            orbit not specified more specifically by `orbit_branch_max_poly_order`.
        orbit_branch_max_poly_order: Optional[dict[int, int]] = None
            Specifies for continuous DoF the maximum polynomial order to generate by
            cluster size, according to
            ``orbit_branch_max_poly_order[cluster_size] = max_poly_order``. By default,
            for a given cluster orbit, polynomials of order up to the cluster size are
            created. Higher order polynomials are requested either according to cluster
            size using `orbit_branch_max_poly_order` or globally using
            `global_max_poly_order`. The most specific level specified is used.
        occ_site_basis_functions_specs: Any = None
            Provides instructions for constructing occupation site basis functions.
            The most common options are "chebychev" or "occupation". This
            parameter corresponds to the value of

            .. code-block:: Python

                "dof_specs": {
                    "occ": {
                        "site_basis_functions": ...
                    }
                }

            as described in detail in the section
            :ref:`DoF Specifications <sec-dof-specifications>` and is required for
            functions of occupation DoF.
        prim_neighbor_list: Optional[libcasm.clexulator.PrimNeighborList] = None
            The :class:`PrimNeighborList` is used to uniquely index sites with local
            variables included in the cluster functions, relative to a reference unit
            cell. If not provided, a PrimNeighborList is constructed using default
            parameters that include all sites with degrees of freedom (DoF) and the
            default shape used by CASM projects. In most cases, the default should be
            used.
        make_equivalents: bool = True
            If True, make all equivalent clusters and functions. Otherwise, only
            construct functions on the prototype cluster (i.e. ``i_equiv=0`` only).
        make_all_local_basis_sets: bool = True
            If True, make local clusters and functions for all phenomenal
            clusters in the primitive cell equivalent by prim factor group symmetry.
            Requires that `make_equivalents` is True.
        make_variable_name_f: Optional[Callable] = None
            Allows specifying a custom class to construct variable names. The default
            class used is :class:`~casm.bset.cluster_functions.MakeVariableName`.
            Custom classes should have the same `__call__` signature as
            :class:`~casm.bset.cluster_functions.MakeVariableName`, and have
            `occ_var_name` and `occ_var_desc` attributes.
        verbose: bool = False
            Print progress statements

        """

        if orbit_branch_max_poly_order is None:
            orbit_branch_max_poly_order = {}

        if prim_neighbor_list is None:
            prim_neighbor_list = make_default_prim_neighbor_list(
                xtal_prim=prim.xtal_prim
            )

        global_dof, local_continuous_dof, local_discrete_dof = _get_dof_types(
            prim.xtal_prim, dofs
        )

        only_discrete = len(global_dof) + len(local_continuous_dof) == 0

        # Validation
        if len(local_discrete_dof) == 0:
            if occ_site_basis_functions_specs is not None:
                print(
                    "\n"
                    "****************************************************************\n"
                    "** Warning in ClusterFunctionsBuilder:                        **\n"
                    "** No local discrete DoF, but:                                **\n"
                    "** - occ_site_basis_functions_specs is not None               **\n"
                    "** Will use occ_site_basis_functions_specs=None               **\n"
                    "****************************************************************\n"
                )
                occ_site_basis_functions_specs = None
        if len(global_dof + local_continuous_dof) > 0:
            if global_max_poly_order is None and len(orbit_branch_max_poly_order) == 0:
                print(
                    "\n"
                    "****************************************************************\n"
                    "** Warning in ClusterFunctionsBuilder:                        **\n"
                    "** Continuous DoF included, but:                              **\n"
                    "** - global_max_poly_order is None, and                       **\n"
                    "** - len(orbit_branch_max_poly_order) == 0                    **\n"
                    "** Did you forget to set global_max_poly_order??              **\n"
                    "****************************************************************\n"
                )

        # Data
        self._prim = prim
        """libcasm.configuration.Prim: The Prim, with symmetry information"""

        self._generating_group = generating_group
        """libcasm.sym_info.SymGroup: The symmetry group used to generate clusters and \
        functions."""

        self.global_dof = global_dof
        """list[str]: List of global continuous DoF included in functions. """

        self.local_continuous_dof = local_continuous_dof
        """list[str]: List of local continuous DoF included in functions."""

        self.local_discrete_dof = local_discrete_dof
        """list[str]: List of local discrete DoF included in functions."""

        local_dof = local_continuous_dof + local_discrete_dof
        self.local_dof = local_dof
        """list[str]: The types of local continuous and discrete degrees of freedom \
        (DoF) included in the matrix representation.
        """

        self.only_discrete = only_discrete
        """bool: True if only discrete DoF are included in functions."""

        self._clusters = clusters
        """list[libcasm.clusterography.Cluster]: A list of clusters with a single \
        cluster from each orbit to generate functions for.
        """

        self._phenomenal = phenomenal
        """Optional[Cluster]: If provided, specifies the sites about which \
        local-clusters orbits are generated. 
        
        The phenomenal cluster must be chosen from one of the equivalents that is 
        generated by :func:`~libcasm.clusterography.make_periodic_orbit` using the prim
        factor group.
        """

        self._global_max_poly_order = global_max_poly_order
        """int: The default maximum polynomial order.
        
        The maximum order of polynomials of continuous DoF to generate, for any orbit
        not specified more specifically by `orbit_branch_max_poly_order`.
        """

        self._orbit_branch_max_poly_order = orbit_branch_max_poly_order
        """dict[int, int]: The per-orbit-branch maximum polynomial order.
        
        Specifies for continuous DoF the maximum polynomial order to generate by
        cluster size, according to
        ``orbit_branch_max_poly_order[cluster_size] = max_poly_order``. By default, for
        a given cluster orbit, polynomials of order up to the cluster size are created.
        Higher order polynomials are requested either according to cluster size
        using `orbit_branch_max_poly_order` or globally using `global_max_poly_order`.
        The most specific level specified is used.
        """

        self._occ_site_basis_functions_specs = occ_site_basis_functions_specs
        """Union[str, list[dict], None]: Instructions for constructing occupation \
        site basis functions.
            
        The accepted options are "chebychev", "occupation", or a `list[dict]`
        a specifying sublattice-specific choice of site basis functions. 
        
        This parameter corresponds to the value of

        .. code-block:: Python

            "dof_specs": {
                "occ": {
                    "site_basis_functions": ...
                }
            }

        as described in detail in the section
        :ref:`DoF Specifications <sec-dof-specifications>` and is required for
        functions of occupation DoF."""

        occ_site_functions = []
        if occ_site_basis_functions_specs is not None:
            occ_site_functions = make_occ_site_functions(
                prim=prim,
                occ_site_basis_functions_specs=occ_site_basis_functions_specs,
            )

        self.occ_site_functions = occ_site_functions
        """list[dict]: List of occupation site basis functions.
        
        For each sublattice with discrete site basis functions, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values,
          as ``value[function_index][occupant_index]``.
       
        """

        occ_site_functions_info = make_occ_site_functions_info(
            prim=prim,
            occ_site_functions=occ_site_functions,
        )
        # note: this generates default occ_var_name and occ_var_desc, which may be
        # overridden by a custom make_variable_name_f
        self.occ_site_functions_info = occ_site_functions_info
        """dict: Information about occupation site basis functions.
        
        Occupation site basis functions info, with format:

        - `"max_function_index"`: int, The maximum site function index, across all
          sublattices.
        - `"all_sublattices_have_same_site_functions"`: bool, True if all _sublattices
          have same site functions; False otherwise.
        - `"occ_var_name"`: str, A variable name template for the site functions,
          which may be formated using `b` for sublattice index and `m` for site function
          index (i..e ``occ_var_name.format(b=0, m=1)``).
        - `"occ_var_desc"`: str, A description of the occupation
          variable, including a description of the subscript indices.
        """

        self.prim_neighbor_list = prim_neighbor_list
        """libcasm.clexulator.PrimNeighborList: The PrimNeighborList, expanded as \
        necessary
        
        The :class:`~libcasm.clexulator.PrimNeighborList` used to uniquely index sites 
        with local variables included in the cluster functions, relative to a reference 
        unit cell.
        """

        self._make_equivalents = make_equivalents
        """bool: If True, make all equivalent clusters and functions. 
        
        Otherwise, only construct functions on the prototype cluster.
        """

        self._make_all_local_basis_sets = make_all_local_basis_sets
        """bool: If True, make all equivalent orbits of local clusters and local \
        functions. 

        If True, make local clusters and functions for all phenomenal clusters in the 
        primitive cell equivalent by prim factor group symmetry. Requires that 
        `make_equivalents` is True.
        """

        if make_variable_name_f is None:
            make_variable_name_f = MakeVariableName(
                occ_var_name=self.occ_site_functions_info.get("occ_var_name"),
                occ_var_desc=self.occ_site_functions_info.get("occ_var_desc"),
            )
        else:
            self.occ_site_functions_info["occ_var_name"] = (
                make_variable_name_f.occ_var_name
            )
            self.occ_site_functions_info["occ_var_desc"] = (
                make_variable_name_f.occ_var_desc
            )
        self._make_variable_name_f = make_variable_name_f
        """Callable: Function used to construct variable names.
        
        Allows specifying a custom class to construct variable names. The default
        class used is :class:`MakeVariableName`. Custom classes should have
        the same `__call__` signature as :class:`MakeVariableName`, have
        `occ_var_name`, `occ_var_desc`, `occ_var_indices` attributes, and a `to_dict`
        method.
        """

        self._verbose = verbose
        """bool: If True, print progress statements"""

        ## Construct cluster functions on a prototype cluster for each orbit
        prototype_basis_sets = []
        orbit_matrix_rep_builders = []
        constraints = []
        for i_orbit, cluster in enumerate(self._clusters):
            _1, _2, _3 = self._build_prototype_basis_set(i_orbit, cluster)
            prototype_basis_sets.append(_1)
            orbit_matrix_rep_builders.append(_2)
            constraints.append(_3)
        self.prototype_basis_sets = prototype_basis_sets
        """list[list[PolynomialFunction]]: For each orbit, the symmetry adapted \
        polynomial cluster functions for the prototype cluster.
        
        The function ``prototype_basis_sets[i_orbit][i_func]`` is the `i_func`-th 
        function on the prototype cluster in the `i_orbit`-th orbit.
        
        The orbits are generated from the input clusters (stored in 
        :py:data:`~casm.bset.cluster_functions.ClusterFunctionBuilder._clusters`), and 
        stored in the same order. The prototype cluster is determined from sorting the 
        equivalent clusters in the orbit and may not be the input cluster.
        """

        self.orbit_matrix_rep_builders = orbit_matrix_rep_builders
        """list[casm.bset.cluster_functions.OrbitMatrixRepBuilder]: For each orbit,
        the OrbitMatrixRepBuilder contains matrix representations of symmetry
        operations for generating symmetry-adapted polynomial cluster functions on the
        orbit prototype, coupling all local and global DoFs, and matrix representations
        for constructing cluster functions on symmetrically equivalent clusters.
        """

        self.constraints = constraints
        """list[list[ExponentSumConstraint]]: For each orbit, the constraints used \
        when constructing `prototype_basis_set` to ensure one and only one of \
        mutually exclusive discrete variables is included in any monomial.
         
        If a local cluster expansion, ``constraints[i_orbit]`` are also the correct
        constraints for the `i_orbit`-th orbit about each equivalent phenomenal cluster.
        """

        ## Construct equivalent cluster functions
        orbit_basis_sets = None
        orbit_clusters = None
        equivalent_orbit_basis_sets = None
        equivalent_orbit_clusters = None
        if make_equivalents:
            orbit_basis_sets = []
            orbit_clusters = []

            ## Transform cluster functions
            # from prototype clusters to equivalent clusters
            for i_orbit, orbit_matrix_rep_builder in enumerate(
                self.orbit_matrix_rep_builders
            ):
                _1, _2 = self._build_orbit_basis_sets(
                    i_orbit=i_orbit,
                    orbit_matrix_rep_builder=orbit_matrix_rep_builder,
                    prototype_basis_set=self.prototype_basis_sets[i_orbit],
                )
                orbit_basis_sets.append(_1)
                orbit_clusters.append(_2)

                ## For local cluster expansions, transform cluster functions
                # from the local basis set around the original phenomenal cluster
                # to the local basis sets around equivalent phenomenal clusters
                if self._phenomenal is not None and make_all_local_basis_sets:
                    _3, _4 = self._build_equivalent_orbit_basis_sets(
                        i_orbit=i_orbit,
                        orbit_matrix_rep_builder=orbit_matrix_rep_builder,
                        orbit_basis_sets=_1,
                        orbit_clusters=_2,
                    )
                    n_clex = len(_3)
                    if equivalent_orbit_basis_sets is None:
                        equivalent_orbit_basis_sets = [list() for _ in range(n_clex)]
                        equivalent_orbit_clusters = [list() for _ in range(n_clex)]
                    for i_clex in range(n_clex):
                        equivalent_orbit_basis_sets[i_clex].append(_3[i_clex])
                        equivalent_orbit_clusters[i_clex].append(_4[i_clex])

        self.functions = orbit_basis_sets
        """list[list[list[PolynomialFunction]]]: The generated cluster functions

        Polynomial functions, where ``functions[i_orbit][i_equiv][i_func]``, is the 
        `i_func`-th function on the cluster given by `clusters[i_orbit][i_equiv]`.
        """

        n_functions = 1
        for orbit_basis_set in orbit_basis_sets:
            n_functions += len(orbit_basis_set[0])
        self.n_functions = n_functions
        """int: The total number of symmetrically distinct cluster functions generated
        for all clusters in all orbits plus one (for the constant term)."""

        self.clusters = orbit_clusters
        """list[list[libcasm.clusterography.Cluster]]: The clusters for which 
        cluster functions have been constructed
        
        The cluster ``clusters[i_orbit][i_equiv]`` is the `i_equiv`-th symmetrically 
        equivalent cluster in the `i_orbit`-th orbit.

        The order of sites in the clusters is not arbitrary, it is consistent with the 
        `cluster_site_index` of the :class:`~casm.bset.polynomial_functions.Variable`
        used in the :class:`~casm.bset.polynomial_functions.PolynomialFunction` stored 
        in :py:data:`~ClusterFunctionsBuilder.functions`.
        """

        self.equivalent_functions = equivalent_orbit_basis_sets
        """Optional[list[list[list[list[\
        casm.bset.polynomial_functions.PolynomialFunction]]]]]: \
        The generated cluster functions about all equivalent phenomenal clusters (if \
        a local cluster expansion).
        
        Symmetry adapted polynomial cluster functions for each cluster in each
        orbit about a symmetrically equivalent phenomenal cluster,
        where ``equivalent_functions[i_clex][i_orbit][i_equiv][i_func]``, is the
        `i_func`-th function on the `i_equiv`-th cluster in the `i_orbit`-th orbit about
        the `i_clex`-th equivalent phenomenal cluster.
        """

        self.equivalent_clusters = equivalent_orbit_clusters
        """Optional[list[list[list[libcasm.clusterography.Cluster]]]]: Orbit of \
        clusters for which cluster functions have been constructed about all \
        equivalent phenomenal clusters (if a local cluster expansion).

        The cluster ``equivalent_clusters[i_clex][i_orbit][i_equiv]`` is the
        `i_equiv`-th symmetrically equivalent cluster in the `i_orbit`-th orbit about 
        the `i_clex`-th equivalent phenomenal cluster.

        The order of sites in the clusters is not arbitrary, it is consistent with the 
        `cluster_site_index` of the :class:`~casm.bset.polynomial_functions.Variable`
        used in the :class:`~casm.bset.polynomial_functions.PolynomialFunction` stored 
        in :py:data:`~ClusterFunctionsBuilder.equivalent_functions`.
        """

    def to_dict(self):
        data = {}

        def _clusters_data(clusters):
            clusters_data = []
            for orbit_clusters in clusters:
                orbit_clusters_data = []
                for cluster in orbit_clusters:
                    orbit_clusters_data.append({"sites": cluster.to_list()})
                clusters_data.append(orbit_clusters_data)
            return clusters_data

        def _functions_data(functions):
            functions_data = []
            for orbit_functions in functions:
                orbit_functions_data = []
                for equiv_functions in orbit_functions:
                    equiv_functions_data = []
                    for function in equiv_functions:
                        equiv_functions_data.append(function.to_dict())
                    orbit_functions_data.append(equiv_functions_data)
                functions_data.append(orbit_functions_data)
            return functions_data

        # Add prim
        data["prim"] = self._prim.to_dict()

        # Add phenomenal
        data["phenomenal"] = (
            {"sites": self._phenomenal.to_list()}
            if self._phenomenal is not None
            else None
        )

        # Add generating_group (as indices)
        data["generating_group"] = self._generating_group.head_group_index

        # Add misc parameters...
        data.update(
            {
                "global_max_poly_order": self._global_max_poly_order,
                "orbit_branch_max_poly_order": self._orbit_branch_max_poly_order,
                "occ_site_basis_functions_specs": self._occ_site_basis_functions_specs,
                "make_equivalents": self._make_equivalents,
                "make_all_local_basis_sets": self._make_all_local_basis_sets,
                "make_variable_name_f": self._make_variable_name_f.to_dict(),
            }
        )

        # Add dof info...
        data.update(
            {
                "global_dof": self.global_dof,
                "local_continuous_dof": self.local_continuous_dof,
                "local_discrete_dof": self.local_discrete_dof,
                "local_dof": self.local_dof,
                "only_discrete": self.only_discrete,
            }
        )

        # Add site functions
        data.update(
            {
                "occ_site_functions": self.occ_site_functions,
                "occ_site_functions_info": self.occ_site_functions_info,
            }
        )

        # Add orbit_matrix_rep_builders
        data["orbit_matrix_rep_builders"] = [
            orbit_matrix_rep_builder.to_dict()
            for orbit_matrix_rep_builder in self.orbit_matrix_rep_builders
        ]

        # Add self.clusters
        data["clusters"] = _clusters_data(self.clusters)

        # Add constraints
        data["constraints"] = [
            [constraint.to_dict() for constraint in orbit_constraints]
            for orbit_constraints in self.constraints
        ]

        # Add self.functions
        data["functions"] = _functions_data(self.functions)

        # Add self.n_functions
        data["n_functions"] = self.n_functions

        # Add self.equivalent_clusters
        data["equivalent_clusters"] = (
            [
                _clusters_data(equiv_clusters)
                for equiv_clusters in self.equivalent_clusters
            ]
            if self.equivalent_clusters is not None
            else None
        )

        # Add self.equivalent_functions
        data["equivalent_functions"] = (
            [
                _functions_data(equiv_functions)
                for equiv_functions in self.equivalent_functions
            ]
            if self.equivalent_functions is not None
            else None
        )

        return data

    def _min_poly_order(self, orbit_prototype):
        if self.only_discrete:
            return len(orbit_prototype)
        return 1

    def _max_poly_order(self, orbit_prototype):
        if self.only_discrete:
            return len(orbit_prototype)
        branch = len(orbit_prototype)
        if branch in self._orbit_branch_max_poly_order:
            return self._orbit_branch_max_poly_order[branch]
        elif self._global_max_poly_order is not None:
            return int(self._global_max_poly_order)
        else:
            return branch

    def _build_prototype_basis_set(
        self,
        i_orbit: int,
        cluster: casmclust.Cluster,
    ):
        """Build the prototype cluster basis set

        Parameters
        ---------
        i_orbit: int
            Orbit number, starting from 0, used for printing information when
            `verbose` is set to `True`.
        cluster: libcasm.clusterography.Cluster
            A cluster in the orbit on which to build the cluster functions.

        Returns
        -------
        (prototype_basis_set, orbit_matrix_rep_builder, constraints):
            prototype_basis_set: list[PolynomialFunction]
                Symmetry adapted polynomial cluster functions for the prototype
                cluster in the orbit generated from `cluster`.

            orbit_matrix_rep_builder: casm.bset.cluster_functions.OrbitMatrixRepBuilder
                The OrbitMatrixRepBuilder contains matrix representations for generating
                polynomial cluster functions on the orbit prototype, coupling all local
                and global DoFs.

                Also includes:

                - Matrix representations for generating equivalent functions on other
                  clusters in the orbit.
                - Matrix representations for constructing local cluster functions for
                  symmetrically equivalent phenomenal clusters (local cluster functions
                  only).

            constraints: list[ExponentSumConstraint]
                The constraints used when constructing `prototype_basis_set` to ensure
                one and only one of mutually exclusive discrete variables
                is included in any monomial.
        """

        if self._verbose:
            print(f"### i_orbit: {i_orbit} ###")
            print()

            if self._phenomenal is not None:
                print("Phenomenal cluster:")
                print(xtal.pretty_json(self._phenomenal.to_dict(self._prim.xtal_prim)))
                print()

            print("Initial cluster:")
            print(xtal.pretty_json(cluster.to_dict(self._prim.xtal_prim)))
            print()

        builder = OrbitMatrixRepBuilder(
            prim=self._prim,
            generating_group=self._generating_group,
            global_dof=self.global_dof,
            local_continuous_dof=self.local_continuous_dof,
            local_discrete_dof=self.local_discrete_dof,
            cluster=cluster,
            phenomenal=self._phenomenal,
            make_variable_name_f=self._make_variable_name_f,
            occ_site_functions=self.occ_site_functions,
        )
        constraints = make_constraints(
            prototype_cluster=cluster,
            prototype_variables=builder.prototype_variables,
            prototype_variable_subsets=builder.prototype_variable_subsets,
            occ_site_functions=self.occ_site_functions,
            local_discrete_dof=self.local_discrete_dof,
        )

        if len(builder.prototype_variables) == 0:
            # if no variables, no functions
            if self._verbose:
                print("No variables")
            prototype_basis_set = []

        elif builder.n_local_variables == 0 and cluster.size() != 0:
            # if no local variables and not the null cluster, no functions
            # (only generate pure global DoF functions on the null cluster)
            if self._verbose:
                print("No local variables")
            prototype_basis_set = []

        else:
            if self._verbose:
                print("Variables:")
                for i_var, var in enumerate(builder.prototype_variables):
                    print(f"{i_var}: {var.name}")
                print()
                print("Variable subsets:")
                for i_subset, subset in enumerate(builder.prototype_variable_subsets):
                    print(f"{i_subset}:", end="")
                    for i_var in subset:
                        print(f"{builder.prototype_variables[i_var].name}, ", end="")
                    print()
                print()

            prototype_basis_set = make_symmetry_adapted_polynomials(
                matrix_rep=builder.prototype_matrix_rep,
                variables=builder.prototype_variables,
                variable_subsets=builder.prototype_variable_subsets,
                min_poly_order=self._min_poly_order(cluster),
                max_poly_order=self._max_poly_order(cluster),
                constraints=constraints,
                orthonormalize_in_place=False,
                verbose=self._verbose,
            )

            if self._verbose:
                print("Prototype cluster basis set:")
                if len(prototype_basis_set) == 0:
                    print("Empty")
                for i_func, func in enumerate(prototype_basis_set):
                    print(f"~~~ order: {func.order()}, function_index: {i_func} ~~~")
                    func._basic_print()
                    print()
                print()

        orbit_matrix_rep_builder = builder
        return (prototype_basis_set, orbit_matrix_rep_builder, constraints)

    def _build_orbit_basis_sets(
        self,
        i_orbit: int,
        orbit_matrix_rep_builder: OrbitMatrixRepBuilder,
        prototype_basis_set: list[PolynomialFunction],
    ):
        """Build basis sets for each cluster in an orbit

        Parameters
        ---------
        i_orbit: int
            Orbit number, starting from 0, used for printing information when
            `verbose` is set to `True`.
        orbit_matrix_rep_builder: casm.bset.cluster_functions.OrbitMatrixRepBuilder
            An OrbitMatrixRepBuilder which contains matrix representations of
            symmetry operations for generating symmetry-adapted polynomial cluster
            functions for the orbit from the orbit prototype, as output from
            :func:`~casm.bset.cluster_functions.ClusterFunctionsBuilder._build_prototype_basis_set`.
        prototype_basis_set: list[casm.bset.polynomial_functions.PolynomialFunction]
            Symmetry adapted polynomial cluster functions for the prototype
            cluster in the orbit, as output from
            :func:`~casm.bset.cluster_functions.ClusterFunctionsBuilder._build_prototype_basis_set`.

        Returns
        -------
        (orbit_basis_sets, orbit_clusters):

            orbit_basis_sets: list[list[\
            casm.bset.polynomial_functions.PolynomialFunction]]
                Symmetry adapted polynomial cluster functions for each cluster in
                the orbit, where ``orbit_basis_sets[i_equiv][i_func]``, is the
                `i_func`-th function on the `i_equiv`-th cluster in the orbit.

            orbit_clusters: list[libcasm.clusterography.Cluster]]:
                Orbit of clusters for which `orbit_basis_sets` has been constructed.

                The cluster ``orbit_clusters[i_equiv]`` is the `i_equiv`-th
                symmetrically equivalent cluster in the orbit.

                The order of sites in the clusters is not arbitrary, it is consistent
                with the `cluster_site_index` of the
                :class:`~casm.bset.polynomial_functions.Variable`
                used in the `orbit_basis_sets` functions.
        """
        builder = orbit_matrix_rep_builder

        orbit_basis_sets = []
        orbit_clusters = []

        # generate basis sets on all equivalent clusters
        if self._verbose:
            print("Generating equivalent cluster functions in orbit...")
            print()
        # i_equiv: index for equivalent clusters
        # M_list: sym rep matrices (1 for each sym op that prototype cluster to
        #     the equivalent cluster, only the first is needed here)
        for i_equiv, M_list in enumerate(builder.equivalence_map_inv_matrix_rep):
            # add equivalent cluster to orbit of clusters
            equiv_cluster = builder.equivalence_map_clusters[i_equiv][0]
            orbit_clusters.append(equiv_cluster)

            if self._verbose:
                print(f"~~~ i_orbit: {i_orbit}, i_equiv: {i_equiv} ~~~")
                print("Equivalent cluster:")
                print(xtal.pretty_json(equiv_cluster.to_dict(self._prim.xtal_prim)))
                print()

            # add basis set for equivalent cluster to orbit of basis sets
            equiv_basis_set = []
            for f_prototype in prototype_basis_set:
                M = M_list[0]
                S = FunctionRep(matrix_rep=M)
                f_equiv = S * f_prototype
                update_variables(
                    prim=self._prim,
                    cluster=equiv_cluster,
                    function=f_equiv,
                    prim_neighbor_list=self.prim_neighbor_list,
                    make_variable_name_f=self._make_variable_name_f,
                    local_discrete_dof=self.local_discrete_dof,
                )
                assert f_equiv.variables is not f_prototype.variables
                for i in range(len(f_prototype.variables)):
                    assert f_equiv.variables[i] is not f_prototype.variables[i]
                equiv_basis_set.append(f_equiv)
            orbit_basis_sets.append(equiv_basis_set)

            if self._verbose:
                print("Equivalent cluster basis set:")
                if len(equiv_basis_set) == 0:
                    print("Empty")
                for i_func, func in enumerate(equiv_basis_set):
                    print(f"~~~ order: {func.order()}, function_index: {i_func} ~~~")
                    func._basic_print()
                    print()
                print()

        return (orbit_basis_sets, orbit_clusters)

    def _build_equivalent_orbit_basis_sets(
        self,
        i_orbit: int,
        orbit_matrix_rep_builder: OrbitMatrixRepBuilder,
        orbit_basis_sets: list[list[PolynomialFunction]],
        orbit_clusters: list[casmclust.Cluster],
    ):
        """Build basis sets for each cluster in the equivalent orbits around \
        equivalent phenomenal cluster.

        Parameters
        ----------
        i_orbit: int
            Orbit number, starting from 0, used for printing information when
            `verbose` is set to `True`.

        orbit_matrix_rep_builder: casm.bset.cluster_functions.OrbitMatrixRepBuilder
            An OrbitMatrixRepBuilder which contains matrix representations of
            symmetry operations for generating polynomial cluster functions for the
            orbit from the orbit prototype, as output from
            :func:`~casm.bset.cluster_functions.ClusterFunctionsBuilder._build_prototype_basis_set`.

        orbit_basis_sets: list[list[casm.bset.polynomial_functions.PolynomialFunction]]
            Symmetry adapted polynomial cluster functions for each cluster in
            the orbit, where ``orbit_basis_sets[i_equiv][i_func]``, is the
            `i_func`-th function on the `i_equiv`-th cluster in the orbit.

        orbit_clusters: list[libcasm.clusterography.Cluster]]:
            Orbit of clusters for which `orbit_basis_sets` has been constructed.

            The cluster ``orbit_clusters[i_equiv]`` is the `i_equiv`-th
            symmetrically equivalent cluster in the orbit.

            The order of sites in the clusters is not arbitrary, it is consistent
            with the `cluster_site_index` of the
            :class:`~casm.bset.polynomial_functions.Variable`
            used in the `orbit_basis_sets` functions.

        Returns
        -------
        (equivalent_orbit_basis_sets, equivalent_orbit_clusters):

            equivalent_orbit_basis_sets: list[list[list[\
            casm.bset.polynomial_functions.PolynomialFunction]]]
                Symmetry adapted polynomial cluster functions for each cluster in each
                equivalent orbit about a symmetrically equivalent phenomenal cluster,
                where ``equivalent_orbit_basis_sets[i_clex][i_equiv][i_func]``, is the
                `i_func`-th function on the `i_equiv`-th cluster in the orbit about
                the `i_clex`-th equivalent phenomenal cluster.

            equivalent_orbit_clusters: list[list[libcasm.clusterography.Cluster]]]:
                Orbit of clusters for which `orbit_basis_sets` has been constructed.

                The cluster ``equivalent_orbit_clusters[i_clex][i_equiv]`` is the
                `i_equiv`-th symmetrically equivalent cluster in the orbit about the
                `i_clex`-th equivalent phenomenal cluster.

                The order of sites in the clusters is not arbitrary, it is consistent
                with the `cluster_site_index` of the
                :class:`~casm.bset.polynomial_functions.Variable`
                used in the `orbit_basis_sets` functions.

        """
        builder = orbit_matrix_rep_builder

        equivalent_orbit_basis_sets = []
        equivalent_orbit_clusters = []

        # generate equivalent local basis sets
        if self._verbose:
            print(
                "Generating local cluster functions "
                "for equivalent phenomenal clusters..."
            )
            print()

        # i_clex: index for equivalent local cluster expansion
        for i_clex in range(len(builder.phenomenal_generating_ops)):
            if self._verbose:
                print(f"*** i_orbit: {i_orbit}, i_clex: {i_clex} ***")
                print("Equivalent phenomenal cluster:")
                site_rep = builder.phenomenal_generating_site_rep[i_clex]
                equiv_phenomenal = site_rep * self._phenomenal
                print(xtal.pretty_json(equiv_phenomenal.to_dict(self._prim.xtal_prim)))
                print()

            _equiv_orbit_basis_sets = []
            _equiv_orbit_clusters = []
            # i_clust: index for equivalent clusters
            # M: sym rep matrix for transforming functions on the `i_clust`-th cluster
            #    to the orbit around the `i_clex`-th equivalent phenomenal cluster
            for i_clust, M in enumerate(
                builder.phenomenal_generating_inv_matrix_rep[i_clex]
            ):
                # add equivalent cluster to orbit of clusters
                site_rep = builder.phenomenal_generating_site_rep[i_clex]
                equiv_cluster = site_rep * orbit_clusters[i_clust]
                _equiv_orbit_clusters.append(equiv_cluster)

                if self._verbose:
                    print(
                        f"~~~ i_orbit: {i_orbit}, "
                        f"i_clex: {i_clex}, "
                        f"i_clust: {i_clust} ~~~"
                    )
                    print("Equivalent cluster:")
                    print(xtal.pretty_json(equiv_cluster.to_dict(self._prim.xtal_prim)))
                    print()

                # add basis set for equivalent cluster to orbit of basis sets
                equiv_basis_set = []
                for f_prototype in orbit_basis_sets[i_clust]:
                    S = FunctionRep(matrix_rep=M)
                    f_equiv = S * f_prototype
                    update_variables(
                        prim=self._prim,
                        cluster=equiv_cluster,
                        function=f_equiv,
                        prim_neighbor_list=self.prim_neighbor_list,
                        make_variable_name_f=self._make_variable_name_f,
                        local_discrete_dof=self.local_discrete_dof,
                    )
                    assert f_equiv.variables is not f_prototype.variables
                    for i in range(len(f_prototype.variables)):
                        assert f_equiv.variables[i] is not f_prototype.variables[i]
                    equiv_basis_set.append(f_equiv)
                _equiv_orbit_basis_sets.append(equiv_basis_set)

                if self._verbose:
                    print("Equivalent cluster basis set:")
                    if len(equiv_basis_set) == 0:
                        print("Empty")
                    for i_func, func in enumerate(equiv_basis_set):
                        print(
                            f"~~~ order: {func.order()}, function_index: {i_func} ~~~"
                        )
                        func._basic_print()
                        print()
                    print()

            equivalent_orbit_basis_sets.append(_equiv_orbit_basis_sets)
            equivalent_orbit_clusters.append(_equiv_orbit_clusters)

        return (equivalent_orbit_basis_sets, equivalent_orbit_clusters)

    def basis_dict(
        self,
        clex_basis_specs: ClexBasisSpecs,
        coordinate_mode: str = "frac",
    ) -> dict:
        R"""Generate the CASM basis.json data

        Parameters
        ----------
        clex_basis_specs: casm.bset.cluster_functions.ClexBasisSpecs
            The specifications for the cluster expansion basis set.
        coordinate_mode: str = "frac"
            The coordinate mode used to represent cluster invariant group operations.
            The default value is "frac". The other option is "cart".

        Returns
        -------
        data: dict
            A description of the generated cluster expansion basis set, as described
            `here <https://prisms-center.github.io/CASMcode_docs/formats/casm/clex/ClexBasis/>`_.
        """  # noqa
        prim = self._prim
        clusters = self.clusters
        functions = self.functions
        occ_site_functions = self.occ_site_functions

        basis_data = {}
        basis_data["bspecs"] = clex_basis_specs.to_dict()
        basis_data["prim"] = prim.xtal_prim.to_dict()

        basis_data["orbits"] = _make_orbits_data(
            prim=self._prim,
            clex_basis_specs=clex_basis_specs,
            clusters=clusters,
            functions=functions,
            coordinate_mode=coordinate_mode,
        )

        basis_data["site_functions"] = _make_site_functions_data(
            prim=prim,
            occ_site_functions=occ_site_functions,
        )

        return basis_data

    def equivalents_info_dict(self) -> dict:
        R"""Generate the CASM equivalents_info.json data

        Returns
        -------
        equivalents_info_dict: dict
            The equivalents info provides the phenomenal cluster and local-cluster
            orbits for all symmetrically equivalent local-cluster expansions, and the
            indices of the factor group operations used to construct each equivalent
            local cluster expansion from the prototype local-cluster expansion. When
            there is an orientation to the local-cluster expansion this information
            allows generating the proper diffusion events, etc. from the prototype.

            A description of the generated cluster expansion basis set, as described
            `here <TODO>`_.
        """  # noqa

        # Equivalents info, prototype
        equivalents_info = {}

        if self._phenomenal is None:
            return equivalents_info
        if len(self.orbit_matrix_rep_builders) == 0:
            raise Exception(
                "Error in ClusterFunctionsBuilder.equivalents_info_dict: No orbits"
            )

        # Write prim factor group info
        equivalents_info["factor_group"] = (
            config_io.symgroup_to_dict_with_group_classification(
                self._prim, self._prim.factor_group
            )
        )

        # Write equivalents generating ops
        # (actually prim factor group indices of those ops and the
        #  translations can be figured out from the phenomenal cluster)
        orbit_matrix_rep_builder = self.orbit_matrix_rep_builders[0]
        equivalents_info["equivalent_generating_ops"] = (
            orbit_matrix_rep_builder.phenomenal_generating_indices
        )

        # Write prototype orbits info
        tmp = {}
        prototype_phenomenal = self._phenomenal
        clusters = self.clusters
        tmp["phenomenal"] = prototype_phenomenal.to_dict(xtal_prim=self._prim.xtal_prim)
        tmp["prim"] = self._prim.to_dict()
        tmp["orbits"] = orbits_to_dict(
            orbits=clusters,
            prim=self._prim,
        )
        equivalents_info["prototype"] = tmp

        # Write equivalent orbits info
        equivalents_info["equivalents"] = []
        for i_clex, clusters in enumerate(self.equivalent_clusters):
            tmp = {}
            site_rep = orbit_matrix_rep_builder.phenomenal_generating_site_rep[i_clex]
            equiv_phenomenal = site_rep * prototype_phenomenal
            tmp["phenomenal"] = equiv_phenomenal.to_dict(xtal_prim=self._prim.xtal_prim)
            tmp["prim"] = self._prim.to_dict()
            tmp["orbits"] = orbits_to_dict(
                orbits=clusters,
                prim=self._prim,
            )
            equivalents_info["equivalents"].append(tmp)

        return equivalents_info


def make_point_functions(
    prim: casmconfig.Prim,
    prim_neighbor_list: PrimNeighborList,
    orbit: list[casmclust.Cluster],
    orbit_functions: list[list[PolynomialFunction]],
    make_variable_name_f: Optional[Callable] = None,
    local_discrete_dof: Optional[dict] = None,
):
    """Construct point functions

    This method uses the data generated by :class:`ClusterFunctionsBuilder` to
    collect all the functions that involve each point. These functions are used
    by Clexulator methods that can calculate point correlations and changes in point
    correlations.

    Parameters
    ----------
    prim_neighbor_list: libcasm.clexulator.PrimNeighborList
        The :class:`PrimNeighborList` is used to uniquely index sites with local
        variables included in the cluster functions, relative to a reference unit cell.

    orbit: list[libcasm.clusterography.Cluster]
        An orbit of clusters, where ``orbit[i_equiv]``, is the
        `i_equiv`-th symmetrically equivalent cluster in the orbit.
        The order of sites in the clusters may not be arbitrary, it must be
        consistent with the `cluster_site_index` of the :class:`Variable` used
        in the :class:`PolynomialFunction` returned in `orbit_functions`.

        This should generally be one orbit as generated by
        :class:`ClusterFunctionsBuilder`.

    orbit_functions: list[list[casm.bset.polynomial_functions.PolynomialFunction]]
        Polynomial functions, where ``functions[i_equiv][i_func]``,
        is the `i_func`-th function on the cluster given by `orbit[i_equiv]`.

        This should generally be the functions for one orbit as generated by
        :class:`ClusterFunctionsBuilder`.

    make_variable_name_f: Optional[Callable] = None
        Allows specifying a custom class to construct variable names. The default
        class used is :class:`~casm.bset.cluster_functions.MakeVariableName`.
        Custom classes should have the same `__call__` signature as
        :class:`~casm.bset.cluster_functions.MakeVariableName`, and have
        `occ_var_name` and `occ_var_desc` attributes.

    local_discrete_dof: Optionla[list[str]] = None
        The types of local discrete degree of freedom (DoF).

    Returns
    -------
    point_functions: list[list[list[casm.bset.polynomial_functions.PolynomialFunction]]]
        Polynomial functions, where ``point_functions[i_func][nlist_index]`` is
        a list of PolynomialFunction that are symmetrically equivalent to the
        `i_func`-th function on the clusters and involve the `nlist_index`-th site
        in the neighbor list.

    """
    if len(orbit) != len(orbit_functions):
        raise Exception(
            "Error in make_point_functions: "
            "orbit size does not match orbit_functions size"
        )

    if not len(orbit):
        return [[[]]]

    points = []
    for sublattice_index in prim_neighbor_list.sublattice_indices():
        points.append(
            xtal.IntegralSiteCoordinate(
                sublattice=sublattice_index,
                unitcell=[0, 0, 0],
            )
        )

    # orbit_point_functions: list[list[list[PolynomialFunction]]]
    # orbit_point_functions[i_func][nlist_index][i_point_function]
    point_functions = []
    for i_func in range(len(orbit_functions[0])):
        equiv_functions = []
        for nlist_index, point in enumerate(points):
            equiv_functions_by_point = []
            for i_equiv, equiv in enumerate(orbit):
                # for each cluster site on same sublattice as point,
                for site in equiv:
                    if site.sublattice() != point.sublattice():
                        continue
                    f = orbit_functions[i_equiv][i_func].copy()
                    update_variables(
                        prim=prim,
                        cluster=equiv,
                        function=f,
                        prim_neighbor_list=prim_neighbor_list,
                        translation=-site.unitcell(),
                        make_variable_name_f=make_variable_name_f,
                        local_discrete_dof=local_discrete_dof,
                    )
                    # add to point functions
                    equiv_functions_by_point.append(f)
            equiv_functions.append(equiv_functions_by_point)
        point_functions.append(equiv_functions)

    return point_functions


def make_local_point_functions(
    prim_neighbor_list: PrimNeighborList,
    orbit: list[casmclust.Cluster],
    orbit_functions: list[list[PolynomialFunction]],
):
    """Construct local cluster point functions

    This method uses the data generated by :class:`ClusterFunctionsBuilder` to
    collect all the local cluster functions that involve each point. These functions
    are used by Clexulator methods that can calculate point correlations and changes
    in point correlations.

    Parameters
    ----------
    prim_neighbor_list: libcasm.clexulator.PrimNeighborList
        The :class:`PrimNeighborList` is used to uniquely index sites with local
        variables included in the cluster functions, relative to a reference unit cell.

    orbit: list[libcasm.clusterography.Cluster]
        An orbit of clusters, where ``orbit[i_equiv]``, is the
        `i_equiv`-th symmetrically equivalent cluster in the orbit.
        The order of sites in the clusters may not be arbitrary, it must be
        consistent with the `cluster_site_index` of the :class:`Variable` used
        in the :class:`PolynomialFunction` returned in `orbit_functions`.

        This should generally be one orbit as generated by
        :class:`ClusterFunctionsBuilder`.

    orbit_functions: list[list[casm.bset.polynomial_functions.PolynomialFunction]]
        Polynomial functions, where ``functions[i_equiv][i_func]``,
        is the `i_func`-th function on the cluster given by `orbit[i_equiv]`.

        This should generally be the functions for one orbit as generated by
        :class:`ClusterFunctionsBuilder`.

    Returns
    -------
    point_functions: list[list[list[casm.bset.polynomial_functions.PolynomialFunction]]]
        Polynomial functions, where ``point_functions[i_func][nlist_index]`` is
        a list of PolynomialFunction that are symmetrically equivalent to the
        `i_func`-th function on the clusters and involve the `nlist_index`-th site
        in the neighbor list.

    """
    if len(orbit) != len(orbit_functions):
        raise Exception(
            "Error in make_local_point_functions: "
            "orbit size does not match orbit_functions size"
        )

    if not len(orbit):
        return [[[]]]

    orbit_neighbor_indices = set()
    for equiv in orbit:
        for site in equiv:
            nlist_index = prim_neighbor_list.neighbor_index(site)
            orbit_neighbor_indices.add(nlist_index)

    # orbit_point_functions: list[list[list[PolynomialFunction]]]
    # orbit_point_functions[i_func][nlist_index][i_point_function]
    prototype_basis_set = orbit_functions[0]
    point_functions = []
    for i_func in range(len(prototype_basis_set)):
        point_functions.append([])
        for nlist_index in range(max(orbit_neighbor_indices) + 1):
            point_functions[i_func].append([])

    for i_equiv, equiv_functions in enumerate(orbit_functions):
        for site in orbit[i_equiv]:
            nlist_index = prim_neighbor_list.neighbor_index(site)
            for i_func, f in enumerate(equiv_functions):
                point_functions[i_func][nlist_index].append(f)

    return point_functions
