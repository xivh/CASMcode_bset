from typing import Any, Optional

import numpy as np

import libcasm.casmglobal as casmglobal
import libcasm.configuration as casmconfig
import libcasm.xtal as xtal
from casm.bset.misc import almost_equal, almost_int, signof


def make_composition_gram_matrix(
    occ_probs: np.ndarray,
) -> np.ndarray:
    R"""Make the Gram matrix used to construct site basis functions about an average \
    composition specified for each sublattice.

    Notes
    -----

    See the method description
    `here <casm.bset.cluster_functions.make_orthonormal_discrete_functions>`_ for the
    value of the Gram matrix returned by this method.

    Parameters
    ----------
    occ_probs: array_like, shape=(m,)
        Occupation probabilities, organized by occupation index on a site.

    Returns
    -------
    gram_matrix: numpy.ndarray[numpy.float64[m, m]]
        The Gram matrix, :math:`\pmb{G}`, is used to compute site basis functions,
        :math:`\pmb{B}`, that are orthogonalized according to
        :math:`\pmb{B}^{\top} \pmb{G} \pmb{B} = \pmb{I}`. This Gram
        matrix is constructed so that it should yield orthonormality of the Chebychev
        polynomials when the occupation probabilities are equal, and orthonormality of
        the occupation basis when only one probability is non-zero.

    """
    p = np.array(occ_probs)
    if len(p.shape) != 1:
        raise Exception(
            "Error in make_composition_gram_matrix: invalid occ_probs shape"
        )
    n_occupants = p.shape[0]
    G = np.zeros((n_occupants, n_occupants))
    if not almost_equal(np.sum(p), 1.0, abs_tol=casmglobal.TOL):
        raise Exception(
            "Error in _make_composition_gram_matrix: occ_probs does not sum to 1.0"
        )
    for i in range(n_occupants):
        G[i, i] += p[i]
        for j in range(n_occupants):
            G[i, i] += pow((p[i] - p[j]), 2)
            G[i, j] += -pow((p[i] - p[j]), 2)
    return G


def make_orthonormal_discrete_functions(
    occ_probs: np.ndarray,
    abs_tol: float = 1e-10,
):
    R"""Construct orthonormal discrete functions

    Notes
    -----

    - Method by John C. Thomas, as implemented in CASM v1
    - Constructs orthonormal discrete functions, as rows of a matrix,
      :math:`\pmb{\varphi}`, with the properties:

      - Row :math:`i` corresponds to the :math:`i`-th site basis function
      - Element :math:`\varphi_{ij}` is the value of the :math:`i`-th function when the
        site occupation index is the :math:`j`-th possible value.
      - The first row is all ones.
      - In the random alloy at a site composition equal to the input occupation
        probabilities, cluster functions evaluate to 0.
      - In the case of equal occupation probabilities, the discrete Chebychev functions
        are returned.
      - In the case of a single non-zero probability, the "occupation" site basis
        functions are returned.


    The Gram matrix, :math:`\pmb{G}`, is used to compute site basis functions,
    :math:`\pmb{\varphi}`, that are orthogonalized according to

    .. math::

        \pmb{\varphi} \pmb{G} \pmb{\varphi}^{\top} = \pmb{I}

    This Gram matrix is constructed so that it should yield orthonormality of the
    Chebychev polynomials when the occupation probabilities are equal, and
    orthonormality of the occupation basis when only one probability is non-zero.
    The value of the Gram matrix is

    .. math::

        \begin{align}
        G_{ii} &= p_i + \sum_{j \neq i} (p_i - p_j)^{2}, \\
        G_{ij} &= \sum_{j \neq i} -(p_i - p_j)^{2}.
        \end{align}

    Then :math:`\pmb{\varphi}` is calculated as:

    .. math::

        \begin{align}
        \pmb{G} &= \pmb{V} \pmb{S} \pmb{V}^{\top} \\
        \pmb{X} &= \pmb{V} \pmb{S}^{-1/2} \pmb{V}^{\top} \\
        \pmb{X}^{-1} \pmb{Y} &= \pmb{Q} \pmb{R} \\
        \pmb{\varphi} &= (\pmb{X} \pmb{Q})^{\top}
        \end{align}

    Here, :math:`\pmb{Y}` is a "seed" matrix, which is either the discrete Chebychev
    polynomials (if :math:`\mathrm{max}(p_i) < 0.75`), or the "occupation" site basis
    functions (otherwise).

    Finally, the convention for the sign for each basis function (row of
    :math:`\pmb{\varphi}`) is that the last occurrence of the maximum value should by
    positive.


    Parameters
    ----------
    occ_probs: array_like, shape=(m,)
        Occupation probabilities, organized by occupation index on a site.

    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    Returns
    -------
    phi: np.ndarray[np.float64[m, m]]
        The orthonormal discrete function matrix, :math:`\pmb{\varphi}`, a
        :math:`m \times m` matrix, with :math:`m` being the number of occupants allowed
        on the site and element :math:`\varphi_{ij}` being the value of the
        :math:`i`-th function when the site occupation index is the :math:`j`-th
        possible value.

    """
    gram_matrix = make_composition_gram_matrix(occ_probs=occ_probs)

    if not np.allclose(gram_matrix, gram_matrix.T, atol=abs_tol):
        raise Exception(
            "Error in make_orthonormal_discrete_functions: gram_matrix is not symmetric"
        )
    c = np.diag(gram_matrix)
    n_occupants = c.shape[0]

    # ** step 1: find a generic 'phi' matrix
    S, V = np.linalg.eigh(gram_matrix)
    if np.min(S) < -np.abs(abs_tol):
        raise Exception(
            "Error in make_orthonormal_discrete_functions: "
            "gram_matrix is not positive definite"
        )

    # phi_columns is indexed phi_columns[occupant_index, function_index]
    phi_columns = V @ np.diag(np.sqrt(1.0 / S)) @ V.T

    c_max = max(c)
    #  step 2: Make seed basis.
    #  This will be used to seed optimized orientation of 'phi'
    if c_max < 0.75:
        tcos_table = np.zeros((n_occupants, n_occupants))
        for i in range(n_occupants):
            tcos_table[i, 0] = 1.0
            x = np.cos(np.pi * (i + 0.5) / n_occupants)
            for j in range(1, n_occupants):
                tcos_table[i, j] = x * tcos_table[i, j - 1]

        # QR decomposition of tcos_table yields Q matrix that holds chebychev basis
        Q, R = np.linalg.qr(tcos_table)
        tseed = Q
    else:
        # there is an outlier probability --> set seed matrix to occupation basis,
        # with specis 'i==max_ind' as solvent
        curr_i = 0
        tseed = np.zeros((n_occupants, n_occupants))
        c_max_index = np.where(c == c_max)[0]
        for i in range(phi_columns.shape[0]):
            tseed[i, 0] = 1.0
            if i == c_max_index:
                continue
            for j in range(i, phi_columns.shape[1]):
                if curr_i + 1 == j:
                    tseed[i, j] = 1.0
            curr_i += 1

    # ** step 3: use seed matric to find a unitary matrix that rotates 'phi' a more
    # optimal form Assume: tseed = phi * Q, with unitary Q approximate Q by finding
    # QR decomposition of (phi.inverse() * tseed)
    # Eigen::MatrixXd Q = (phi.inverse() * tseed).householderQr().householderQ();

    Q, R = np.linalg.qr(np.linalg.inv(phi_columns) @ tseed)

    # Rotate 'phi'
    phi_columns = phi_columns @ Q

    # Sign convention
    for i in range(phi_columns.shape[1]):
        sign_change = 1
        max_abs = 0.0
        for j in range(phi_columns.shape[0]):
            if np.abs(phi_columns[j, i]) > (max_abs - abs_tol):
                max_abs = np.abs(phi_columns[j, i])
                sign_change = signof(phi_columns[j, i], abs_tol=abs_tol)
        phi_columns[:, i] *= sign_change

    for i in range(phi_columns.shape[0]):
        for j in range(phi_columns.shape[1]):
            if almost_int(phi_columns[i, j], abs_tol=abs_tol):
                phi_columns[i, j] = round(phi_columns[i, j])

    return phi_columns.transpose()


def _is_chebychev_site_functions(site_basis_functions_specs: Any):
    if not isinstance(site_basis_functions_specs, str):
        return False
    return site_basis_functions_specs.lower() == "chebychev"


def make_chebychev_site_functions(
    prim: casmconfig.Prim,
    abs_tol: float = 1e-10,
):
    """Make discrete occupation site functions using the "chebychev" basis, which
    expands about the random alloy.

    Parameters
    ----------
    prim: libcasm.configation.Prim
        The prim, with symmetry information.

    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    Returns
    -------
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions includes:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.

    """
    all_indices = xtal.asymmetric_unit_indices(prim.xtal_prim)
    occ_dofs = prim.xtal_prim.occ_dof()
    site_rep = prim.integral_site_coordinate_symgroup_rep
    indicator_matrix_rep = prim.local_dof_matrix_rep("occ")

    phi = {}
    for unit_indices in all_indices:
        b_init = unit_indices[0]

        n_allowed_occs = len(occ_dofs[b_init])
        if n_allowed_occs < 2:
            continue

        occ_probs = np.array([1.0 / n_allowed_occs] * n_allowed_occs)
        phi[b_init] = make_orthonormal_discrete_functions(occ_probs, abs_tol)

        site_init = xtal.IntegralSiteCoordinate(sublattice=b_init, unitcell=[0, 0, 0])

        for i_factor_group, site_rep_op in enumerate(site_rep):
            site_final = site_rep_op * site_init
            b_final = site_final.sublattice()
            M_init = indicator_matrix_rep[i_factor_group][b_init]

            if b_final not in phi:
                phi[b_final] = phi[b_init] @ M_init.T

    occ_site_functions = []
    for key, value in phi.items():
        occ_site_functions.append({"sublattice_index": key, "value": value.tolist()})

    return occ_site_functions


def _is_occupation_site_functions(site_basis_functions_specs: Any):
    if isinstance(site_basis_functions_specs, dict):
        if "type" in site_basis_functions_specs:
            return site_basis_functions_specs["type"].lower() == "occupation"
        else:
            return False
    if not isinstance(site_basis_functions_specs, str):
        return False
    return site_basis_functions_specs.lower() == "occupation"


def make_occupation_site_functions(
    prim: casmconfig.Prim,
    abs_tol: float = 1e-10,
    reference_occ: list[int] = None,
):
    """Make discrete occupation site functions using the "occupation" basis, which \
    expands about the default configuration with each sublattice occupied by the first \
    occupant listed in the prim.

    Parameters
    ----------
    prim: libcasm.configation.Prim
        The prim, with symmetry information.

    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    reference_occ: list[str] = None
        For each sublattice, the names of the occupant that should only give a non-zero
        value for the constant site function. If not provided, the first occupant
        listed in the prim is used.

    Returns
    -------
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions includes:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.

    """
    all_indices = xtal.asymmetric_unit_indices(prim.xtal_prim)
    occ_dofs = prim.xtal_prim.occ_dof()
    site_rep = prim.integral_site_coordinate_symgroup_rep
    indicator_matrix_rep = prim.local_dof_matrix_rep("occ")

    if reference_occ is None:
        reference_occ = [site_occ_dofs[0] for site_occ_dofs in occ_dofs]

    if len(reference_occ) != len(occ_dofs):
        raise Exception(
            "Error in make_occupation_site_functions: "
            "reference_occ must have the same length as the number of sublattices."
        )
    for b in range(len(occ_dofs)):
        if reference_occ[b] not in occ_dofs[b]:
            raise Exception(
                "Error in make_occupation_site_functions: "
                f"for sublattice {b}, reference_occ[{b}]={reference_occ[b]} "
                "is not an allowed occupant."
            )

    phi = {}
    for unit_indices in all_indices:
        b_init = unit_indices[0]

        n_allowed_occs = len(occ_dofs[b_init])
        if n_allowed_occs < 2:
            continue

        occ_probs = np.zeros((n_allowed_occs,))
        reference_index = occ_dofs[b_init].index(reference_occ[b_init])
        occ_probs[reference_index] = 1.0
        phi[b_init] = make_orthonormal_discrete_functions(occ_probs, abs_tol)

        site_init = xtal.IntegralSiteCoordinate(sublattice=b_init, unitcell=[0, 0, 0])

        for i_factor_group, site_rep_op in enumerate(site_rep):
            site_final = site_rep_op * site_init
            b_final = site_final.sublattice()
            M_init = indicator_matrix_rep[i_factor_group][b_init]

            if b_final not in phi:
                phi[b_final] = phi[b_init] @ M_init.T

    occ_site_functions = []
    for key, value in phi.items():
        occ_site_functions.append({"sublattice_index": key, "value": value.tolist()})

    return occ_site_functions


def _is_composition_site_functions(site_basis_functions_specs: Any):
    if not isinstance(site_basis_functions_specs, list):
        return False
    for site in site_basis_functions_specs:
        if not isinstance(site, dict):
            return False
        if "sublat_indices" not in site:
            return False
        if "composition" not in site:
            return False
    return True


def make_composition_site_functions(
    site_basis_functions_specs: list[dict],
    prim: casmconfig.Prim,
    abs_tol: float = 1e-10,
):
    """Construct site basis functions about an average composition specified for \
    each sublattice.

    Parameters
    ----------
    site_basis_functions_specs: list[dict]
        The average composition on each sublattice with >1 allowed occupant. Example:

        .. code-block:: Python

            [
              { // composition on sublattices 0 and 1, as listed
              in prim
                "sublat_indices": [0, 1],
                "composition": {"A": 0.25, "B": 0.75}
              },
              { // composition on sublattices 2 and 3, as listed
              in prim
                "sublat_indices": [2, 3],
                "composition": {"A": 0.75, "B": 0.25}
              }
            ]

    prim: libcasm.configation.Prim
        The prim, with symmetry information.

    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    Returns
    -------
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions includes:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.
    """

    all_indices = xtal.asymmetric_unit_indices(prim.xtal_prim)
    occ_dofs = prim.xtal_prim.occ_dof()
    site_rep = prim.integral_site_coordinate_symgroup_rep
    indicator_matrix_rep = prim.local_dof_matrix_rep("occ")

    def get_composition(
        site_basis_functions_specs: list[dict],
        unit_indices: list[int],
    ):
        for _specs in site_basis_functions_specs:
            for _b in unit_indices:
                if _b in _specs["sublat_indices"]:
                    return (_b, _specs["composition"])
        return (None, None)

    phi = {}
    for unit_indices in all_indices:
        b_init, composition = get_composition(
            site_basis_functions_specs=site_basis_functions_specs,
            unit_indices=unit_indices,
        )

        if b_init is None:
            raise Exception(
                "Error in make_composition_site_functions: "
                f"No composition found for sublattices {unit_indices}."
            )

        n_allowed_occs = len(occ_dofs[b_init])
        if n_allowed_occs < 2:
            continue

        occ_probs = np.zeros((n_allowed_occs,))
        if len(composition) != n_allowed_occs:
            raise Exception(
                "Error in make_composition_site_functions: "
                f"for sublattice {b_init} "
                f"the number of allowed occupants ({n_allowed_occs}) "
                f"does not match the number of compositions provided."
            )
        for name, value in composition.items():
            if name not in occ_dofs[b_init]:
                raise Exception(
                    "Error in make_composition_site_functions: "
                    f"For sublattice {b_init}, {name} is not an allowed occupant."
                )
            occ_probs[occ_dofs[b_init].index(name)] = value
        if not almost_equal(np.sum(occ_probs), 1.0, abs_tol=casmglobal.TOL):
            raise Exception(
                "Error in make_composition_site_functions: "
                f"For sublattice {b_init}, composition does not sum to 1.0."
            )
        phi[b_init] = make_orthonormal_discrete_functions(occ_probs, abs_tol)

        site_init = xtal.IntegralSiteCoordinate(sublattice=b_init, unitcell=[0, 0, 0])

        for i_factor_group, site_rep_op in enumerate(site_rep):
            site_final = site_rep_op * site_init
            b_final = site_final.sublattice()
            M_init = indicator_matrix_rep[i_factor_group][b_init]

            if b_final not in phi:
                phi[b_final] = phi[b_init] @ M_init.T

    occ_site_functions = []
    for key, value in phi.items():
        occ_site_functions.append({"sublattice_index": key, "value": value.tolist()})

    return occ_site_functions


def _is_direct_site_functions(site_basis_functions_specs: Any):
    if not isinstance(site_basis_functions_specs, list):
        return False
    for site in site_basis_functions_specs:
        if not isinstance(site, dict):
            return False
        if "sublat_indices" not in site:
            return False
        if "value" not in site:
            return False
    return True


def make_direct_site_functions(
    site_basis_functions_specs: list[dict],
    prim: casmconfig.Prim,
    abs_tol: float = 1e-10,
):
    """Construct site basis functions as directly specified for each sublattice.

    .. warning::

        With this method it is possible to incorrectly use site basis functions
        that are not consistent with the symmetry of the prim. It should be
        considered a feature for developers and advanced users who understand
        how to check the results.

    The site basis functions can be directly specified on each sublattice using an array of dict, with the attributes:

    - "value": list[list[float]], Species the site basis function values, :math:`\varphi_{ms}`,
      where the row index, :math:`m`, corresponds to a function index, and the column, :math:`s`,
      is the site occupation index. One row must be the vector of ones.

    - "sublat_indices": list[int], Specifies the sublattices for which the site basis
      function values apply.

    Parameters
    ----------
    site_basis_functions_specs: list[dict]
        The site basis function values, as ``value[function_index][occupant_index]``,
        on each sublattice with >1 allowed occupant. Example:

        .. code-block:: Python

            [
              {
                "sublat_indices": [0, 1],
                "value": [
                  [1., 1., 1.],
                  [0., 1., 0.],
                  [0., 0., 1.],
                ]
              },
              {
                "sublat_indices": [2, 3],
                "value": [
                  [1., 0., 0.],
                  [0., 1., 0.],
                  [1., 1., 1.],
                ]
              }
            ]

    prim: libcasm.configation.Prim
        The prim, with symmetry information.

    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    Returns
    -------
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions includes:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.
    """
    occ_site_functions = []
    for i_sublat, site_occ_dof in enumerate(prim.xtal_prim.occ_dof()):
        if len(site_occ_dof) < 2:
            continue
        occ_site_functions.append(
            {
                "sublattice": i_sublat,
            }
        )

    occ_dofs = prim.xtal_prim.occ_dof()

    found_sublat_indices = set()
    _occ_site_functions = {}
    """dict[int,numpy.ndarray]: Sublattice index -> site functions"""

    for i_site, site in enumerate(site_basis_functions_specs):
        values = np.array(site["value"], dtype=float)
        if len(values.shape) != 2:
            raise Exception(
                "Error in make_direct_site_functions: "
                "value must be a list of lists, as value[function_index][occupant_index]."
            )
        # enforce that exactly one row is all ones
        all_ones = np.all(np.isclose(values, 1.0, atol=abs_tol), axis=1)
        n_all_ones = int(np.sum(all_ones))
        if n_all_ones != 1:
            raise Exception(
                f"Error in make_direct_site_functions: "
                f"site {i_site} has {n_all_ones} rows of all ones, "
                f"expected one row of all ones."
            )
        for i_sublat in site["sublat_indices"]:
            found_sublat_indices.add(i_sublat)
            n_allowed_occs = len(occ_dofs[i_sublat])
            if values.shape[1] != n_allowed_occs:
                raise Exception(
                    "Error in make_direct_site_functions: "
                    f"for sublattice {i_sublat} "
                    f"the number of allowed occupants ({n_allowed_occs}) "
                    f"does not match the number of values per function "
                    f"({values.shape[1]})."
                )
            if values.shape[0] != n_allowed_occs:
                raise Exception(
                    "Error in make_direct_site_functions: "
                    f"for sublattice {i_sublat} "
                    f"the number of allowed occupants ({n_allowed_occs}) "
                    f"does not match the number of functions ({values.shape[0]})."
                )
            _occ_site_functions[i_sublat] = values

    # check that all sublattices with >1 occupant were specified
    for i_sublat, site_occ_dofs in enumerate(occ_dofs):
        if len(site_occ_dofs) > 1 and i_sublat not in found_sublat_indices:
            raise Exception(
                "Error in make_direct_site_functions: "
                f"No values provided for sublattice {i_sublat}."
            )

    indices = list(found_sublat_indices)
    indices.sort()
    return [
        {"sublattice_index": i, "value": _occ_site_functions[i].tolist()}
        for i in indices
    ]


def make_occ_site_functions(
    prim: casmconfig.Prim,
    occ_site_basis_functions_specs: Any,
    abs_tol: float = 1e-10,
) -> list[dict]:
    """Make discrete occupation site functions from `dof_specs` input

    Parameters
    ----------
    prim: libcasm.configation.Prim
        The prim, with symmetry information.

    occ_site_basis_functions_specs: Any
        Provides instructions for constructing occupation site basis functions. As
        described in detail in the section
        :ref:`DoF Specifications <sec-dof-specifications>`, the options are:

        - "chebychev": Chebychev site basis functions give an expansion (with
          correlation values all equal to `0`) about the idealized random alloy where
          the probability of any of the allowed occupants on a particular site is the
          same.
        - "occupation": The "occupation" site basis functions give an expansion (with
          correlation values all equal to `0`) about the default configuration where
          each site is occupied by the first allowed occupant in the prim
          :func:`~libcasm.xtal.Prim.occ_dof` list.
        - `{"type": "occupation", ...}`: An alternative way to specify "occupation"
          site basis functions that allows specifying a "reference_occ" list of occupant
          names to use as the reference occupant for each sublattice. If
          "reference_occ" is not provided, the first occupant listed in the prim is
          used. The reference occupant must the same on all symmetrically equivalent
          sites. For example, an FCC A-B-C binary could use any of the following:

          .. code-block:: Python

              A_ref_specs = {
                  "type": "occupation",
                  "reference_occ": ["A"],
              }

              B_ref_specs = {
                  "type": "occupation",
                  "reference_occ": ["B"],
              }

              C_ref_specs = {
                  "type": "occupation",
                  "reference_occ": ["C"],
              }

        - `list[dict]`: A list of dictionaries in one of the following formats:

          - Composition-centered basis functions, which give an expansion (with
            correlation values all equal to `0`) for the idealized random configuration
            with a particular composition for each asymmetric unit. Expected format
            for the dictionaries is:

            .. code-block:: Python

                {
                    "sublat_indices": [0, 1],
                    "composition": {"A": 0.25, "B": 0.75},
                }

            Only one sublattice in each asymmetric unit with >1 allowed occupant is
            required to be given. The specified composition is used to construct
            discrete basis functions on one site and then symmetry is used to
            construct an equivalent basis on other sites in the asymmetric unit. For
            anisotropic occupants there may be multiple ways consistent with the prim
            factor group to construct the site basis functions on other sites (i.e.
            the choice of which spin state or molecular orientation of an occupant gets
            site basis function values of -1 or +1 may be arbitrary as long as it is
            done consistently). The particular choice made is based on the order in
            which symmetry operations are sorted in the prim factor group and should
            be consistent for a particular choice of prim. An exception will be raised
            if sublattices in different asymmetric units are incorrectly grouped, or if
            no site is given for an asymmetric unit with >1 allowed occupant.

          - Directly set basis functions values.

            .. warning::

                With this method it is possible to incorrectly use site basis functions
                that are not consistent with the symmetry of the prim. It should be
                considered a feature for developers and advanced users who understand
                how to check the results.

            The expected format for the dictionaries is:

            .. code-block:: Python

                {
                    "sublat_indices": [0, 1],
                    "value":  [
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [1., 1., 1.],
                    ]
                }

            where site basis function values are specified using
            `value[function_index][occupant_index]`, the `function_index` being the
            site basis function index on the site, and the `occupant_index` being the
            index of the occupant in the prim :func:`~libcasm.xtal.Prim.occ_dof` list
            for the site.


    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    Returns
    -------
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.

    """
    x = occ_site_basis_functions_specs

    if _is_chebychev_site_functions(x):
        return make_chebychev_site_functions(prim=prim, abs_tol=abs_tol)
    elif _is_occupation_site_functions(x):
        reference_occ = None
        if isinstance(x, dict):
            reference_occ = x.get("reference_occ", None)
        return make_occupation_site_functions(
            prim=prim,
            abs_tol=abs_tol,
            reference_occ=reference_occ,
        )
    elif _is_composition_site_functions(x):
        return make_composition_site_functions(
            site_basis_functions_specs=x, prim=prim, abs_tol=abs_tol
        )
    elif _is_direct_site_functions(x):
        return make_direct_site_functions(
            site_basis_functions_specs=x, prim=prim, abs_tol=abs_tol
        )

    raise Exception(
        "Error in make_occ_site_functions: "
        "Invalid dof_specs/occ/site_basis_functions value"
    )


def get_occ_site_functions(
    occ_site_functions: list[dict],
    sublattice_index: int,
    site_function_index: Optional[int] = None,
):
    """Get a specified occupation site function

    Parameters
    ----------
    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.

    sublattice_index: int
        The sublattice to get a site function for.

    site_function_index: Optional[int] = None
        The particular site function to get. If None, get all site functions.

    Returns
    -------
    phi: np.ndarray
        If `site_function_index` is None, returns the `shape=(n_occupants, n_occupants)`
        array with rows representing site functions and columns representing occupant
        index, if it exists for the specified sublattice; else returns a `shape=(0,0)`
        array. If `site_function_index` is not None, returns the `shape=(n_occupants,)`
        array representing the `site_function_index`-th site function, with indices
        representing occupant index, on the specified sublattice.
    """
    for site_funcs in occ_site_functions:
        if site_funcs["sublattice_index"] == sublattice_index:
            site_functions = np.array(site_funcs["value"])
            if site_function_index is None:
                return site_functions
            elif site_function_index < 0:
                raise Exception(
                    "Error in get_occ_site_functions: "
                    f"invalid site_function_index={site_function_index}"
                )
            elif site_function_index < site_functions.shape[0]:
                return site_functions[site_function_index, :]
            else:
                raise Exception(
                    "Error in get_occ_site_functions: "
                    f"invalid site_function_index={site_function_index}"
                )
    return np.zeros((0, 0))


def make_occ_site_functions_info(
    prim: casmconfig.Prim,
    occ_site_functions: list[dict],
    abs_tol: float = 1e-10,
) -> dict:
    """Get info used for naming the occupation variables in latex

    Parameters
    ----------
    prim: libcasm.configation.Prim
        The prim, with symmetry information.

    occ_site_functions: list[dict]
        List of occupation site basis functions. For each sublattice with discrete
        site basis functions, must include:

        - `"sublattice_index"`: int, index of the sublattice
        - `"value"`: list[list[float]], list of the site basis function values, as
          ``value = functions[function_index][occupant_index]``.

    abs_tol: float = 1e-10
        A absolute tolerance for comparing values.

    Returns
    -------
    occ_site_functions_info: dict
        Occupation site basis functions info, with format:

        - `"max_function_index"`: int, The maximum site function index, across all
          sublattices.
        - `"all_sublattices_have_same_site_functions"`: bool, True if all _sublattices
          have same site functions; False otherwise.
        - `"occ_var_name"`: str, A variable name template for the site functions,
          which may be formated using `b` for sublattice index and `m` for site function
          index (i..e ``occ_var_name.format(b=0, m=1)``).
        - `"occ_var_desc": occ_var_desc, A description of the occupation
          variable, including a description of the subscript indices.
        - `"occ_var_indices"`: list[list[str, str]], A list of lists, where each sublist
          contains the variable name and description for each subscript index.
    """
    max_function_index = 0
    all_sublattices_have_same_site_functions = True
    phi_0 = None
    occ_dof_0 = None
    occ_dof = prim.xtal_prim.occ_dof()
    for i, site_funcs in enumerate(occ_site_functions):
        phi = np.array(site_funcs["value"])
        b = site_funcs["sublattice_index"]
        if phi.shape[0] - 1 > max_function_index:
            max_function_index = phi.shape[0] - 1
        if i == 0:
            phi_0 = np.array(site_funcs["value"])
            occ_dof_0 = occ_dof[b]
        else:
            if not phi.shape == phi_0.shape:
                all_sublattices_have_same_site_functions = False
            elif not np.allclose(phi, phi_0, atol=abs_tol):
                all_sublattices_have_same_site_functions = False
            elif occ_dof[b] != occ_dof_0:
                all_sublattices_have_same_site_functions = False

    if max_function_index < 2:
        if all_sublattices_have_same_site_functions is True:
            occ_var_name = "\\phi"
            occ_var_desc = "Occupation site function"
            occ_var_indices = []
        else:
            occ_var_name = "\\phi_{{{b}}}"
            occ_var_desc = "$\\phi_{b}$, where $b$ is the sublattice index"
            occ_var_indices = [
                ["b", "sublattice index"],
            ]
    else:
        if all_sublattices_have_same_site_functions is True:
            occ_var_name = "\\phi_{{{m}}}"
            occ_var_desc = "$\\phi_{m}$, where $m$ is the site function index"
            occ_var_indices = [
                ["m", "site function index"],
            ]
        else:
            occ_var_name = "\\phi_{{{b},{m}}}"
            occ_var_desc = (
                "$\\phi_{{{b},{m}}}$, where "
                "$b$ is the sublattice index and "
                "$m$ is the site function index"
            )
            occ_var_indices = [
                ["b", "sublattice index"],
                ["m", "site function index"],
            ]

    return {
        "max_function_index": max_function_index,
        "all_sublattices_have_same_site_functions": all_sublattices_have_same_site_functions,  # noqa: E501
        "occ_var_name": occ_var_name,
        "occ_var_desc": occ_var_desc,
        "occ_var_indices": occ_var_indices,
    }
