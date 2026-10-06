import json

import numpy as np
import pytest
from utils.helpers import (
    assert_expected_cluster_functions_detailed,
)

import libcasm.configuration as casmconfig
import libcasm.xtal as xtal
from casm.bset import (
    build_cluster_functions,
)
from casm.bset.cluster_functions import (
    make_direct_site_functions,
)

# site basis matrices printed by test_composition_occ_fcc_1a
SUBLAT_0 = [[1.0, 1.0], [1.1338934190276813, -0.37796447300922736]]
SUBLAT_1_TO_3 = [[1.0, 1.0], [-0.37796447300922736, 1.1338934190276813]]


def make_fcc_prim():
    return xtal.Prim(
        lattice=xtal.Lattice(
            column_vector_matrix=np.eye(3),
        ),
        coordinate_frac=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ).T,
        occ_dof=[
            ["A", "B"],
            ["B", "C"],
            ["B", "C"],
            ["B", "C"],
        ],
    )


def test_direct_site_functions_fcc_1_matches_composition(session_shared_datadir):
    """Test that direct site basis functions reproduce the composition cluster functions.

    Uses the site basis matrices from test_composition_occ_fcc_1a.
    """
    xtal_prim = make_fcc_prim()
    builder = build_cluster_functions(
        prim=xtal_prim,
        clex_basis_specs={
            "cluster_specs": {
                "orbit_branch_specs": {
                    "2": {"max_length": 1.01},
                    "3": {"max_length": 1.01},
                },
            },
            "basis_function_specs": {
                "dof_specs": {
                    "occ": {
                        "site_basis_functions": [
                            {"sublat_indices": [0], "value": SUBLAT_0},
                            {"sublat_indices": [1, 2, 3], "value": SUBLAT_1_TO_3},
                        ]
                    }
                }
            },
        },
        verbose=False,
    )
    functions, clusters = (builder.functions, builder.clusters)

    expected_site = {0: SUBLAT_0, 1: SUBLAT_1_TO_3, 2: SUBLAT_1_TO_3, 3: SUBLAT_1_TO_3}
    for b, value in expected_site.items():
        assert np.allclose(builder.occ_site_functions[b]["value"], value)

    with open(
        session_shared_datadir / "expected_occ_site_functions_fcc_1_composition.json"
    ) as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))


def test_direct_site_functions_returns_values_unchanged():
    """Test that direct site functions are not changed.

    `make_direct_site_functions` performs some validity checks,
    but does not construct functions by itself.
    """
    prim = casmconfig.Prim(xtal_prim=make_fcc_prim())
    result = make_direct_site_functions(
        [
            {"sublat_indices": [0], "value": SUBLAT_0},
            {"sublat_indices": [1, 2, 3], "value": SUBLAT_1_TO_3},
        ],
        prim=prim,
    )
    assert [r["sublattice_index"] for r in result] == [0, 1, 2, 3]
    assert np.allclose(result[0]["value"], SUBLAT_0)
    for r in result[1:]:
        assert np.allclose(r["value"], SUBLAT_1_TO_3)


def test_direct_site_functions_wrong_number_of_occupants():
    """Test that including the wrong number of occupants fails.

    The columns of the site basis functions correspond to allowed
    occupants, and must equal the number of possible occupants on
    the sublattice.
    """
    prim = casmconfig.Prim(xtal_prim=make_fcc_prim())
    with pytest.raises(Exception, match="does not match the number of values"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]}],
            prim=prim,
        )
    with pytest.raises(Exception, match="does not match the number of values"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": [[1.0], [0.0]]}],
            prim=prim,
        )


def test_direct_site_functions_wrong_number_of_functions():
    """Test that including the wrong number of functions fails.

    The rows of the site basis functions correspond to the function index,
    and must equal the number of possible occupants on the sublattice in
    order to span the occupation space.
    """
    prim = casmconfig.Prim(xtal_prim=make_fcc_prim())
    with pytest.raises(Exception, match="does not match the number of functions"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]}],
            prim=prim,
        )
    with pytest.raises(Exception, match="does not match the number of functions"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": [[1.0, 1.0]]}],
            prim=prim,
        )


def test_direct_site_functions_no_ones_row():
    """Test that site functions must contain a row of ones."""
    prim = casmconfig.Prim(xtal_prim=make_fcc_prim())
    with pytest.raises(Exception, match="rows of all ones"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": [[1.0, 0.0], [0.0, 1.0]]}],
            prim=prim,
        )


def test_direct_site_functions_two_ones_rows():
    """Test that making two rows all ones fails."""
    prim = casmconfig.Prim(xtal_prim=make_fcc_prim())
    with pytest.raises(Exception, match="rows of all ones"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": [[1.0, 1.0], [1.0, 1.0]]}],
            prim=prim,
        )


def test_direct_site_functions_missing_sublattice():
    """Test that skipping site functions on a sublattice fails."""
    prim = casmconfig.Prim(xtal_prim=make_fcc_prim())
    with pytest.raises(Exception, match="No values provided for sublattice 1"):
        make_direct_site_functions(
            [{"sublat_indices": [0], "value": SUBLAT_0}],
            prim=prim,
        )
