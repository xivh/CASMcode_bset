import json

# from utils.expected_disp_functions import (
#     expected_occ_functions_fcc_1,
#     expected_occ_functions_hcp_1,
#     expected_occ_functions_lowsym_1,
# )
from utils.helpers import (
    assert_expected_cluster_functions_detailed,
)

import libcasm.clusterography as casmclust
import libcasm.configuration as casmconfig
import libcasm.occ_events as occ_events
import libcasm.xtal.prims as xtal_prims
from casm.bset import (
    build_cluster_functions,
)
from casm.bset.cluster_functions import (
    BasisFunctionSpecs,
    ClexBasisSpecs,
    make_neighborhood,
    make_occevent_cluster_specs,
)


def test_occ_fcc_1a(session_shared_datadir):
    """build_functions with libcasm.xtal.Prim / dict bspecs"""
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B", "C"],
    )
    # print(xtal.pretty_json(xtal_prim.to_dict()))

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
                "dof_specs": {"occ": {"site_basis_functions": "occupation"}}
            },
        },
    )
    functions, clusters = (builder.functions, builder.clusters)

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_fcc_1.json",
    # )
    with open(session_shared_datadir / "expected_occ_functions_fcc_1.json") as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))


def test_occ_fcc_1b(session_shared_datadir):
    """build_functions with libcasm.configuration.Prim / ClexBasisSpecs"""
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B", "C"],
    )

    prim = casmconfig.Prim(xtal_prim)
    builder = build_cluster_functions(
        prim=prim,
        clex_basis_specs=ClexBasisSpecs(
            cluster_specs=casmclust.ClusterSpecs(
                xtal_prim=prim.xtal_prim,
                generating_group=prim.factor_group,
                max_length=[0.0, 0.0, 1.01, 1.01],
            ),
            basis_function_specs=BasisFunctionSpecs(
                dof_specs={"occ": {"site_basis_functions": "occupation"}},
            ),
        ),
    )
    functions, clusters = (builder.functions, builder.clusters)

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_fcc_1.json",
    # )
    with open(session_shared_datadir / "expected_occ_functions_fcc_1.json") as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))


def test_occ_fcc_1c_verbose(session_shared_datadir):
    """build_functions with libcasm.xtal.Prim / dict bspecs"""
    xtal_prim = xtal_prims.FCC(
        r=0.5,
        occ_dof=["A", "B", "C"],
    )
    # print(xtal.pretty_json(xtal_prim.to_dict()))

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
                "dof_specs": {"occ": {"site_basis_functions": "occupation"}}
            },
        },
        verbose=True,
    )
    functions, clusters = (builder.functions, builder.clusters)

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_fcc_1.json",
    # )
    with open(session_shared_datadir / "expected_occ_functions_fcc_1.json") as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))


def test_occ_fcc_local_1(session_shared_datadir):
    xtal_prim = xtal_prims.FCC(
        a=1.0,
        occ_dof=["A", "B", "Va"],
    )
    prim = casmconfig.Prim(xtal_prim)
    # print(xtal.pretty_json(xtal_prim.to_dict()))

    occ_system = occ_events.OccSystem(xtal_prim=prim.xtal_prim)

    occevent = occ_events.OccEvent.from_dict(
        data={
            "trajectories": [
                [
                    {"coordinate": [0, 0, 0, 0], "occupant_index": 0},
                    {"coordinate": [0, 1, 0, 0], "occupant_index": 0},
                ],
                [
                    {"coordinate": [0, 1, 0, 0], "occupant_index": 2},
                    {"coordinate": [0, 0, 0, 0], "occupant_index": 2},
                ],
            ]
        },
        system=occ_system,
    )

    cluster_specs = make_occevent_cluster_specs(
        prim=prim,
        phenomenal_occ_event=occevent,
        max_length=[0.0, 0.0, 1.01],
        cutoff_radius=[0.0, 1.01, 1.01],
    )
    assert cluster_specs.generating_group().head_group is prim.factor_group
    orbits = cluster_specs.make_orbits()

    expected_neighborhood_size = 26
    assert len(make_neighborhood(clusters=orbits)) == expected_neighborhood_size

    builder = build_cluster_functions(
        prim=prim,
        clex_basis_specs=ClexBasisSpecs(
            cluster_specs=cluster_specs,
            basis_function_specs=BasisFunctionSpecs(
                dof_specs={"occ": {"site_basis_functions": "occupation"}},
            ),
        ),
    )
    functions, clusters = (builder.functions, builder.clusters)
    equivalent_functions, equivalent_clusters = (
        builder.equivalent_functions,
        builder.equivalent_clusters,
    )

    expected_n_clex = 6
    assert len(equivalent_functions) == expected_n_clex
    assert len(equivalent_clusters) == expected_n_clex

    assert len(make_neighborhood(clusters=clusters)) == expected_neighborhood_size
    for _clusters in equivalent_clusters:
        assert len(make_neighborhood(clusters=_clusters)) == expected_neighborhood_size

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_fcc_local_1_prototype.json",
    # )
    with open(
        session_shared_datadir / "expected_occ_functions_fcc_local_1_prototype.json"
    ) as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))
    for i_clex, _functions in enumerate(equivalent_functions):
        _clusters = equivalent_clusters[i_clex]
        # print_expected_cluster_functions_detailed(
        #     _functions,
        #     file=pathlib.Path(os.path.realpath(__file__)).parent
        #     / "data"
        #     / f"expected_occ_functions_fcc_local_1_equiv_{i_clex}.json",
        # )
        with open(
            session_shared_datadir
            / f"expected_occ_functions_fcc_local_1_equiv_{i_clex}.json"
        ) as f:
            assert_expected_cluster_functions_detailed(
                _functions, _clusters, json.load(f)
            )


def test_occ_fcc_local_2_verbose(session_shared_datadir):
    xtal_prim = xtal_prims.FCC(
        a=1.0,
        occ_dof=["A", "B", "Va"],
    )
    prim = casmconfig.Prim(xtal_prim)
    # print(xtal.pretty_json(xtal_prim.to_dict()))

    occ_system = occ_events.OccSystem(xtal_prim=prim.xtal_prim)

    occevent = occ_events.OccEvent.from_dict(
        data={
            "trajectories": [
                [
                    {"coordinate": [0, 0, 0, 0], "occupant_index": 0},
                    {"coordinate": [0, 1, 0, 0], "occupant_index": 0},
                ],
                [
                    {"coordinate": [0, 1, 0, 0], "occupant_index": 2},
                    {"coordinate": [0, 0, 0, 0], "occupant_index": 2},
                ],
            ]
        },
        system=occ_system,
    )

    cluster_specs = make_occevent_cluster_specs(
        prim=prim,
        phenomenal_occ_event=occevent,
        max_length=[0.0, 0.0, 1.01],
        cutoff_radius=[0.0, 1.01, 1.01],
    )
    assert cluster_specs.generating_group().head_group is prim.factor_group
    orbits = cluster_specs.make_orbits()

    expected_neighborhood_size = 26
    assert len(make_neighborhood(clusters=orbits)) == expected_neighborhood_size

    builder = build_cluster_functions(
        prim=prim,
        clex_basis_specs=ClexBasisSpecs(
            cluster_specs=cluster_specs,
            basis_function_specs=BasisFunctionSpecs(
                dof_specs={"occ": {"site_basis_functions": "occupation"}},
            ),
        ),
        verbose=True,
    )
    functions, clusters = (builder.functions, builder.clusters)
    equivalent_functions, equivalent_clusters = (
        builder.equivalent_functions,
        builder.equivalent_clusters,
    )

    expected_n_clex = 6
    assert len(equivalent_functions) == expected_n_clex
    assert len(equivalent_clusters) == expected_n_clex

    assert len(make_neighborhood(clusters=clusters)) == expected_neighborhood_size
    for _clusters in equivalent_clusters:
        assert len(make_neighborhood(clusters=_clusters)) == expected_neighborhood_size

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_fcc_local_1_prototype.json",
    # )
    with open(
        session_shared_datadir / "expected_occ_functions_fcc_local_1_prototype.json"
    ) as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))
    for i_clex, _functions in enumerate(equivalent_functions):
        _clusters = equivalent_clusters[i_clex]
        # print_expected_cluster_functions_detailed(
        #     _functions,
        #     file=pathlib.Path(os.path.realpath(__file__)).parent
        #     / "data"
        #     / f"expected_occ_functions_fcc_local_1_equiv_{i_clex}.json",
        # )
        with open(
            session_shared_datadir
            / f"expected_occ_functions_fcc_local_1_equiv_{i_clex}.json"
        ) as f:
            assert_expected_cluster_functions_detailed(
                _functions, _clusters, json.load(f)
            )


def test_occ_hcp_1(session_shared_datadir):
    xtal_prim = xtal_prims.HCP(
        r=0.5,
        occ_dof=["A", "B", "C"],
    )

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
                "dof_specs": {"occ": {"site_basis_functions": "chebychev"}}
            },
        },
    )
    functions, clusters = (builder.functions, builder.clusters)

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_hcp_1.json",
    # )
    with open(session_shared_datadir / "expected_occ_functions_hcp_1.json") as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))


def test_occ_lowsym_1(lowsym_occ_prim, session_shared_datadir):
    xtal_prim = lowsym_occ_prim

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
                "dof_specs": {"occ": {"site_basis_functions": "chebychev"}}
            },
        },
    )
    functions, clusters = (builder.functions, builder.clusters)

    # import os
    # import pathlib
    # from utils.helpers import print_expected_cluster_functions_detailed
    #
    # print_expected_cluster_functions_detailed(
    #     functions,
    #     file=pathlib.Path(os.path.realpath(__file__)).parent
    #     / "data"
    #     / "expected_occ_functions_lowsym_1.json",
    # )
    with open(session_shared_datadir / "expected_occ_functions_lowsym_1.json") as f:
        assert_expected_cluster_functions_detailed(functions, clusters, json.load(f))
