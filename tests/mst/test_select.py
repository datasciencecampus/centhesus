"""Unit tests for the selection methods in `centhesus.MST`."""

import itertools
import platform
from unittest import mock

import networkx as nx
import pytest
from hypothesis import given, settings

from ..strategies import (
    mocked_mst,
    st_importances,
    st_single_marginals,
    st_subgraphs,
)


@given(st_single_marginals(kind="pair"))
def test_calculate_importance_of_pair(params):
    """Test the importance of a column pair can be calculated."""

    population_type, area_type, dimensions, clique, table = params
    table = table["count"]
    mst = mocked_mst(population_type, area_type, dimensions)

    interim = mock.MagicMock()
    interim.project.return_value.datavector.return_value = table.sample(
        frac=1.0
    ).reset_index(drop=True)
    with mock.patch("centhesus.mst.MST.get_marginal") as get_marginal:
        get_marginal.return_value = table
        weight = mst._calculate_importance_of_pair(interim, clique)

    assert isinstance(weight, float)
    assert weight >= 0

    interim.project.assert_called_once_with(clique)
    interim.project.return_value.datavector.assert_called_once_with()
    get_marginal.assert_called_once_with(clique)


@given(st_single_marginals(kind="pair"))
def test_calculate_importance_of_pair_failed_call(params):
    """Test that a failed call doesn't stop importance processing."""

    population_type, area_type, dimensions, clique, _ = params
    mst = mocked_mst(population_type, area_type, dimensions)

    interim = mock.MagicMock()
    with mock.patch("centhesus.mst.MST.get_marginal") as get_marginal:
        get_marginal.return_value = None
        weight = mst._calculate_importance_of_pair(interim, clique)

    assert weight is None

    interim.project.assert_not_called()
    interim.project.return_value.datavector.assert_not_called()
    get_marginal.assert_called_once_with(clique)


@pytest.mark.skipif(tuple(map(int, platform.python_version_tuple())) > (3, 8))
@settings(deadline=None)
@given(st_importances())
def test_calculate_importances(params):
    """Test that a set of importances can be calculated."""

    population_type, area_type, dimensions, domain, importances = params
    mst = mocked_mst(population_type, area_type, dimensions, domain=domain)

    with mock.patch("centhesus.mst.MST._calculate_importance_of_pair") as calc:
        mst._calculate_importances = calc
        calc.side_effect = importances
        weights = mst._calculate_importances("interim")

    pairs = list(itertools.combinations(domain, 2))
    calc.call_count == len(pairs)
    call_args = [call.args for call in calc.call_args_list]
    assert set(call_args) == set(("interim", pair) for pair in pairs)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(pairs)

    pairs_execution_order = [pair for _, pair in call_args]
    for pair, importance in zip(pairs_execution_order, importances):
        assert weights[pair] == importance


@given(st_importances())
def test_find_maximum_spanning_tree(params):
    """Test an MST can be found from a set of importances."""

    *api_params, domain, importances = params
    mst = mocked_mst(*api_params, domain=domain)
    weights = dict(zip(itertools.combinations(domain, 2), importances))

    tree = mst._find_maximum_spanning_tree(weights)

    assert isinstance(tree, nx.Graph)
    assert set(tree.nodes) == set(domain)
    assert set(tree.edges).issubset(weights.keys())
    for edge in tree.edges:
        assert tree.edges[edge]["weight"] == -weights[edge]


@given(st_subgraphs())
def test_select(params):
    """Test that a set of two-way cliques can be found correctly."""

    *api_params, domain, tree = params
    mst = mocked_mst(*api_params, domain=domain)

    with mock.patch("centhesus.mst.MST.fit_model") as fit, mock.patch(
        "centhesus.mst.MST._calculate_importances"
    ) as calc, mock.patch(
        "centhesus.mst.MST._find_maximum_spanning_tree"
    ) as find:
        fit.return_value = "interim"
        calc.return_value = "weights"
        find.return_value = tree
        cliques = mst.select("measurements")

    possible_edges = [set(pair) for pair in itertools.combinations(domain, 2)]
    assert isinstance(cliques, list)
    for clique in cliques:
        assert set(clique) in possible_edges

    fit.assert_called_once_with("measurements", iters=1000)
    calc.assert_called_once_with("interim")
    find.assert_called_once_with("weights")
