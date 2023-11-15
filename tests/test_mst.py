"""Tests for the `centhesus.mst` module."""

import itertools
import string
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from census21api import CensusAPI
from census21api.constants import (
    DIMENSIONS_BY_POPULATION_TYPE,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from mbi import Domain, GraphicalModel
from scipy import sparse

from centhesus import MST

from .strategies import (
    st_api_parameters,
    st_feature_metadata_parameters,
    st_importances,
    st_single_marginals,
    st_subgraphs,
)


def mocked_mst(population_type, area_type, dimensions, domain=None):
    """Create an instance of MST with mocked `get_domain`."""

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = domain
        mst = MST(population_type, area_type, dimensions)

    get_domain.assert_called_once_with()

    return mst


@given(st_api_parameters())
def test_init(params):
    """Test instantiation of the MST class."""

    population_type, area_type, dimensions = params

    mst = mocked_mst(population_type, area_type, dimensions)

    assert isinstance(mst, MST)
    assert mst.population_type == population_type
    assert mst.area_type == area_type
    assert mst.dimensions == dimensions

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain is None


@given(st_api_parameters())
def test_init_none_area_type(params):
    """Test instantiation of the MST class when area type is None."""

    population_type, _, dimensions = params

    mst = mocked_mst(population_type, None, dimensions)

    assert isinstance(mst, MST)
    assert mst.population_type == population_type
    assert mst.area_type is None
    assert mst.dimensions == dimensions

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain is None


@given(st_api_parameters())
def test_init_none_dimensions(params):
    """Test instantiation of the MST class when dimensions is None."""

    population_type, area_type, _ = params

    mst = mocked_mst(population_type, area_type, None)

    assert isinstance(mst, MST)
    assert mst.population_type == population_type
    assert mst.area_type == area_type
    assert mst.dimensions == DIMENSIONS_BY_POPULATION_TYPE[population_type]

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain is None


@given(st_feature_metadata_parameters())
def test_get_domain_of_feature(params):
    """Test the domain of a feature can be retrieved correctly."""

    population_type, area_type, dimensions, feature, metadata = params

    mst = mocked_mst(population_type, area_type, dimensions)

    with mock.patch("centhesus.mst.CensusAPI.query_feature") as query:
        query.return_value = metadata
        domain = mst._get_domain_of_feature(feature)

    assert isinstance(domain, dict)

    items = [area_type] if feature == "area-types" else dimensions
    assert list(domain.keys()) == metadata["id"].to_list() == items
    assert list(domain.values()) == metadata["total_count"].to_list()

    query.assert_called_once_with(population_type, feature, *items)


@given(st_feature_metadata_parameters())
def test_get_domain_of_feature_none_area_type(params):
    """Test the feature domain getter when area type is None."""

    population_type, _, dimensions, _, metadata = params

    mst = mocked_mst(population_type, None, dimensions)

    with mock.patch("centhesus.mst.CensusAPI.query_feature") as query:
        domain = mst._get_domain_of_feature("area-types")

    assert isinstance(domain, dict)
    assert domain == {}

    query.assert_not_called()


@given(st_api_parameters(), st.text())
def test_get_domain_of_feature_raises_error(params, feature):
    """Test the domain getter raises an error for invalid features."""

    mst = mocked_mst(*params)

    with pytest.raises(ValueError, match="^Feature"):
        mst._get_domain_of_feature(feature)


@given(
    st_api_parameters(),
    st.dictionaries(
        st.text(string.ascii_uppercase, min_size=1), st.integers()
    ),
    st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1), st.integers()
    ),
)
def test_get_domain(params, area_type_domain, dimensions_domain):
    """Test the domain getter can process metadata correctly."""

    mst = mocked_mst(*params)

    with mock.patch("centhesus.mst.MST._get_domain_of_feature") as feature:
        feature.side_effect = [area_type_domain, dimensions_domain]
        domain = mst.get_domain()

    assert isinstance(domain, Domain)
    assert domain.attrs == (
        *area_type_domain.keys(),
        *dimensions_domain.keys(),
    )

    assert feature.call_count == 2
    assert [call.args for call in feature.call_args_list] == [
        ("area-types",),
        ("dimensions",),
    ]


@given(st_single_marginals(), st.booleans())
def test_get_marginal(params, flatten):
    """Test that a marginal table can be processed correctly."""

    population_type, area_type, dimensions, clique, table = params
    mst = mocked_mst(population_type, area_type, dimensions)

    with mock.patch("centhesus.mst.CensusAPI.query_table") as query:
        query.return_value = table
        marginal = mst.get_marginal(clique, flatten)

    if flatten:
        assert isinstance(marginal, np.ndarray)
        assert (marginal == table["count"]).all()
    else:
        assert isinstance(marginal, pd.Series)
        assert marginal.name == "count"
        assert (marginal.reset_index() == table).all().all()

    query.assert_called_once()


@given(st_single_marginals(), st.booleans())
def test_get_marginal_failed_call(params, flatten):
    """Test that a failed call can be processed still."""

    population_type, area_type, dimensions, clique, _ = params
    mst = mocked_mst(population_type, area_type, dimensions)

    with mock.patch("centhesus.mst.CensusAPI.query_table") as query:
        query.return_value = None
        marginal = mst.get_marginal(clique, flatten)

    assert marginal is None

    query.assert_called_once()


@settings(deadline=None)
@given(st_single_marginals(), st.integers(1, 5))
def test_measure(params, num_cliques):
    """Test a set of cliques can be measured."""

    population_type, area_type, dimensions, clique, table = params
    mst = mocked_mst(population_type, area_type, dimensions)

    with mock.patch("centhesus.mst.MST.get_marginal") as get_marginal:
        get_marginal.return_value = table
        measurements = mst.measure([clique] * num_cliques)

    assert isinstance(measurements, list)
    assert len(measurements) == num_cliques

    for measurement in measurements:
        assert isinstance(measurement, tuple)
        assert len(measurement) == 4

        ident, marg, noise, cliq = measurement
        assert isinstance(ident, sparse._dia.dia_matrix)
        assert ident.shape == (marg.size,) * 2
        assert ident.sum() == marg.size
        assert marg.equals(table)
        assert noise == 1e-12
        assert cliq == clique


@given(st_single_marginals(), st.integers(1, 5))
def test_fit_model(params, iters):
    """Test that a model can be fitted to some measurements."""

    population_type, area_type, dimensions, clique, table = params
    domain = Domain.fromdict(table.drop("count", axis=1).nunique().to_dict())
    table = table["count"]
    mst = mocked_mst(population_type, area_type, dimensions, domain=domain)

    measurements = [(sparse.eye(table.size), table, 1e-12, clique)]
    model = mst.fit_model(measurements, iters)

    assert isinstance(model, GraphicalModel)
    assert model.domain == mst.domain
    assert model.cliques == [clique]
    assert model.elimination_order == list(clique)
    assert model.total == table.sum() or 1


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


@settings(deadline=None)
@given(st_importances())
def test_calculate_importances(params):
    """Test that a set of importances can be calculated."""

    population_type, area_type, dimensions, domain, importances = params
    mst = mocked_mst(population_type, area_type, dimensions, domain=domain)

    with mock.patch("centhesus.mst.MST._calculate_importance_of_pair") as calc:
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

    with mock.patch("centhesus.mst.MST.fit_model") as fit, mock.patch("centhesus.mst.MST._calculate_importances") as calc, mock.patch("centhesus.mst.MST._find_maximum_spanning_tree") as find:
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
