"""Tests for the `centhesus.mst` module."""

import string
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from census21api import CensusAPI
from census21api.constants import (
    DIMENSIONS_BY_POPULATION_TYPE,
)
from hypothesis import given
from hypothesis import strategies as st
from mbi import Domain

from centhesus import MST

from .strategies import (
    st_api_parameters,
    st_feature_metadata_parameters,
    st_marginals,
)


def mocked_mst(population_type, area_type, dimensions, return_value=None):
    """Create an instance of MST with mocked `get_domain`."""

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = return_value
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


@given(st_marginals(), st.booleans())
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
