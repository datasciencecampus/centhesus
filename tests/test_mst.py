"""Tests for the `centhesus.mst` module."""

from unittest import mock

import pandas as pd
from census21api import CensusAPI
from census21api.constants import (
    AREA_TYPES_BY_POPULATION_TYPE,
    DIMENSIONS_BY_POPULATION_TYPE,
    POPULATION_TYPES,
)
from hypothesis import given
from hypothesis import strategies as st

from centhesus import MST


@st.composite
def st_api_parameters(draw):
    """Create a valid set of Census API parameters."""

    population_type = draw(st.sampled_from(POPULATION_TYPES))
    area_type = draw(
        st.sampled_from(AREA_TYPES_BY_POPULATION_TYPE[population_type]),
    )
    dimensions = draw(
        st.sets(
            st.sampled_from(DIMENSIONS_BY_POPULATION_TYPE[population_type]),
            min_size=1,
        ).map(sorted),
    )

    return population_type, area_type, dimensions


@st.composite
def st_feature_metadata_parameters(draw):
    """Create a parameter set and feature metadata for a test."""

    population_type, area_type, dimensions = draw(st_api_parameters())

    feature = draw(st.sampled_from(("area-types", "dimensions")))
    items = [area_type] if feature == "area-types" else dimensions
    metadata = pd.DataFrame(
        ((item, draw(st.integers())) for item in items),
        columns=("id", "total_count"),
    )

    return population_type, area_type, dimensions, feature, metadata


@given(st_api_parameters())
def test_init(params):
    """Test instantiation of the MST class."""

    population_type, area_type, dimensions = params

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = "domain"
        mst = MST(population_type, area_type, dimensions)

    assert isinstance(mst, MST)
    assert mst.population_type == population_type
    assert mst.area_type == area_type
    assert mst.dimensions == dimensions

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain == "domain"

    get_domain.assert_called_once_with()


@given(st_api_parameters())
def test_init_none_area_type(params):
    """Test instantiation of the MST class when area type is None."""

    population_type, _, dimensions = params

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = "domain"
        mst = MST(population_type, None, dimensions)

    assert isinstance(mst, MST)
    assert mst.population_type == population_type
    assert mst.area_type is None
    assert mst.dimensions == dimensions

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain == "domain"

    get_domain.assert_called_once_with()


@given(st_api_parameters())
def test_init_none_dimensions(params):
    """Test instantiation of the MST class when dimensions is None."""

    population_type, area_type, _ = params

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = "domain"
        mst = MST(population_type, area_type, None)

    assert isinstance(mst, MST)
    assert mst.population_type == population_type
    assert mst.area_type == area_type
    assert mst.dimensions == DIMENSIONS_BY_POPULATION_TYPE[population_type]

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain == "domain"

    get_domain.assert_called_once_with()


@given(st_feature_metadata_parameters())
def test_get_domain_of_feature(params):
    """Test the domain of a feature can be retrieved correctly."""

    population_type, area_type, dimensions, feature, metadata = params

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = "domain"
        mst = MST(population_type, area_type, dimensions)

    with mock.patch("centhesus.mst.CensusAPI.query_feature") as query:
        query.return_value = metadata
        domain = mst._get_domain_of_feature(feature)

    assert isinstance(domain, dict)

    items = [area_type] if feature == "area-types" else dimensions
    assert list(domain.keys()) == items
    assert list(domain.values()) == metadata["total_count"].to_list()

    query.assert_called_once_with(population_type, feature, *items)
    get_domain.assert_called_once_with()
