"""Tests for the `centhesus.mst` module."""


from census21api import CensusAPI
from census21api.constants import DIMENSIONS_BY_POPULATION_TYPE
from hypothesis import given

from centhesus import MST

from ..strategies import (
    mocked_mst,
    st_api_parameters,
)


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
