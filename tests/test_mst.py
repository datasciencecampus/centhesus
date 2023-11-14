"""Tests for the `centhesus.mst` module."""

from unittest import mock

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
        st.sampled_from(AREA_TYPES_BY_POPULATION_TYPE[population_type])
    )
    dimensions = draw(
        st.one_of(
            (
                st.just(None),
                st.sets(
                    st.sampled_from(
                        DIMENSIONS_BY_POPULATION_TYPE[population_type]
                    ),
                    min_size=1,
                ),
            )
        )
    )

    return population_type, area_type, dimensions


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

    if dimensions is None:
        assert mst.dimensions == DIMENSIONS_BY_POPULATION_TYPE[population_type]
    else:
        assert mst.dimensions == dimensions

    assert isinstance(mst.api, CensusAPI)
    assert mst.domain == "domain"

    get_domain.assert_called_once_with()
