"""Unit tests for the `centhesus.api` module."""

from centhesus import CensusAPI


def _test_params_by_pop(params_by_pop, populations):
    """Check a parameter dictionary. Helper function."""

    assert isinstance(params_by_pop, dict)
    assert set(populations) == set(params_by_pop.keys())
    for params in params_by_pop.values():
        assert isinstance(params, list)
        assert all(isinstance(param, str) for param in params)


def test_init():
    """Test that the `CensusAPI` class can be instantiated correctly."""

    api = CensusAPI()

    assert api._root == "https://api.beta.ons.gov.uk/v1/population-types"

    populations = api.populations
    assert isinstance(populations, list)
    assert all(isinstance(pop, str) for pop in populations)

    for params in (api.areas_by_population, api.dimensions_by_population):
        _test_params_by_pop(params, populations)

    assert api._current_data is None
