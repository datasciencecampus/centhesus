"""Unit tests for the measurement methods in `centhesus.MST`."""

import platform
from unittest import mock

import dask
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import sparse

from ..strategies import mocked_mst, st_single_marginals


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


@pytest.mark.skipif(
    tuple(map(int, platform.python_version_tuple())) < (3, 9),
    reason="Requires Python 3.9+",
)
@settings(deadline=None)
@given(st_single_marginals(), st.integers(1, 5))
def test_measure(params, num_cliques):
    """Test a set of cliques can be measured."""

    population_type, area_type, dimensions, clique, table = params
    mst = mocked_mst(population_type, area_type, dimensions)

    with mock.patch(
        "centhesus.mst.MST.get_marginal"
    ) as get_marginal, dask.config.set(scheduler="synchronous"):
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
        assert noise == 1
        assert cliq == clique
