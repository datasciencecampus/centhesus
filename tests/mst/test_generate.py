"""Unit tests for the generation methods of `centhesus.MST`."""

from unittest import mock

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from centhesus import MST

from ..strategies import st_group_marginals


@settings(deadline=None)
@given(
    arrays(
        float,
        st.integers(2, 10),
        elements=st.one_of((st.just(0), st.floats(1, 50))),
    ),
    st.integers(10, 100),
)
def test_synthesise_column(marginal, total):
    """Test a column can be synthesised from a marginal."""

    assume(marginal.sum())

    prng = da.random.default_rng(0)
    column = MST._synthesise_column(marginal, total, prng)

    assert isinstance(column, dd.Series)
    assert dask.compute(*column.shape) == (total,)
    assert column.dtype == int

    uniques, counts = dask.compute(
        *da.unique(column.to_dask_array(lengths=True), return_counts=True)
    )
    if len(uniques) == marginal.size:
        assert np.array_equal(uniques, np.arange(marginal.size))
        assert np.all(counts - marginal * total / marginal.sum() <= 1)
    else:
        assert set(uniques).issubset(range(marginal.size))
        assert np.all(
            counts - marginal[uniques] * total / marginal[uniques].sum() <= 1
        )


@given(st_group_marginals())
def test_synthesise_group(params):
    """
    Test that a dependent column can be synthesised in groups.

    We only test the case where there is a single group currently.
    """

    group, marginal = params

    column, prng = "foo", da.random.default_rng(0)
    with mock.patch("centhesus.mst.MST._synthesise_column") as synth:
        synth.return_value.compute.return_value = marginal
        synthetic = (
            group.copy()
            .groupby("a")
            .apply(MST._synthesise_column_in_group, column, [[]], prng)
        )

    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape[0] == group.shape[0]
    assert synthetic.columns.to_list() == [*group.columns.to_list(), column]

    assert np.array_equal(synthetic[column], marginal)

    synth.assert_called_once_with([], group.shape[0], prng, 1e6)
    synth.return_value.compute.called_once_with()
