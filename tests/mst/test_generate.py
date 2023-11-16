"""Unit tests for the generation methods of `centhesus.MST`."""

import dask
import dask.array as da
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from centhesus import MST


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

    assert isinstance(column, da.Array)
    assert column.shape == (total,)
    assert column.dtype == int

    uniques, counts = dask.compute(*da.unique(column, return_counts=True))
    if len(uniques) == marginal.size:
        assert np.array_equal(uniques, np.arange(marginal.size))
        assert np.all(counts - marginal * total / marginal.sum() <= 1)
    else:
        assert set(uniques).issubset(range(marginal.size))
        assert np.all(
            counts - marginal[uniques] * total / marginal[uniques].sum() <= 1
        )
