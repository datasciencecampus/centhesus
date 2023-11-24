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

from ..strategies import st_existing_new_columns


@given(
    st.floats(1, 100),
    st.lists(st.text(), min_size=1, max_size=10),
    st.lists(st.tuples(st.text(), st.text()), min_size=1, max_size=5),
    st.one_of((st.just(None), st.integers(1, 100))),
    st.integers(0, 10),
)
def test_setup_generate(total, elimination_order, cliques_, nrows, seed):
    """Test that generation can be set up correctly."""

    model = mock.MagicMock()
    model.total = total
    model.elimination_order = elimination_order
    model.cliques = cliques_

    nrows, prng, cliques, column, order = MST._setup_generate(
        model, nrows, seed
    )

    assert isinstance(nrows, int)
    assert nrows == total or int(model.total)
    assert isinstance(prng, da.random.Generator)
    assert cliques == [set(clique) for clique in cliques_]
    assert column == elimination_order[-1]
    assert order == elimination_order[-2::-1]


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


@given(st_existing_new_columns())
def test_synthesise_column_in_group(params):
    """Test that a dependent column can be synthesised in groups."""

    existing, new = params

    num_groups = existing["a"].nunique()
    column, prng = "foo", da.random.default_rng(0)
    empty_marginal = [[]] * num_groups

    with mock.patch("centhesus.mst.MST._synthesise_column") as synth:
        synth.return_value = new
        synthetic = (
            existing.copy()
            .groupby("a")
            .apply(
                MST._synthesise_column_in_group, column, empty_marginal, prng
            )
        )

    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape[0] == existing.shape[0]
    assert synthetic.columns.to_list() == [*existing.columns.to_list(), column]

    assert np.array_equal(synthetic[column], new * num_groups)

    assert synth.call_count == num_groups
    for i, call in enumerate(synth.call_args_list):
        assert call.args == ([], (existing["a"] == i).sum(), prng, 1e6)

    assert synth.call_count == num_groups


@settings(deadline=None)
@given(
    arrays(
        int,
        st.integers(2, 10),
        elements=st.integers(0, 50),
    ),
    st.text(min_size=1),
    st.integers(10, 100),
)
def test_synthesise_first_column(values, column, nrows):
    """Test that a single column frame can be created."""

    prng = da.random.default_rng(0)
    model = mock.MagicMock()
    model.project.return_value.datavector.return_value = "marginal"

    with mock.patch("centhesus.mst.MST._synthesise_column") as synth:
        synth.return_value = dd.from_array(values)
        first = MST._synthesise_first_column(model, column, nrows, prng)

    assert isinstance(first, dd.DataFrame)
    assert first.columns.to_list() == [column]
    assert np.array_equal(first[column].compute(), values)

    model.project.assert_called_once_with([column])
    model.project.return_value.datavector.called_once_with(flatten=False)
    synth.assert_called_once_with("marginal", nrows, prng)
