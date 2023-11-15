"""Unit test(s) for the model fitting in `centhesus.MST`."""

from hypothesis import given
from hypothesis import strategies as st
from mbi import Domain, GraphicalModel
from scipy import sparse

from ..strategies import mocked_mst, st_single_marginals


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
