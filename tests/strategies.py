"""Custom strategies for testing the package."""

import itertools
import math
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
from census21api.constants import (
    AREA_TYPES_BY_POPULATION_TYPE,
    DIMENSIONS_BY_POPULATION_TYPE,
    POPULATION_TYPES,
)
from hypothesis import strategies as st
from mbi import Domain

from centhesus import MST


def mocked_mst(population_type, area_type, dimensions, domain=None):
    """Create an instance of MST with mocked `get_domain`."""

    with mock.patch("centhesus.mst.MST.get_domain") as get_domain:
        get_domain.return_value = domain
        mst = MST(population_type, area_type, dimensions)

    get_domain.assert_called_once_with()

    return mst


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


@st.composite
def st_single_marginals(draw, kind=None):
    """Create a marginal table and its parameters for a test."""

    population_type, area_type, dimensions = draw(st_api_parameters())

    min_size, max_size = 1, 2
    if kind == "single":
        max_size = 1
    if kind == "pair":
        min_size = 2

    clique = draw(
        st.sets(
            st.sampled_from((area_type, *dimensions)),
            min_size=min_size,
            max_size=max_size,
        ).map(tuple)
    )

    num_uniques = [draw(st.integers(2, 5)) for _ in clique]
    num_rows = int(np.prod(num_uniques))
    counts = draw(
        st.lists(st.integers(0, 100), min_size=num_rows, max_size=num_rows)
    )

    marginal = pd.DataFrame(
        itertools.product(*(range(num_unique) for num_unique in num_uniques)),
        columns=clique,
    )
    marginal["count"] = counts

    return population_type, area_type, dimensions, clique, marginal


@st.composite
def st_domains(draw):
    """Create a domain and its parameters for a test."""

    population_type, area_type, dimensions = draw(st_api_parameters())

    num = len(dimensions) + 1
    sizes = draw(st.lists(st.integers(2, 10), min_size=num, max_size=num))
    domain = Domain.fromdict(dict(zip((area_type, *dimensions), sizes)))

    return population_type, area_type, dimensions, domain


@st.composite
def st_importances(draw):
    """Create a domain and set of importances for a test."""

    population_type, area_type, dimensions, domain = draw(st_domains())

    num = len(domain)
    importances = draw(
        st.lists(
            st.floats(max_value=0, allow_infinity=False, allow_nan=False),
            min_size=math.comb(num, 2),
            max_size=math.comb(num, 2),
        )
    )

    return population_type, area_type, dimensions, domain, importances


@st.composite
def st_subgraphs(draw):
    """Create a subgraph and its parameters for a test."""

    population_type, area_type, dimensions, domain = draw(st_domains())

    edges = draw(
        st.sets(st.sampled_from(list(itertools.combinations(domain, 2))))
    )
    graph = nx.Graph()
    graph.add_edges_from(edges)

    return population_type, area_type, dimensions, domain, graph


@st.composite
def st_group_marginals(draw):
    """Create a group and matching marginal for a test."""

    num_rows_in_group = draw(st.integers(10, 50))
    group = pd.DataFrame({"a": [0] * num_rows_in_group}).groupby("a")

    marginal = draw(
        st.lists(
            st.integers(0, 3),
            min_size=num_rows_in_group,
            max_size=num_rows_in_group,
        )
    )

    return group, marginal
