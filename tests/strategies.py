"""Custom strategies for testing the package."""

import pandas as pd
from census21api.constants import (
    AREA_TYPES_BY_POPULATION_TYPE,
    DIMENSIONS_BY_POPULATION_TYPE,
    POPULATION_TYPES,
)
from hypothesis import strategies as st


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
