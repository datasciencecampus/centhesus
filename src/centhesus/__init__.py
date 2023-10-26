"""Synthesising the 2021 England and Wales Census with public data."""


from .api import CensusAPI
from .constants import (
    API_ROOT,
    AREA_TYPES_BY_POPULATION_TYPE,
    DIMENSIONS_BY_POPULATION_TYPE,
    POPULATION_TYPES,
)

__version__ = "0.0.1"

__all__ = [
    "API_ROOT",
    "AREA_TYPES_BY_POPULATION_TYPE",
    "CensusAPI",
    "DIMENSIONS_BY_POPULATION_TYPE",
    "POPULATION_TYPES",
    "__version__",
]
