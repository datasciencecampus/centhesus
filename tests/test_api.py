"""Unit tests for the `centhesus.api` module."""

import json
from unittest import mock

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from centhesus import CensusAPI
from centhesus.api import _extract_records_from_observations
from centhesus.constants import (
    API_ROOT,
    AREA_TYPES_BY_POPULATION_TYPE,
    DIMENSIONS_BY_POPULATION_TYPE,
    POPULATION_TYPES,
)

MOCK_URL = "mock://test.com/"


@st.composite
def st_table_queries(draw):
    """Create a set of table query parameters for a test."""

    population_type = draw(st.sampled_from(POPULATION_TYPES))

    area_types_available = AREA_TYPES_BY_POPULATION_TYPE[population_type]
    area_type = draw(st.sampled_from(area_types_available))

    dimensions_available = DIMENSIONS_BY_POPULATION_TYPE[population_type]
    dimensions = draw(
        st.lists(st.sampled_from(dimensions_available), min_size=1, max_size=3)
    )

    return population_type, area_type, dimensions


@st.composite
def st_observations(draw, max_nrows=10):
    """Create a set of observations for a test."""

    _, area_type, dimensions = draw(st_table_queries())

    nrows = draw(st.integers(1, max_nrows))
    observations = []
    for _ in range(nrows):
        observation = {}
        observation["dimensions"] = [
            {"option": draw(st.text()), "option_id": draw(st.text())}
            for _ in dimensions
        ]
        observation["observation"] = draw(st.integers())

        observations.append(observation)

    return observations


@st.composite
def st_records_and_queries(draw, max_nrows=10):
    """Create a set of records and query parameters to go with them."""

    query = *_, dimensions = draw(st_table_queries())

    nrows = draw(st.integers(1, max_nrows))
    records = []
    for _ in range(nrows):
        record = (
            draw(st.text()),
            *(draw(st.text()) for _ in dimensions),
            draw(st.integers()),
        )
        records.append(record)

    return records, *query


def test_init():
    """Test that the `CensusAPI` class can be instantiated correctly."""

    api = CensusAPI()

    assert api._current_data is None
    assert api._current_url is None


@given(st.dictionaries(st.text(), st.text()))
def test_process_response_valid(json):
    """Test a valid response can be processed correctly."""

    api = CensusAPI()

    response = mock.MagicMock()
    response.status_code = 200
    response.json.return_value = json

    data = api._process_response(response)

    assert data == json

    response.json.assert_called_once()


@given(st.one_of((st.integers(max_value=199), st.integers(300))))
def test_process_response_invalid_status_code(status):
    """Test an invalid status code returns no data and a warning."""

    api = CensusAPI()

    response = mock.MagicMock()
    response.status_code = status
    response.body = "foo"

    with pytest.warns(UserWarning, match="Unsuccessful GET from"):
        data = api._process_response(response)

    assert data is None


def test_process_response_invalid_json():
    """Test for valid coded, invalid JSON responses.

    We expect the processor to return no data and a warning.
    """

    api = CensusAPI()

    response = mock.MagicMock()
    response.status_code = 200
    response.json.side_effect = json.JSONDecodeError("foo", "bar", 42)

    with pytest.warns(UserWarning, match="Error decoding data from"):
        data = api._process_response(response)

    assert data is None


@given(st.dictionaries(st.text(), st.text()))
def test_get(json):
    """Test that the API only gives data from successful responses."""

    api = CensusAPI()

    with mock.patch("centhesus.api.requests.get") as get, mock.patch(
        "centhesus.api.CensusAPI._process_response"
    ) as process:
        response = mock.MagicMock()
        get.return_value = response
        process.return_value = json

        data = api.get(MOCK_URL)

    assert api._current_url == MOCK_URL
    assert api._current_data == data
    assert data == json

    get.assert_called_once_with(MOCK_URL, verify=True)
    process.assert_called_once_with(response)


@given(st_table_queries(), st.dictionaries(st.text(), st.text()))
def test_query_table_json(query, json):
    """Test that the table querist makes URLs and returns correctly."""

    population_type, area_type, dimensions = query
    url = (
        f"{API_ROOT}/{population_type}/census-observations"
        f"?area-type={area_type}&dimensions={','.join(dimensions)}"
    )

    api = CensusAPI()

    with mock.patch("centhesus.api.CensusAPI.get") as get:
        get.return_value = json

        data = api._query_table_json(population_type, area_type, dimensions)

    assert data == json

    get.assert_called_once_with(url)


@given(st_observations(), st.booleans())
def test_extract_records_from_observations(observations, use_id):
    """Test the record extractor extracts correctly."""

    records = _extract_records_from_observations(observations, use_id)

    assert isinstance(records, list)

    option = "option_id" if use_id else "option"
    for record, observation in zip(records, observations):
        assert isinstance(record, tuple)
        assert len(record) == len(observation["dimensions"]) + 1

        *dimensions, count = record
        assert count == observation["observation"]
        for i, dimension in enumerate(dimensions):
            assert dimension == observation["dimensions"][i][option]


@given(st_records_and_queries(), st.booleans())
def test_query_table_valid(records_and_query, use_id):
    """Test that the querist can create a data frame."""

    records, population_type, area_type, dimensions = records_and_query

    api = CensusAPI()

    with mock.patch(
        "centhesus.api.CensusAPI._query_table_json"
    ) as query, mock.patch(
        "centhesus.api._extract_records_from_observations"
    ) as extract:
        query.return_value = {"observations": "foo"}
        extract.return_value = records

        data = api.query_table(population_type, area_type, dimensions, use_id)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == len(records)

    expected_columns = [area_type, *dimensions, "count", "population_type"]
    assert data.columns.to_list() == expected_columns
    assert (data["population_type"].unique() == population_type).all()

    for i, row in data.drop("population_type", axis=1).iterrows():
        assert tuple(row) == records[i]

    query.assert_called_once_with(population_type, area_type, dimensions)
    extract.assert_called_once_with("foo", use_id)
