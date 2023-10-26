"""Unit tests for the `centhesus.api` module."""

import json
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from centhesus import CensusAPI

MOCK_URL = "mock://test.com/"
CensusAPI._root = MOCK_URL


def test_init():
    """Test that the `CensusAPI` class can be instantiated correctly."""

    api = CensusAPI()

    assert api._root == MOCK_URL
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
    """Test that invalid JSON would return no data and a warning."""

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
