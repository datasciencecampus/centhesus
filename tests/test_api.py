"""Unit tests for the `centhesus.api` module."""

from unittest import mock

from hypothesis import given
from hypothesis import strategies as st

from centhesus import CensusAPI

MOCK_URL = "mock://test.com/"
CensusAPI._root = MOCK_URL


@st.composite
def st_names(draw):
    """Create a name for an item in a JSON response."""

    parts = draw(st.lists(st.text(min_size=1), min_size=5, max_size=10))
    name = "-".join(parts)

    return name


@st.composite
def st_jsons(draw):
    """Create a JSON response for property-based testing."""

    num_items = draw(st.integers(1, 10))
    items = [{"name": draw(st_names())} for _ in range(num_items)]
    json = {"items": items}

    return json


def test_init():
    """Test that the `CensusAPI` class can be instantiated correctly."""

    api = CensusAPI()

    assert api._root == MOCK_URL
    assert api._current_data is None
    assert api._current_url is None


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
