"""Unit tests for the `centhesus.api` module."""

from unittest import mock

from hypothesis import given
from hypothesis import strategies as st

from centhesus import CensusAPI

MOCK_URL = "mock://test.com/"


@st.composite
def st_jsons(draw):
    """Create a JSON response for property-based testing."""

    num_items = draw(st.integers(1, 30))
    json = {"items": [{"name": draw(st.text())} for _ in range(num_items)]}

    return json


def test_init():
    """Test that the `CensusAPI` class can be instantiated correctly."""

    api = CensusAPI()

    assert (
        api._root
        == CensusAPI._root
        == "https://api.beta.ons.gov.uk/v1/population-types"
    )

    assert api.populations is None
    assert api.areas_by_population is None
    assert api.dimensions_by_population is None
    assert api._current_data is None


@given(st_jsons(), st.integers(100, 1000))
def test_get(json, status):
    """Test that the API only gives data from successful responses."""

    api = CensusAPI()

    with mock.patch("centhesus.api.requests.get") as requests_get:
        response = mock.MagicMock()
        requests_get.return_value = response
        response.status_code = status
        response.json.return_value = json

        data = api.get(MOCK_URL)

    if 200 <= status <= 299:
        assert data == json
    else:
        assert data is None

    requests_get.assert_called_once_with(MOCK_URL, verify=True)
    response.json.assert_called_once()
