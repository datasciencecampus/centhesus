"""Module for connecting to the 2021 Census API."""

import requests


class CensusAPI:
    """
    A wrapper for the 2021 Census API.

    Attributes
    ----------
    _root : pathlib.Path
        The URL to the 2021 Census API endpoint.
    _current_data : dict or None
        The data dictionary returned by the most recent API call. If no
        call has been made or the last call failed, this is `None`.
    populations : list[str]
        Population types available.
    areas_by_population : dict[str, list]
        Mapping of population types to available area types.
    dimensions_by_population : dict[str, list]
        Mapping of population types to available dimensions (columns).
    """

    _root = "https://api.beta.ons.gov.uk/v1/population-types"

    def __init__(self):

        self.populations = self._get_population_types()
        self.areas_by_population = self._get_areas_by_population()
        self.dimensions_by_population = self._get_dimensions_by_population()

        self._current_data = None

    def get(self, url):
        """
        Make a call to, and retrieve some data from, the API.

        The only validation we do is to check the status code of the
        response.

        Parameters
        ----------
        url : str
            URL from which to retrieve data.

        Attributes
        ----------
        _current_data : dict or None
            Data from the response of this API call if it is successful,
            and `None` otherwise.

        Returns
        -------
        _current_data : dict or None
            Data from the response of this API call if it is successful,
            and `None` otherwise.
        """

        response = requests.get(url, verify=True)
        data = response.json()

        code_is_valid = 200 <= response.status_code <= 299
        self._current_data = data if code_is_valid else None

        return self._current_data

    def _get_population_types(self):
        """Retrieve all available population types from the API."""

    def _get_areas_by_population(self):
        """Map each population to their available area types."""

    def _get_dimensions_by_population(self):
        """Map each population to their available dimensions."""
