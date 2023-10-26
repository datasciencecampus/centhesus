"""Module for connecting to the 2021 Census API."""

import json
import warnings

import requests

from centhesus.constants import API_ROOT


class CensusAPI:
    """
    A wrapper for the 2021 Census API.

    Attributes
    ----------
    _current_data : dict or None
        The data dictionary returned by the most recent API call. If no
        call has been made or the last call failed, this is `None`.
    _current_url : str or None
        The URL of the most recent API call. If no call has been made,
        this is `None`.
    """

    def __init__(self):

        self._current_data = None
        self._current_url = None

    def _process_response(self, response):
        """
        Validate and extract data from a response.

        Parameters
        ----------
        response : requests.Response
            Response to be processed.

        Returns
        -------
        data : dict or None
            Data dictionary if the response is valid and `None` if not.
        """

        data = None
        if not 200 <= response.status_code <= 299:
            warnings.warn(
                "\n".join(
                    (
                        f"Unsuccessful GET from {self._current_url}",
                        f"Status code: {response.status_code}",
                        response.body,
                    )
                ),
                UserWarning,
            )
            return data

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            warnings.warn(
                "\n".join(
                    (f"Error decoding data from {self._current_url}:", str(e))
                ),
                UserWarning,
            )

        return data

    def get(self, url):
        """
        Make a call to, and retrieve some data from, the API.

        The only validation we do is to check the status code of the
        response.

        Parameters
        ----------
        url : str
            URL from which to retrieve data.

        Returns
        -------
        data : dict or None
            Data from the response of this API call if it is successful,
            and `None` otherwise.
        """

        self._current_url = url
        response = requests.get(url, verify=True)

        data = self._process_response(response)
        self._current_data = data

        return self._current_data

    def query_table(self, population_type, *dimensions, area_type="nat"):
        """Retrieve the JSON for a table from the API."""

        base = "/".join((API_ROOT, population_type, "census-observations"))
        parameters = f"area-type={area_type}&dimensions={','.join(dimensions)}"
        url = "?".join((base, parameters))

        data = self.get(url)

        return data
