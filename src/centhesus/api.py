"""Module for connecting to the 2021 Census API."""

import json
import warnings

import pandas as pd
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

        Parameters
        ----------
        url : str
            URL from which to retrieve data.

        Returns
        -------
        data : dict or None
            JSON data from the response of this API call if it is
            successful, and `None` otherwise.
        """

        self._current_url = url
        response = requests.get(url, verify=True)

        data = self._process_response(response)
        self._current_data = data

        return self._current_data

    def _query_table_json(self, population_type, area_type, dimensions):
        """
        Retrieve the JSON for a table query from the API.

        Parameters
        ----------
        population_type : str
            Population type to query. See `centhesus.POPULATION_TYPES`.
        area_type : str
            Area type to query.
            See `centhesus.AREA_TYPES_BY_POPULATION_TYPE`.
        dimensions : list of str
            Dimensions to query.
            See `centhesus.DIMENSIONS_BY_POPULATION_TYPE`.

        Returns
        -------
        data : dict or None
            JSON data from the API call if it is successful, and `None`
            otherwise.
        """

        base = "/".join((API_ROOT, population_type, "census-observations"))
        parameters = f"area-type={area_type}&dimensions={','.join(dimensions)}"
        url = "?".join((base, parameters))

        data = self.get(url)

        return data

    def query_table(self, population_type, area_type, dimensions, use_id=True):
        """
        Query and convert a JSON response to a Pandas data frame.

        Parameters
        ----------
        population_type : str
            Population type to query. See `centhesus.POPULATION_TYPES`.
        area_type : str
            Area type to query.
            See `centhesus.AREA_TYPES_BY_POPULATION_TYPE`.
        dimensions : list of str
            Dimensions to query.
            See `centhesus.DIMENSIONS_BY_POPULATION_TYPE`.
        use_id : bool, default True
            If `True` (the default) use the ID for each dimension and
            area type. Otherwise, use the full label.

        Returns
        -------
        data : pandas.DataFrame or None
            Data frame containing the data from the API call if it is
            successful, and `None` otherwise.
        """

        data_dict = self._query_table_json(
            population_type, area_type, dimensions
        )

        if data_dict is None or "observations" not in data_dict:
            return None

        records = _extract_records_from_observations(
            data_dict["observations"], use_id
        )

        columns = (area_type, *dimensions, "count")
        data = pd.DataFrame(records, columns=columns)
        data["population_type"] = population_type

        return data


def _extract_records_from_observations(observations, use_id):
    """
    Extract record information from a set of JSON observations.

    Parameters
    ----------
    observations : dict
        Dictionary of dimension options and count for the observation.
    use_id : bool
        If `True`, use the ID for each dimension option and area type.
        Otherwise, use the full label.

    Returns
    -------
    records : list of tuple
        List of records to be formed into a data frame.
    """

    option = f"option{'_id' * use_id}"

    records = []
    for observation in observations:
        record = (
            *(dimension[option] for dimension in observation["dimensions"]),
            observation["observation"],
        )
        records.append(record)

    return records
