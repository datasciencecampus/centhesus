"""Module for the Maximum Spanning Tree generator."""

import dask
from census21api import CensusAPI
from census21api.constants import DIMENSIONS_BY_POPULATION_TYPE as DIMENSIONS
from mbi import Domain, FactoredInference
from scipy import sparse


class MST:
    """
    Data synthesiser based on the Maximum Spanning Tree (MST) method.

    This class uses the principles of the
    [MST method](https://doi.org/10.29012/jpc.778) that won the 2018
    NIST Differential Privacy Synthetic Data Challenge. The original
    method makes use of a formal privacy framework to protect the
    confidentiality of the dataset being synthesised. In our case, we
    use the publicly available tables to create our synthetic data.
    These tables have undergone stringent statistical disclosure control
    to make them safe to be in the public domain.

    As such, we adapt MST by removing the formal privacy mechanisms. We
    do not add noise to the public tables, and we use Kruskal's
    algorithm to find the true maximum spanning tree of the feature
    graph. We still make use of the Private-PGM method to generate the
    graphical model and subsequent synthetic data with a nominal amount
    of noise (1e-10).

    The public tables are drawn from the ONS "Create a custom dataset"
    API, which is accessed via the `census21api` package. See
    `census21api.constants` for details of available population types,
    area types, and dimensions.

    Parameters
    ----------
    population_type : str
        Population type to synthesise. Defaults to usual residents in
        households (`"UR_HH"`).
    area_type : str, optional
        Area type to synthesise. If you wish to include an area type
        column (like local authority) in the final dataset, include it
        here. The lowest recommended level is MSOA because of issues
        handling too-large marginal tables.
    dimensions : list of str, optional
        Dimensions to synthesise. All features (other than an area type)
        you would like in the final dataset. If not specified, all
        available dimensions will be included.

    Attributes
    ----------
    api : census21api.CensusAPI
        Client instance to connect to the 2021 Census API.
    domain : mbi.Domain
        Dictionary-like object defining the domain size of every column
        in the synthetic dataset (area type and dimensions).
    """

    def __init__(
        self, population_type="UR_HH", area_type=None, dimensions=None
    ):
        self.population_type = population_type
        self.area_type = area_type
        self.dimensions = dimensions or DIMENSIONS[self.population_type]

        self.api = CensusAPI()
        self.domain = self.get_domain()

    def _get_domain_of_feature(self, feature):
        """
        Retrieve the domain for items in a feature of the API.

        Parameters
        ----------
        feature : {"area-types", "dimensions"}
            Feature of the API from which to call.

        Raises
        ------
        ValueError
            If `feature` is invalid.

        Returns
        -------
        domain : dict
            Dictionary containing the domain metadata. Empty if
            `feature` is `"area-types"` and `self.area_type` is `None`.
        """

        if feature == "area-types" and self.area_type is None:
            return {}
        elif feature == "area-types":
            items = [self.area_type]
        elif feature == "dimensions":
            items = self.dimensions
        else:
            raise ValueError(
                "Feature must be one of 'area-types' or 'dimensions', "
                f"not '{feature}'"
            )

        metadata = self.api.query_feature(
            self.population_type, feature, *items
        )
        domain = dict(metadata[["id", "total_count"]].to_dict("split")["data"])

        return domain

    def get_domain(self):
        """
        Retrieve domain metadata from the API.

        Returns
        -------
        domain : mbi.Domain
            Dictionary-like object defining the domain size of every column
            in the synthetic dataset (area type and dimensions).
        """

        area_type_domain = self._get_domain_of_feature("area-types")
        dimension_domain = self._get_domain_of_feature("dimensions")

        domain = Domain.fromdict({**area_type_domain, **dimension_domain})

        return domain

    def get_marginal(self, clique, flatten=True):
        """
        Retrieve the marginal table for a clique from the API.

        This function also returns the metadata to "measure" the
        marginal in the package that underpins the synthesis, `mbi`.

        Parameters
        ----------
        clique : tuple of str
            Tuple defining the columns of the clique to be measured.
            Should be of the form `(col,)` or `(col1, col2)`.
        flatten : bool
            Whether the marginal should be flattened or not. Default is
            `True` to work with `mbi`. Flattened marginals are NumPy
            arrays rather than Pandas series.

        Returns
        -------
        marginal : numpy.ndarray or pandas.Series or None
            Marginal table if the API call succeeds and `None` if not.
            On a success, if `flatten` is `True`, this a flat array.
            Otherwise, the indexed series is returned.
        """

        area_type = self.area_type or "nat"
        dimensions = [col for col in clique if col != area_type]
        if not dimensions:
            dimensions = self.dimensions[0:1]

        marginal = self.api.query_table(
            self.population_type, area_type, dimensions
        )

        if marginal is not None:
            marginal = marginal.groupby(list(clique))["count"].sum()
            if flatten is True:
                marginal = marginal.to_numpy().flatten()

        return marginal

    def measure(self, cliques):
        """
        Measure the marginals of a set of cliques.

        This function returns a list of "measurements" to be passed to
        the `mbi` package. Each measurement consists of a sparse
        identity matrix, the marginal table, a nominally small float
        representing the "noise" added to the marginal, and the clique
        associated with the marginal.

        Although we are not applying differential privacy to our tables,
        `mbi` requires non-zero noise for each measurement to form the
        graphical model.

        If a column pair has been blocked by the API, then their
        marginal is `None` and we skip over them.

        Parameters
        ----------
        cliques : iterable of tuple
            The cliques to measure. These cliques should be of the form
            `(col,)` or `(col1, col2)`.

        Returns
        -------
        measurements : list of tuple
            Measurement tuples for each clique.
        """

        tasks = []
        for clique in cliques:
            marginal = dask.delayed(self.get_marginal)(clique)
            tasks.append(marginal)

        marginals = dask.compute(*tasks)

        measurements = [
            (sparse.eye(marginal.size), marginal, 1e-12, clique)
            for marginal, clique in zip(marginals, cliques)
            if marginal is not None
        ]

        return measurements

    def fit_model(self, measurements, iters=5000):
        """
        Fit a graphical model to some measurements.

        Parameters
        ----------
        measurements : list of tuple
            Measurement tuples associated with some cliques to fit.
        iters : int
            Number of iterations to use when fitting the model. Default
            is 5000.

        Returns
        -------
        model : mbi.GraphicalModel
            Fitted graphical model.
        """

        engine = FactoredInference(self.domain, iters=iters)
        model = engine.estimate(measurements)

        return model
