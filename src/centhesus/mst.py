"""Module for the Maximum Spanning Tree generator."""

import itertools

import dask
import dask.array as da
import dask.dataframe as dd
import networkx as nx
import numpy as np
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

        We use `dask` to compute these marginals in parallel.

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
            (sparse.eye(marginal.size), marginal, 1, clique)
            for clique, marginal in zip(cliques, marginals)
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

    def _calculate_importance_of_pair(self, interim, pair):
        """
        Determine the importance of a column pair with an interim model.

        Importance is defined as the L1 norm between the observed
        marginal table for the column pair and that estimated by our
        interim model.

        Parameters
        ----------
        interim : mbi.GraphicalModel
            Interim model based on one-way marginals only.
        pair : tuple of str
            Column pair to be assessed.

        Returns
        -------
        pair : tuple of str
            Assessed column pair.
        weight : float or None
            Importance of the pair given as the L1 norm between the
            observed and estimated marginals for the pair. If the API
            call fails, this is `None`.
        """

        weight = None
        marginal = self.get_marginal(pair)
        if marginal is not None:
            estimate = interim.project(pair).datavector()
            weight = np.linalg.norm(marginal - estimate, 1)

        return weight

    def _calculate_importances(self, interim):
        """
        Determine every column pair's importance given an interim model.

        We use `dask` to compute these importances in parallel.

        Parameters
        ----------
        interim : mbi.GraphicalModel
            Interim model based on one-way marginals only.

        Returns
        -------
        weights : dict
            Dictionary mapping column pairs to their weight. If a column
            pair is blocked by the API, it is skipped.
        """

        pairs = list(itertools.combinations(self.domain.attrs, 2))
        tasks = []
        for pair in pairs:
            importance = dask.delayed(self._calculate_importance_of_pair)(
                interim, pair
            )
            tasks.append(importance)

        importances = dask.compute(*tasks)

        weights = {
            pair: importance
            for pair, importance in zip(pairs, importances)
            if importance is not None
        }

        return weights

    def _find_maximum_spanning_tree(self, weights):
        """
        Find the maximum spanning tree given a set of edge importances.

        To find the tree, we use Kruskal's algorithm to find the minimum
        spanning tree with negative weights.

        Parameters
        ----------
        weights : dict
            Dictionary mapping edges (column pairs) to their importance.

        Returns
        -------
        tree : nx.Graph
            Maximum spanning tree of all column pairs.
        """

        graph = nx.Graph()
        graph.add_nodes_from(self.domain)
        for edge, weight in weights.items():
            graph.add_edge(*edge, weight=-weight)

        tree = nx.minimum_spanning_tree(graph)

        return tree

    def select(self, measurements):
        """
        Select the most informative two-way cliques.

        To determine how informative a column pair is, we first create
        an interim graphical model from all observed one-way marginals.
        Then, each column pair's importance is defined as the L1
        difference between its observed two-way marginal and the
        estimated marginal from the interim model.

        With all the importances calculated, we model the column pairs
        as a weighted graph where columns are nodes and an edge
        represents the importance of the column pair at its endpoints.
        In this way, the smallest set of the most informative column
        pairs is given as the maximum spanning tree of this graph.

        The selected two-way cliques are the edges of this tree.

        Parameters
        ----------
        measurements : list of tuple
            One-way marginal measurements with which to fit an interim
            graphical model.

        Returns
        -------
        cliques : list of tuple
            Edges of the maximum spanning tree of our weighted graph.
        """

        interim = self.fit_model(measurements, iters=1000)
        weights = self._calculate_importances(interim)
        tree = self._find_maximum_spanning_tree(weights)

        return list(tree.edges)

    @staticmethod
    def _setup_generate(model, nrows, seed):
        """
        Set everything up for the generation of the synthetic data.

        Parameters
        ----------
        model : mbi.GraphicalModel
            Model from which the synthetic data will be drawn.
        nrows : int or None
            Number of rows in the synthetic data. Inferred from `model`
            if `None`.
        seed : int or None
            Pseudo-random seed. If `None`, randomness not reproducible.

        Returns
        -------
        nrows : int
            Number of rows to generate.
        prng : dask.array.random.Generator
            Pseudo-random number generator.
        cliques : list of set
            Cliques identified by the graphical model.
        order : list of str
            Order in which to synthesise the columns.
        """

        nrows = int(model.total) if nrows is None else nrows
        prng = da.random.default_rng(seed)
        cliques = [set(clique) for clique in model.cliques]
        column, *order = model.elimination_order[::-1]

        return nrows, prng, cliques, column, order

    @staticmethod
    def _synthesise_column(marginal, nrows, prng, chunksize=1e6):
        """
        Sample a column of given length based on a marginal.

        Columns are synthesised to match the distribution of the
        marginal very closely. The process for synthesising the column
        is as follows:

        1. Scale the marginal against the total count required, and then
           separate its integer and fractional components.
        2. If there are insufficient integer counts, distribute the
           additional elements among the integer counts randomly
           using the fractional component as a weight. In this way, the
           difference between the normalised marginal and the final
           counts in the synthetic data will be at most one.
        3. Create an array by repeating the index of the marginal
           according to the adjusted integer counts.
        4. Permute the array to give a synthetic column.

        Parameters
        ----------
        marginal : np.ndarray
            Marginal counts from which to synthesise the column.
        nrows : int
            Number of elements in the synthesised column.
        prng : dask.array.random.Generator
            Pseudo-random number generator. We use this to distribute
            additional elements in the synthetic column, and to shuffle
            its elements after creation.
        chunksize : int or float
            Target size of a chunk or partition in the column.

        Returns
        -------
        column : dask.dataframe.Series
            Synthetic column closely matching the distribution of the
            marginal.
        """

        marginal *= nrows / marginal.sum()
        fractions, integers = np.modf(marginal)

        integers = integers.astype(int)
        extra = nrows - integers.sum()
        if extra > 0:
            idx = prng.choice(
                marginal.size, extra, False, fractions / fractions.sum()
            ).compute()
            integers[idx] += 1

        uniques = np.arange(integers.size)
        repeats = (
            unique * da.ones(shape=count, dtype=int)
            for unique, count in zip(uniques, integers)
        )

        values = da.concatenate(repeats).rechunk(chunksize)
        column = dd.from_dask_array(prng.permutation(values))

        return column

    @staticmethod
    def _synthesise_column_in_group(
        group, column, marginal, prng, chunksize=1e6
    ):
        """
        Synthesise a column inside a group-by operation.

        This operation is used for synthesising columns that depend on
        those that have already been synthesised. By performing this
        synthesis in a group-by operation, we ensure a close matching to
        the marginal distribution estimated by the graphical model given
        what has already been synthesised.

        Parameters
        ----------
        group : pandas.DataFrame
            Group data frame on which to operate.
        column : str
            Name of column to be synthesised.
        marginal : np.ndarray
            Marginal estimated from the graphical model for the column
            and all the columns it depends on.
        prng : dask.array.random.Generator
            Pseudo-random number generator. Used to synthesise the
            column within this group.

        Returns
        -------
        group :
            Group with new synthetic column.
        """

        idx = group.name
        group[column] = MST._synthesise_column(
            marginal[idx], group.shape[0], prng, chunksize
        )

        return group

    def _synthesise_first_column(self, model, column, nrows, prng):
        """
        Sample the first column from the model as a data frame.

        Parameters
        ----------
        model : mbi.GraphicalModel
            Model from which to synthesise the column.
        column : str
            Name of the column to synthesise.
        nrows : int
            Number of rows to generate.
        prng : dask.array.random.Generator
            Pseudo-random number generator.

        Returns
        -------
        data : dask.dataframe.DataFrame
            Data frame containing the first synthetic column.
        """

        marginal = model.project([column]).datavector(flatten=False)
        data = self._synthesise_column(marginal, nrows, prng).to_frame(
            name=column
        )

        return data

    @staticmethod
    def _find_prerequisite_columns(column, cliques, used):
        """
        Find the columns that inform the synthesis of a new column.

        Parameters
        ----------
        column : str
            Name of column to be synthesised.
        cliques : list of set
            Cliques identified by the graphical model.
        used : set of str
            Names of columns that have already been synthesised.

        Returns
        -------
        prerequisites : tuple of str
            All columns needed to synthesise the new column.
        """

        member_of_cliques = [clique for clique in cliques if column in clique]
        prerequisites = used.intersection(set.union(*member_of_cliques))

        return tuple(prerequisites)

    def generate(self, model, nrows=None, seed=None):
        """
        Generate a synthetic dataset from the estimated model.

        Columns are synthesised in the order determined by the graphical
        model. With each column after the first, we search for all the
        columns on which it depends according to the model that have
        been synthesised already.

        Parameters
        ----------
        model : mbi.GraphicalModel
            Model from which to draw synthetic data. This model should
            be fit to all the marginal tables you care about.
        nrows : int, optional
            Number of rows in the synthetic dataset. If not specified,
            the length of the dataset is inferred from the model.
        seed : int, optional
            Seed for pseudo-random number generation. If not specified,
            the results will not be reproducible.

        Returns
        -------
        data : dask.dataframe.DataFrame
            Data frame containing the synthetic data. We use Dask to
            allow for larger-than-memory datasets. As such, it is lazily
            executed.
        """

        nrows, prng, cliques, column, order = self._setup_generate(
            model, nrows, seed
        )
        data = self._synthesise_first_column(model, column, nrows, prng)
        used = {column}

        for column in order:
            clique = self._find_prerequisite_columns(column, cliques, used)
            used.add(column)

            marginal = model.project(clique + (column,)).datavector(
                flatten=False
            )

            if len(clique) >= 1:
                data = (
                    data.groupby(list(clique))
                    .apply(
                        self._synthesise_column_in_group,
                        column,
                        marginal,
                        prng,
                        meta={**data.dtypes, column: int},
                    )
                    .reset_index(drop=True)
                )
            else:
                data[column] = self._synthesise_column(marginal, nrows, prng)

        data = data.repartition("100MB")

        return data
