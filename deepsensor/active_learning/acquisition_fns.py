import copy

import numpy as np

from scipy.stats import norm

from deepsensor.model.model import ProbabilisticModel
from deepsensor.data.task import Task


class AcquisitionFunction:
    """
    Parent class for acquisition functions.
    """

    def __init__(self, model: ProbabilisticModel):
        """
        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        context_set_idx : int
            Index of context set to add new observations to when computing the
            acquisition function.
        """
        self.model = model
        self.min_or_max = -1

    def __call__(self, task: Task) -> np.ndarray:
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.

        Returns
        -------
        numpy.ndarray
            Acquisition function value/s. Shape ().

        Raises
        ------
        NotImplementedError
            Because this is an abstract method, it must be implemented by the
            subclass.
        """
        raise NotImplementedError


class AcquisitionFunctionOracle(AcquisitionFunction):
    """
    Signifies that the acquisition function is computed using the true
    target values.
    """


class AcquisitionFunctionParallel(AcquisitionFunction):
    """
    Parent class for acquisition functions that are computed across all search
    points in parallel.
    """

    def __call__(self, task: Task, X_s: np.ndarray) -> np.ndarray:
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.
        X_s : numpy.ndarray
            Search points. Shape (2, N_search).

        Returns
        -------
        numpy.ndarray
            Should return acquisition function value/s. Shape (N_search,).

        Raises
        ------
        NotImplementedError
            Because this is an abstract method, it must be implemented by the
            subclass.
        """
        raise NotImplementedError


class MeanStddev(AcquisitionFunction):
    """Mean of the marginal variances."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task, target_set_idx: int = 0):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...
        target_set_idx : int, optional
            ..., by default 0

        Returns
        -------
        ...
            ...
        """
        return np.mean(self.model.stddev(task)[target_set_idx])


class MeanVariance(AcquisitionFunction):
    """Mean of the marginal variances."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task, target_set_idx: int = 0):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...
        target_set_idx : int, optional
            ..., by default 0

        Returns
        -------
        ...
            ...
        """
        return np.mean(self.model.variance(task)[target_set_idx])


class pNormStddev(AcquisitionFunction):
    """p-norm of the vector of marginal standard deviations."""

    def __init__(self, *args, p: int = 1, **kwargs):
        """
        ...

        Parameters
        ----------
        p : int, optional
            ..., by default 1
        """
        super().__init__(*args, **kwargs)
        self.p = p
        self.min_or_max = "min"

    def __call__(self, task: Task, target_set_idx: int = 0):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...
        target_set_idx : int, optional
            ..., by default 0

        Returns
        -------
        ...
            ...
        """
        return np.linalg.norm(
            self.model.stddev(task)[target_set_idx].ravel(), ord=self.p
        )


class MeanMarginalEntropy(AcquisitionFunction):
    """Mean of the entropies of the marginal predictive distributions."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...

        Returns
        -------
        ...
            ...
        """
        marginal_entropy = self.model.mean_marginal_entropy(task)
        return marginal_entropy


class JointEntropy(AcquisitionFunction):
    """Joint entropy of the predictive distribution."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...

        Returns
        -------
        ...
            ...
        """
        return self.model.joint_entropy(task)


class OracleMAE(AcquisitionFunctionOracle):
    """Oracle mean absolute error."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...

        Returns
        -------
        ...
            ...
        """
        pred = self.model.mean(task)
        true = task["Y_t"]
        return np.mean(np.abs(pred - true))


class OracleRMSE(AcquisitionFunctionOracle):
    """Oracle root mean squared error."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...

        Returns
        -------
        ...
            ...
        """
        pred = self.model.mean(task)
        true = task["Y_t"]
        return np.sqrt(np.mean((pred - true) ** 2))


class OracleMarginalNLL(AcquisitionFunctionOracle):
    """Oracle marginal negative log-likelihood."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...

        Returns
        -------
        ...
            ...
        """
        pred = self.model.mean(task)
        true = task["Y_t"]
        return -np.mean(norm.logpdf(true, loc=pred, scale=self.model.stddev(task)))


class OracleJointNLL(AcquisitionFunctionOracle):
    """Oracle joint negative log-likelihood."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "min"

    def __call__(self, task: Task):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...

        Returns
        -------
        ...
            ...
        """
        return -self.model.logpdf(task)


class Random(AcquisitionFunctionParallel):
    """Random acquisition function."""

    def __init__(self, seed: int = 42):
        """
        ...

        Parameters
        ----------
        seed : int, optional
            Random seed, by default 42.
        """
        self.rng = np.random.default_rng(seed)
        self.min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...
        X_s : numpy.ndarray
            ...

        Returns
        -------
        float
            A random acquisition function value.
        """
        return self.rng.random(X_s.shape[1])


class ContextDist(AcquisitionFunctionParallel):
    """Distance to closest context point."""

    def __init__(self, context_set_idx: int = 0):
        """
        ...

        Parameters
        ----------
        context_set_idx : int, optional
            ..., by default 0
        """
        self.context_set_idx = context_set_idx
        self.min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...
        X_s : numpy.ndarray
            ...

        Returns
        -------
        ...
            ...
        """
        X_c = task["X_c"][self.context_set_idx]

        if X_c.size == 0:
            # No sensors placed yet, so arbitrarily choose first query point by setting its
            #    acquisition fn to non-zero and all others to zero
            dist_to_closest_sensor = np.zeros(X_s.shape[-1])
            dist_to_closest_sensor[0] = 1
        else:
            # Use broadcasting to get matrix of distances from each possible
            #   new sensor location to each existing sensor location
            dists_all = np.linalg.norm(
                X_s[..., np.newaxis] - X_c[..., np.newaxis, :], axis
            )  # Shape (n_possible_locs, n_context + n_placed_sensors)

            # Compute distance to nearest sensor
            dist_to_closest_sensor = dists_all.min(axis=1)
        return dist_to_closest_sensor


class Stddev(AcquisitionFunctionParallel):
    """Random acquisition function."""

    def __init__(self, model: ProbabilisticModel):
        """
        ...

        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        """
        super().__init__(model)
        self.min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, target_set_idx: int = 0):
        """
        ...

        Parameters
        ----------
        task : deepsensor.data.task.Task
            ...
        X_s : numpy.ndarray
            ...
        target_set_idx : int, optional
            ..., by default 0

        Returns
        -------
        ...
            ...
        """
        # Set the target points to the search points
        task = copy.deepcopy(task)
        task["X_t"] = X_s

        return self.model.stddev(task)[target_set_idx]


class ExpectedImprovement(AcquisitionFunctionParallel):
    """
    Expected improvement acquisition function.

    .. note::

        The current implementation of this acquisition function is only valid
        for maximisation.
    """

    def __init__(self, model: ProbabilisticModel, context_set_idx: int = 0):
        """
        Parameters
        ----------
        model : deepsensor.model.model.ProbabilisticModel
            ...
        context_set_idx : int
            Index of context set to add new observations to when computing the
            acquisition function.
        """
        super().__init__(model)
        self.context_set_idx = context_set_idx
        self.min_or_max = "max"

    def __call__(
        self, task: Task, X_s: np.ndarray, target_set_idx: int = 0
    ) -> np.ndarray:
        """
        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.
        X_s : numpy.ndarray
            Search points. Shape (2, N_search).
        target_set_idx : int
            Index of target set to compute acquisition function for.

        Returns
        -------
        numpy.ndarray
            Acquisition function value/s. Shape (N_search,).
        """
        # Set the target points to the search points
        task = copy.deepcopy(task)
        task["X_t"] = X_s

        # Compute the predictive mean and variance of the target set
        mean = self.model.mean(task)[target_set_idx]

        if task["Y_c"][self.context_set_idx].size == 0:
            # No previous context points, so heuristically use the predictive mean as the
            # acquisition function. This will at least select the most positive predicted mean.
            return self.model.mean(task)[target_set_idx]
        else:
            # Determine the best target value seen so far
            best_target_value = task["Y_c"][self.context_set_idx].max()

        # Compute the standard deviation of the context set
        stddev = self.model.stddev(task)[self.context_set_idx]

        # Compute the expected improvement
        Z = (mean - best_target_value) / stddev
        ei = stddev * (mean - best_target_value) * norm.cdf(Z) + stddev * norm.pdf(Z)

        return ei
