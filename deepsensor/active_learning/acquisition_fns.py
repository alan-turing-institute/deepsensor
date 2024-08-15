import copy
from typing import Optional

import numpy as np

from scipy.stats import norm

from deepsensor.model.model import ProbabilisticModel
from deepsensor.data.task import Task


class AcquisitionFunction:
    """Parent class for acquisition functions."""

    # Class attribute to indicate whether the acquisition function should be
    #   minimised or maximised
    min_or_max = None

    def __init__(
        self,
        model: Optional[ProbabilisticModel] = None,
        context_set_idx: int = 0,
        target_set_idx: int = 0,
    ):
        """Args:
        model (:class:`~.model.model.ProbabilisticModel`):
            [Description of the model parameter.]
        context_set_idx (int):
            Index of context set to add new observations to when computing
            the acquisition function.
        target_set_idx (int):
            Index of target set to compute acquisition function for.
        """
        self.model = model
        self.context_set_idx = context_set_idx
        self.target_set_idx = target_set_idx

    def __call__(self, task: Task, *args, **kwargs) -> np.ndarray:
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            :class:`numpy:numpy.ndarray`:
                Acquisition function value/s. Shape ().

        Raises:
            NotImplementedError:
                Because this is an abstract method, it must be implemented by
                the subclass.
        """
        raise NotImplementedError


class AcquisitionFunctionOracle(AcquisitionFunction):
    """Signifies that the acquisition function is computed using the true
    target values.
    """


class AcquisitionFunctionParallel(AcquisitionFunction):
    """Parent class for acquisition functions that are computed across all search
    points in parallel.
    """

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs) -> np.ndarray:
        """...

        :param **kwargs:
        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.
            X_s (:class:`numpy:numpy.ndarray`):
                Search points. Shape (2, N_search).

        Returns:
            :class:`numpy:numpy.ndarray`:
                Should return acquisition function value/s. Shape (N_search,).

        Raises:
            NotImplementedError:
                Because this is an abstract method, it must be implemented by
                the subclass.
        """
        raise NotImplementedError


class MeanStddev(AcquisitionFunction):
    """Mean of the marginal variances."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                [Description of the task parameter.]

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        return np.mean(self.model.stddev(task)[self.target_set_idx])


class MeanVariance(AcquisitionFunction):
    """Mean of the marginal variances."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                [Description of the task parameter.]

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        return np.mean(self.model.variance(task)[self.target_set_idx])


class pNormStddev(AcquisitionFunction):
    """p-norm of the vector of marginal standard deviations."""

    min_or_max = "min"

    def __init__(self, *args, p: int = 1, **kwargs):
        """...

        :no-index:

        Args:
            p (int, optional):
                [Description of the parameter p.], default is 1
        """
        super().__init__(*args, **kwargs)
        self.p = p

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                [Description of the task parameter.]

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        return np.linalg.norm(
            self.model.stddev(task)[self.target_set_idx].ravel(), ord=self.p
        )


class MeanMarginalEntropy(AcquisitionFunction):
    """Mean of the entropies of the marginal predictive distributions."""

    min_or_max = "min"

    def __call__(self, task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        marginal_entropy = self.model.mean_marginal_entropy(task)
        return marginal_entropy


class JointEntropy(AcquisitionFunction):
    """Joint entropy of the predictive distribution."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        return self.model.joint_entropy(task)


class OracleMAE(AcquisitionFunctionOracle):
    """Oracle mean absolute error."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        pred = self.model.mean(task)
        if isinstance(pred, list):
            pred = pred[self.target_set_idx]
        true = task["Y_t"][self.target_set_idx]
        return np.mean(np.abs(pred - true))


class OracleRMSE(AcquisitionFunctionOracle):
    """Oracle root mean squared error."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        pred = self.model.mean(task)
        if isinstance(pred, list):
            pred = pred[self.target_set_idx]
        true = task["Y_t"][self.target_set_idx]
        return np.sqrt(np.mean((pred - true) ** 2))


class OracleMarginalNLL(AcquisitionFunctionOracle):
    """Oracle marginal negative log-likelihood."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        pred = self.model.mean(task)
        if isinstance(pred, list):
            pred = pred[self.target_set_idx]
        true = task["Y_t"][self.target_set_idx]
        return -np.mean(norm.logpdf(true, loc=pred, scale=self.model.stddev(task)))


class OracleJointNLL(AcquisitionFunctionOracle):
    """Oracle joint negative log-likelihood."""

    min_or_max = "min"

    def __call__(self, task: Task):
        """...

        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        return -self.model.logpdf(task)


class Random(AcquisitionFunctionParallel):
    """Random acquisition function."""

    min_or_max = "max"

    def __init__(self, *args, seed: int = 42, **kwargs):
        """...

        :no-index:

        Args:
            seed (int, optional):
                Random seed, defaults to 42.
        """
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs):
        """...

        :param **kwargs:
        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                [Description of the task parameter.]
            X_s (:class:`numpy:numpy.ndarray`):
                [Description of the X_s parameter.]

        Returns:
            float:
                A random acquisition function value.
        """
        return self.rng.random(X_s.shape[1])


class ContextDist(AcquisitionFunctionParallel):
    """Distance to closest context point."""

    min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs):
        """...

        :param **kwargs:
        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                [Description of the task parameter.]
            X_s (:class:`numpy:numpy.ndarray`):
                [Description of the X_s parameter.]

        Returns:
            [Type of the return value]:
                [Description of the return value.]
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
                X_s[..., np.newaxis] - X_c[..., np.newaxis, :],
                axis=0,
            )  # Shape (n_possible_locs, n_context + n_placed_sensors)

            # Compute distance to nearest sensor
            dist_to_closest_sensor = dists_all.min(axis=1)
        return dist_to_closest_sensor


class Stddev(AcquisitionFunctionParallel):
    """Model standard deviation."""

    min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs):
        """...

        :param **kwargs:
        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                [Description of the task parameter.]
            X_s (:class:`numpy:numpy.ndarray`):
                [Description of the X_s parameter.]

        Returns:
            [Type of the return value]:
                [Description of the return value.]
        """
        # Set the target points to the search points
        task = copy.deepcopy(task)
        task["X_t"] = X_s

        return self.model.stddev(task)[self.target_set_idx]


class ExpectedImprovement(AcquisitionFunctionParallel):
    """Expected improvement acquisition function.

    .. note::

        The current implementation of this acquisition function is only valid
        for maximisation.
    """

    min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs) -> np.ndarray:
        """:param **kwargs:
        :no-index:

        Args:
            task (:class:`~.data.task.Task`):
                Task object containing context and target sets.
            X_s (:class:`numpy:numpy.ndarray`):
                Search points. Shape (2, N_search).

        Returns:
            :class:`numpy:numpy.ndarray`:
                Acquisition function value/s. Shape (N_search,).
        """
        # Set the target points to the search points
        task = copy.deepcopy(task)
        task["X_t"] = X_s

        # Compute the predictive mean and variance of the target set
        mean = self.model.mean(task)[self.target_set_idx]

        if task["Y_c"][self.context_set_idx].size == 0:
            # No previous context points, so heuristically use the predictive mean as the
            # acquisition function. This will at least select the most positive predicted mean.
            return self.model.mean(task)[self.target_set_idx]
        else:
            # Determine the best target value seen so far
            best_target_value = task["Y_c"][self.context_set_idx].max()

        # Compute the standard deviation of the context set
        stddev = self.model.stddev(task)[self.context_set_idx]

        # Compute the expected improvement
        Z = (mean - best_target_value) / stddev
        ei = stddev * (mean - best_target_value) * norm.cdf(Z) + stddev * norm.pdf(Z)

        return ei
