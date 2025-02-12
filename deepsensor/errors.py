class TaskSetIndexError(Exception):
    """Raised when the task context/target set index is out of range."""

    def __init__(self, index, set_length, context_or_target):
        super().__init__(
            f"{context_or_target} set index {index} is out of range for task with "
            f"{set_length} {context_or_target} sets."
        )


class GriddedDataError(Exception):
    """Raised during invalid operation with gridded data."""

    pass


class InvalidSamplingStrategyError(Exception):
    """Raised when TaskLoader sampling strategy is invalid."""

    pass


class SamplingTooManyPointsError(ValueError):
    """Raised when the number of points to sample is greater than the number of points in the dataset."""

    def __init__(self, requested: int, available: int):
        super().__init__(
            f"Requested {requested} points to sample, but only {available} are available."
        )
