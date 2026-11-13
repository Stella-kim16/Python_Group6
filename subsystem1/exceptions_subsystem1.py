"""Custom exceptions for Subsystem 1."""


class EnergyPipelineError(Exception):
    """Base exception for all Subsystem 1 energy pipeline errors. """
    pass


class DataValidationError(EnergyPipelineError, ValueError):
    """
    Multiple Inheritance ( DataValidationError -> EnergyPipelineError -> ValueError).
    """
    pass


class StreamProcessingError(EnergyPipelineError, RuntimeError):
    """Raised when a stream processor encounters an runtime failure."""
    pass


class DatabaseOperationError(EnergyPipelineError, RuntimeError):
    """Raised when SQLite executions fail."""
    pass
