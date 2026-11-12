class Subsystem2Error(Exception):
    """Base exception class for all Subsystem 2 errors."""
    pass


class ModelTrainingError(Subsystem2Error, RuntimeError):
    """Raised when the AI model is used before it has been trained."""
    pass


class ForecastError(Subsystem2Error, ValueError):
    """Raised when a demand forecast cannot be produced, e.g. for an unknown building."""
    pass


class ModelEvaluationError(Subsystem2Error, ValueError):
    """Raised when the model cannot be evaluated, e.g. without actual values."""
    pass
