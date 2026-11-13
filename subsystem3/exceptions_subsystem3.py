class Subsystem3Error(Exception):
    """Base exception class for all Subsystem 3 errors."""
    pass


class ScenarioGenerationError(Subsystem3Error):
    """Raised when the engine fails to generate operational scenarios."""
    pass


class OptimisationEvaluationError(Subsystem3Error):
    """Raised when evaluating a scenario fails or produces invalid metrics."""
    pass


class VisualisationError(Subsystem3Error):
    """Raised when the ResultsVisualiser fails to render or save a chart."""
    pass


class DataLoadError(Subsystem3Error):
    """Raised when Subsystem 2 data files cannot be found or parsed."""
    pass


class OptimizationError(Subsystem3Error):
    """Raised when scenario generation or evaluation fails."""
    pass


class AIAdvisorError(Subsystem3Error):
    """Raised when the KI:connect NRW API call fails or times out."""
    pass
