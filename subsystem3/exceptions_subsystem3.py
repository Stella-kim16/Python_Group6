class Subsystem3Error(Exception):
    """Base exception class for all Subsystem 3 errors."""
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