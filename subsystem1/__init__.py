"""Subsystem 1 package: Event-Driven Stream Processing, Real-Time Energy Monitoring & Data Management."""

from subsystem1.database import DatabaseManager
from subsystem1.feature_engineering import DEFAULT_BUILDING_CATALOG, FeatureEngineer
from subsystem1.ingestion import (
    DataIngestionEngine,
    SmartMeterStream,
)
from subsystem1.metrics import MetricsCalculator
from subsystem1.exceptions_subsystem1 import (
    DatabaseOperationError,
    DataValidationError,
    EnergyPipelineError,
    StreamProcessingError,
)
from subsystem1.models import Building, EnergyLog
from subsystem1.pipeline import (
    StreamPipeline,
    StreamProcessor,
)

__all__ = [
    "EnergyPipelineError",
    "DataValidationError",
    "StreamProcessingError",
    "DatabaseOperationError",
    "StreamProcessor",
    "StreamPipeline",
    "Building",
    "EnergyLog",
    "SmartMeterStream",
    "DataIngestionEngine",
    "FeatureEngineer",
    "DEFAULT_BUILDING_CATALOG",
    "MetricsCalculator",
    "DatabaseManager",
]

