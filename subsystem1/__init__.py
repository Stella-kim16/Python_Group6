"""Subsystem 1 package: Event-Driven Stream Processing, Real-Time Energy Monitoring & Data Management."""

from subsystem1.database import DatabaseManager
from subsystem1.feature_engineering import DEFAULT_BUILDING_CATALOG, FeatureEngineer
from subsystem1.ingestion import (
    DataIngestionEngine,
    SmartMeterStream,
    SubThresholdMeterFilter,
)
from subsystem1.metrics import LoadBalanceAndTariffTransformer, MetricsCalculator
from subsystem1.models import (
    Building,
    DatabaseOperationError,
    DataValidationError,
    EnergyLog,
    EnergyPipelineError,
    StreamProcessingError,
)
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
    "SubThresholdMeterFilter",
    "DataIngestionEngine",
    "FeatureEngineer",
    "DEFAULT_BUILDING_CATALOG",
    "MetricsCalculator",
    "LoadBalanceAndTariffTransformer",
    "DatabaseManager",
]

