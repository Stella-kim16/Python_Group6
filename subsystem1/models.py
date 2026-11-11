"""Domain models, Custom Exceptions, Mixins, and Abstract Base Classes for Subsystem 1."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


# =====================================================================
# Centralized Exception Hierarchy (Fault Tolerance & Layer Isolation)
# =====================================================================


class EnergyPipelineError(Exception):
    """Base exception for all Subsystem 1 energy pipeline errors."""

    pass


class DataValidationError(EnergyPipelineError, ValueError):
    """
    Raised when incoming sensor payloads or CSV schemas fail validation.
    Demonstrates Multiple Inheritance (MRO: DataValidationError -> EnergyPipelineError -> ValueError).
    """

    pass


class StreamProcessingError(EnergyPipelineError, RuntimeError):
    """Raised when a stream processor encounters an unrecoverable runtime failure."""

    pass


class DatabaseOperationError(EnergyPipelineError, RuntimeError):
    """Raised when SQLite persistence, schema initialization, or query execution fails."""

    pass


# =====================================================================
# Mixin Class (Demonstrates Multiple Inheritance & Python C3 MRO)
# =====================================================================


class AuditLoggingMixin:
    """
    Mixin providing structured audit logging and telemetry counters.
    Used alongside ABCs to demonstrate conflict-free Multiple Inheritance and MRO.
    """

    def _init_audit_logger(self, name: str, logger: Optional[logging.Logger] = None) -> None:
        """Initialize or inject a logger instance for structured component auditing."""
        self._logger: logging.Logger = logger or logging.getLogger(f"subsystem1.{name}")

    def log_audit(self, message: str, level: int = logging.DEBUG) -> None:
        """Emit a structured audit log entry for observability."""
        if hasattr(self, "_logger") and self._logger is not None:
            self._logger.log(level, f"[{self.__class__.__name__}] {message}")


# =====================================================================
# Abstract Base Class & Composable Pipeline
# =====================================================================


class StreamProcessor(ABC, AuditLoggingMixin):
    """
    Abstract Base Class (ABC) defining the uniform interface contract for all
    event-driven stream processors in the Smart Energy Management System.

    Demonstrates:
    1. Abstract Base Classes (ABCs) & Abstraction: `@abstractmethod process`.
    2. Encapsulation: Private attributes (`__processed_count`, `__dropped_count`)
       with `@property` getters and protected mutator methods.
    3. Class vs. Instance Variables: `PROCESSOR_API_VERSION` (class attribute) vs.
       instance state initialized in `__init__`.
    4. Multiple Inheritance & MRO: Inherits from `ABC` and `AuditLoggingMixin`.
    5. Operator Overloading: `__add__` (+) and `__or__` (|) to compose pipelines.
    """

    # Class variable shared across all StreamProcessor instances
    PROCESSOR_API_VERSION: str = "2.0"

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self._init_audit_logger(self.name, logger=logger)
        # Private instance attributes (name-mangled) for strict encapsulation
        self.__processed_count: int = 0
        self.__dropped_count: int = 0

    @property
    def processed_count(self) -> int:
        """Read-only property returning total events successfully processed."""
        return self.__processed_count

    @property
    def dropped_count(self) -> int:
        """Read-only property returning total events filtered out or dropped."""
        return self.__dropped_count

    def _record_processed(self, count: int = 1) -> None:
        """Protected helper for subclasses to increment the processed counter."""
        self.__processed_count += int(count)

    def _record_dropped(self, count: int = 1) -> None:
        """Protected helper for subclasses to increment the dropped counter."""
        self.__dropped_count += int(count)

    def reset_metrics(self) -> None:
        """Reset internal telemetry counters for idempotent re-runs and testing."""
        self.__processed_count = 0
        self.__dropped_count = 0

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process a single streaming event (dict / EnergyLog) or batch DataFrame.
        Returns None if the event is filtered out (e.g., inactive or sub-threshold meter).
        """
        pass

    def __add__(self, other: "StreamProcessor") -> "StreamPipeline":
        """Compose two stream processors into a pipeline using the `+` operator."""
        if not isinstance(other, StreamProcessor):
            return NotImplemented
        left_steps = self.processors if isinstance(self, StreamPipeline) else [self]
        right_steps = other.processors if isinstance(other, StreamPipeline) else [other]
        return StreamPipeline(left_steps + right_steps)

    def __or__(self, other: "StreamProcessor") -> "StreamPipeline":
        """Compose two stream processors into a pipeline using the `|` operator."""
        return self.__add__(other)

    def __call__(self, data: Any) -> Any:
        """Allow processor instances to be called directly as functions."""
        return self.process(data)

    def __str__(self) -> str:
        """Return a human-readable summary of the processor and its execution metrics."""
        return (
            f"{self.name}(v{self.PROCESSOR_API_VERSION}, "
            f"processed={self.__processed_count}, dropped={self.__dropped_count})"
        )


class StreamPipeline(StreamProcessor):
    """
    Composable event-driven stream processing pipeline created by chaining
    `StreamProcessor` instances with `+` (`__add__`) or `|` (`__or__`).

    Demonstrates:
    - Composition over Inheritance: Holds references to `StreamProcessor` objects
      (`self.processors`) and stream sources (`self._source`).
    - Dependency Injection: Accepts processors, stream source, and logger via `__init__`.
    - Fault Tolerance & Delivery Guarantees: Tracks seen `(timestamp, building_id)`
      keys for idempotent deduplication and per-building watermarks to detect
      out-of-order telemetry events.
    """

    def __init__(
        self,
        processors: Optional[List[StreamProcessor]] = None,
        source: Optional[Iterable[Any]] = None,
        deduplicate_stream: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name="StreamPipeline", logger=logger)
        self.processors: List[StreamProcessor] = list(processors or [])
        self._source: Optional[Iterable[Any]] = source
        self._iterator: Optional[Iterator[Any]] = None
        self.deduplicate_stream: bool = deduplicate_stream

        # Stream delivery state: duplicate suppression & out-of-order watermark tracking
        self._seen_event_keys: Set[Tuple[str, str]] = set()
        self._high_watermarks: Dict[str, datetime] = {}
        self.__duplicate_count: int = 0
        self.__out_of_order_count: int = 0

    @property
    def duplicate_count(self) -> int:
        """Return number of duplicate streaming events suppressed by the pipeline."""
        return self.__duplicate_count

    @property
    def out_of_order_count(self) -> int:
        """Return number of out-of-order streaming events detected via watermarks."""
        return self.__out_of_order_count

    def reset_stream_state(self) -> None:
        """Clear deduplication cache, watermarks, and counters for idempotent runs."""
        self._seen_event_keys.clear()
        self._high_watermarks.clear()
        self.__duplicate_count = 0
        self.__out_of_order_count = 0
        self.reset_metrics()
        for proc in self.processors:
            proc.reset_metrics()

    def bind_source(self, source: Iterable[Any]) -> "StreamPipeline":
        """Bind an iterable or stream source to this pipeline for iterator consumption."""
        self._source = source
        self._iterator = None
        return self

    def _extract_event_identity(self, event: Any) -> Optional[Tuple[str, str, Optional[datetime]]]:
        """Extract `(timestamp_str, building_id, parsed_dt)` from a dict or EnergyLog event."""
        if isinstance(event, EnergyLog):
            ts_dt = event.timestamp
            return ts_dt.strftime("%Y-%m-%d %H:%M:%S"), event.building_id, ts_dt
        if isinstance(event, dict):
            raw_ts = event.get("Timestamp", event.get("timestamp"))
            bld_id = str(event.get("building_id", "UNASSIGNED"))
            if raw_ts is None:
                return None
            try:
                ts_dt = (
                    raw_ts
                    if isinstance(raw_ts, datetime)
                    else datetime.fromisoformat(str(raw_ts).strip())
                )
                ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                ts_dt = None
                ts_str = str(raw_ts).strip()
            return ts_str, bld_id, ts_dt
        return None

    def _check_delivery_guarantees(self, event: Any) -> bool:
        """
        Enforce effectively-once stream semantics:
        1. Detect and drop duplicate `(timestamp, building_id)` messages.
        2. Detect and log out-of-order arrivals against per-building high-watermarks.
        Returns True if the event is unique and should be emitted; False if duplicate.
        """
        identity = self._extract_event_identity(event)
        if identity is None:
            return True

        ts_str, bld_id, ts_dt = identity
        event_key = (ts_str, bld_id)

        if self.deduplicate_stream and event_key in self._seen_event_keys:
            self.__duplicate_count += 1
            self._record_dropped()
            self.log_audit(f"Duplicate event dropped for {event_key}")
            return False

        self._seen_event_keys.add(event_key)

        if ts_dt is not None:
            last_dt = self._high_watermarks.get(bld_id)
            if last_dt is not None and ts_dt < last_dt:
                self.__out_of_order_count += 1
                self.log_audit(
                    f"Out-of-order event detected for {bld_id}: {ts_dt} < watermark {last_dt}"
                )
            else:
                self._high_watermarks[bld_id] = ts_dt

        return True

    def process(self, data: Any) -> Any:
        """
        Pass an event (or DataFrame) sequentially through all composed processors.
        If any filter stage returns None, short-circuit and drop the event.
        """
        current = data
        try:
            for processor in self.processors:
                if current is None:
                    self._record_dropped()
                    return None
                current = processor.process(current)
        except EnergyPipelineError:
            raise
        except Exception as exc:
            self.log_audit(f"Pipeline stage error: {exc}", level=logging.ERROR)
            raise StreamProcessingError(f"Pipeline execution failed: {exc}") from exc

        if current is None:
            self._record_dropped()
        else:
            self._record_processed()
        return current

    def process_stream(self, stream: Iterable[Any]) -> Iterator[Any]:
        """
        Generator that lazily consumes an input stream on the fly and yields
        only valid, unique, transformed events (skipping filtered or duplicate events).
        """
        self.reset_stream_state()
        for event in stream:
            result = self.process(event)
            if result is not None and self._check_delivery_guarantees(result):
                yield result

    def __iter__(self) -> "StreamPipeline":
        """Initialize stream iteration over the bound source."""
        if self._source is None:
            raise DataValidationError(
                "No stream source bound to StreamPipeline. Call `bind_source(stream)` first."
            )
        self.reset_stream_state()
        self._iterator = iter(self._source)
        return self

    def __next__(self) -> Any:
        """
        Fetch and process the next valid, non-duplicate event from the bound stream source,
        automatically skipping events filtered out (where `process` returns None).
        """
        if self._iterator is None:
            raise StopIteration
        while True:
            raw_event = next(self._iterator)  # Raises StopIteration when exhausted
            processed = self.process(raw_event)
            if processed is not None and self._check_delivery_guarantees(processed):
                return processed

    def __str__(self) -> str:
        """Return a readable diagram of the composed pipeline stages and counters."""
        stage_names = " -> ".join(p.name for p in self.processors) or "Empty"
        return (
            f"StreamPipeline([{stage_names}], "
            f"processed={self.processed_count}, dropped={self.dropped_count}, "
            f"duplicates={self.__duplicate_count}, out_of_order={self.__out_of_order_count})"
        )


# =====================================================================
# Core Domain Entities
# =====================================================================


@dataclass
class Building:
    """Represents a monitored smart building (Home, Office, or University)."""

    building_id: str
    building_type: str
    area_sqft: float
    max_occupancy: int
    baseline_kwh: float

    def get_building_info(self) -> Dict[str, Any]:
        """Return building metadata as a dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Return a formatted description of the building instance."""
        return (
            f"Building(id={self.building_id}, type={self.building_type}, "
            f"area={self.area_sqft:.0f} sqft, max_occ={self.max_occupancy}, "
            f"baseline={self.baseline_kwh:.1f} kWh)"
        )


@dataclass
class EnergyLog:
    """Represents a single validated and feature-engineered energy telemetry record."""

    timestamp: datetime
    building_id: str
    building_type: str
    area_sqft: float
    temperature: float
    humidity: float
    occupancy: int
    hvac_usage: str
    lighting_usage: str
    renewable_energy_pct: float
    day_of_week: str
    holiday: str
    energy_consumption_kwh: float
    carbon_footprint_kg: float = 0.0
    energy_cost_eur: float = 0.0
    is_anomaly: bool = False

    def validate_data_types(self) -> bool:
        """Validate domain constraints and data types for this telemetry record."""
        if not isinstance(self.timestamp, datetime):
            return False
        if not isinstance(self.occupancy, int) or self.occupancy < 0:
            return False
        if self.area_sqft <= 0:
            return False
        if not (0.0 <= self.humidity <= 100.0):
            return False
        if not (0.0 <= self.renewable_energy_pct <= 100.0):
            return False
        if self.hvac_usage not in ("On", "Off") or self.lighting_usage not in ("On", "Off"):
            return False
        if self.holiday not in ("Yes", "No"):
            return False
        if self.energy_consumption_kwh < 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the energy log record into a JSON-compatible dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return data

    def __str__(self) -> str:
        """Return a concise human-readable telemetry summary."""
        anomaly_tag = " [ANOMALY]" if self.is_anomaly else ""
        return (
            f"EnergyLog({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{self.building_id} ({self.building_type}) | "
            f"{self.energy_consumption_kwh:.2f} kWh | "
            f"EUR {self.energy_cost_eur:.2f} | "
            f"CO2 {self.carbon_footprint_kg:.2f} kg{anomaly_tag})"
        )

