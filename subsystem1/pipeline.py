"""Abstract Base Class (StreamProcessor) and Composable StreamPipeline for Subsystem 1."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
from subsystem1.exceptions_subsystem1 import (DataValidationError, EnergyPipelineError, StreamProcessingError)
from subsystem1.models import EnergyLog


class StreamProcessor(ABC):
    """
    Abstract Base Class defining the uniform interface contract for all
    event-driven stream processors in the Smart Energy Management System.
    """

    # Class variable shared across all StreamProcessor instances
    PROCESSOR_API_VERSION: str = "2.0"

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.logger: logging.Logger = logger or logging.getLogger(f"subsystem1.{self.name}")
        self.__processed_count: int = 0   # private instance attributes for encapsulation
        self.__dropped_count: int = 0

    @property
    def processed_count(self) -> int:
        """Read-only property returning total events successfully processed."""
        return self.__processed_count
    # We keep it private with double underscores and expose it via @property without a @setter
    # so external code can read the counter like an attribute but cannot overwrite or corrupt it directly (encapsulation).

    @property
    def dropped_count(self) -> int:
        """Read-only property returning total events filtered out or dropped."""
        return self.__dropped_count

    def _record_processed(self, count: int = 1) -> None:  # one underscore -> protected method for subclasses
        """method for subclasses to increment the processed counter."""
        self.__processed_count += int(count)

    def _record_dropped(self, count: int = 1) -> None:
        """method for subclasses to increment the dropped counter."""
        self.__dropped_count += int(count)

    def reset_metrics(self) -> None:
        """reset internal counters for idempotent re-runs and testing."""
        self.__processed_count = 0
        self.__dropped_count = 0

    @abstractmethod   # Ensure subclasses implement this method
    def process(self, data: Any) -> Any:
        """
        Process a single streaming event or batch DataFrame. Returns None if the event is filtered out.
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
    Composable event-driven stream processing pipeline created by chaining `StreamProcessor` instances.
    """

    def __init__(
        self,
        processors: Optional[List[StreamProcessor]] = None,  # List of processors to include in the pipeline (composition)
        deduplicate_stream: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name="StreamPipeline", logger=logger)
        self.processors: List[StreamProcessor] = list(processors or [])
        self.deduplicate_stream: bool = deduplicate_stream    # Flag to enable/disable deduplication of streaming events
        self._high_watermarks: Dict[str, str] = {}   # Keep track of the latest timestamp seen per building for out-of-order detection
        self._seen_event_keys: Set[Tuple[str, str]] = set()  # Keep track of deduplicated events in the stream (timestamp, building_id)
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

    def _extract_event_identity(self, event: Any) -> Optional[Tuple[str, str]]:
        """Extract `(timestamp_str, building_id)` identity from an event."""
        if isinstance(event, EnergyLog):
            return str(event.timestamp), event.building_id
        if isinstance(event, dict) and ("Timestamp" in event or "timestamp" in event):
            return str(event.get("Timestamp") or event.get("timestamp")), str(event.get("building_id", "UNASSIGNED"))
        return None

    def _check_delivery_guarantees(self, event: Any) -> bool:
        """
        Enforce effectively-once stream semantics:
        - Detect and drop duplicate messages.
        - Detect and log out-of-order arrivals against per-building high-watermarks.

        Returns True if the event is unique and should be emitted; False if duplicate.
        """
        event_key = self._extract_event_identity(event)
        if event_key is None:
            return True

        ts_str, bld_id = event_key

        # Check for duplicates
        if self.deduplicate_stream and event_key in self._seen_event_keys:
            self.__duplicate_count += 1
            self._record_dropped()
            self.logger.debug(f"Duplicate event dropped for {event_key}")
            return False

        self._seen_event_keys.add(event_key)  # Mark this event as seen for future duplicate detection

        last_ts = self._high_watermarks.get(bld_id)  # Check for out-of-order events based on the last seen timestamp for this building
        if last_ts is not None and ts_str < last_ts:
            self.__out_of_order_count += 1
            self.logger.debug(f"Out-of-order event detected for {bld_id}: {ts_str} < {last_ts}")
        else:
            self._high_watermarks[bld_id] = ts_str

        return True

    def process(self, data: Any) -> Any:
        """
        Pass an event (or DataFrame) sequentially through all composed processors.
        If any filter stage returns None, short-circuit and drop the event.
        """
        current = data
        try:
            for processor in self.processors: # Pass the event through each processor in the pipeline
                if current is None:
                    self._record_dropped() 
                    return None
                current = processor.process(current)
        except EnergyPipelineError:
            raise
        except Exception as exc:
            self.logger.error(f"Pipeline stage error: {exc}")
            raise StreamProcessingError(f"Pipeline execution failed: {exc}") from exc

        if current is None:
            self._record_dropped()
        else:
            self._record_processed()
        return current

    def process_stream(self, stream: Iterable[Any]) -> Iterator[Any]:  # Process a stream of events
        """
        Generator that lazily consumes an input stream on the fly and yields
        only valid, unique, transformed events (skipping filtered or duplicate events).
        """
        self.reset_stream_state()
        for event in stream:
            result = self.process(event)  
            if result is not None and self._check_delivery_guarantees(result):
                yield result  # yield only valid, non-duplicate events to the consumer of this generator

    def __str__(self) -> str:  # Return a string representation of the pipeline
        """Return a readable diagram of the composed pipeline stages and counters."""
        stage_names = " -> ".join(p.name for p in self.processors) or "Empty"
        return (
            f"StreamPipeline([{stage_names}], "
            f"processed={self.processed_count}, dropped={self.dropped_count}, "
            f"duplicates={self.__duplicate_count}, out_of_order={self.__out_of_order_count})"
        )
