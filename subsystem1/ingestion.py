"""Data Ingestion Engine, Smart Meter Stream Iterator, and Sub-Threshold Filter."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
import pandas as pd
from subsystem1.models import (
    DataValidationError,
    EnergyLog,
    StreamProcessor,
)


REQUIRED_RAW_COLUMNS: List[str] = [
    "Timestamp",
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "HVACUsage",
    "LightingUsage",
    "RenewableEnergy",
    "DayOfWeek",
    "Holiday",
    "EnergyConsumption",
]


class SmartMeterStream:
    """
    Custom Iterator and Generator for consuming
    high-frequency telemetry readings from smart meters across buildings on the fly.
    Uses only the exact columns present in the CSV dataset.

    Demonstrates:
    - Encapsulation: Protected (`_records`, `_index`) and private (`__emitted_count`) state.
    - Iterator Protocol (`__iter__`, `__next__`) & Generator (`event_generator`).
    """

    def __init__(self, source: Union[str, Path, pd.DataFrame, Iterable[Dict[str, Any]]]) -> None:
        self._source_name: str = str(source) if isinstance(source, (str, Path)) else type(source).__name__
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Stream source file not found: {path}")
            df = pd.read_csv(path)
            self._records: List[Dict[str, Any]] = df.to_dict(orient="records")
        elif isinstance(source, pd.DataFrame):
            self._records = source.to_dict(orient="records")
        else:
            self._records = list(source)
        self._index: int = 0
        self.__emitted_count: int = 0

    @property
    def total_records(self) -> int:
        """Return total number of buffered telemetry events in the stream source."""
        return len(self._records)

    @property
    def emitted_count(self) -> int:
        """Return count of events emitted during the current iteration cycle."""
        return self.__emitted_count

    def __len__(self) -> int:
        """Return total number of events available in the stream."""
        return len(self._records)

    def __iter__(self) -> "SmartMeterStream":
        """Reset stream cursor and return the iterator instance."""
        self._index = 0
        self.__emitted_count = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """Return the next smart meter reading or raise StopIteration."""
        if self._index >= len(self._records):
            raise StopIteration
        raw = dict(self._records[self._index])
        self._index += 1
        self.__emitted_count += 1
        return raw

    def event_generator(self) -> Iterator[Dict[str, Any]]:
        """Generator function (`yield`) emitting smart meter events one by one."""
        for record in self:
            yield record

    def __str__(self) -> str:
        """Return a human-readable summary of the smart meter stream state."""
        return (
            f"SmartMeterStream(source='{self._source_name}', "
            f"cursor={self._index}/{len(self._records)}, emitted={self.__emitted_count})"
        )


class SubThresholdMeterFilter(StreamProcessor):
    """
    Specialized StreamProcessor that filters out inactive or sub-threshold
    smart meter readings based on `EnergyConsumption` (and optional HVAC/Lighting
    activity) using the existing CSV/DB columns (Pipeline Goal 1).
    """

    # Class attribute defining default threshold across instances
    DEFAULT_MIN_POWER_KWH: float = 25.0

    def __init__(
        self,
        min_power_kwh: float = DEFAULT_MIN_POWER_KWH,
        require_active_systems: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name="SubThresholdMeterFilter", logger=logger)
        self.min_power_kwh = float(min_power_kwh)
        self.require_active_systems = require_active_systems

    def process(
        self, data: Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]
    ) -> Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]:
        """
        Polymorphic `process` implementation:
        - Single event (dict or EnergyLog): returns None if power usage is below
          `min_power_kwh` (or if meter systems are inactive), otherwise returns the event.
        - Batch DataFrame: filters out invalid or sub-threshold rows.
        """
        if data is None:
            self._record_dropped()
            return None

        if isinstance(data, pd.DataFrame):
            initial_len = len(data)
            df = data.copy()
            if "EnergyConsumption" in df.columns:
                df = df[pd.to_numeric(df["EnergyConsumption"], errors="coerce") >= self.min_power_kwh]
            if self.require_active_systems and {"HVACUsage", "LightingUsage"}.issubset(df.columns):
                df = df[(df["HVACUsage"] == "On") | (df["LightingUsage"] == "On")]
            kept = len(df)
            self._record_processed(kept)
            self._record_dropped(max(0, initial_len - kept))
            return df.reset_index(drop=True)

        if isinstance(data, EnergyLog):
            if data.energy_consumption_kwh < self.min_power_kwh or not data.validate_data_types():
                self._record_dropped()
                return None
            if (
                self.require_active_systems
                and data.hvac_usage == "Off"
                and data.lighting_usage == "Off"
            ):
                self._record_dropped()
                return None
            self._record_processed()
            return data

        if isinstance(data, dict):
            try:
                power_kwh = float(data.get("EnergyConsumption", data.get("energy_consumption_kwh", -1.0)))
                sqft = float(data.get("SquareFootage", data.get("area_sqft", 1.0)))
                hum = float(data.get("Humidity", data.get("humidity", 50.0)))
                ren = float(data.get("RenewableEnergy", data.get("renewable_energy_pct", 0.0)))
            except (TypeError, ValueError):
                self._record_dropped()
                return None

            if (
                power_kwh < self.min_power_kwh
                or sqft <= 0
                or not (0.0 <= hum <= 100.0)
                or not (0.0 <= ren <= 100.0)
            ):
                self._record_dropped()
                return None

            if self.require_active_systems:
                hvac = str(data.get("HVACUsage", data.get("hvac_usage", "Off"))).strip().title()
                light = str(data.get("LightingUsage", data.get("lighting_usage", "Off"))).strip().title()
                if hvac == "Off" and light == "Off":
                    self._record_dropped()
                    return None

            self._record_processed()
            return data

        raise TypeError(f"Unsupported data type for SubThresholdMeterFilter: {type(data)}")

    def __str__(self) -> str:
        """Return a readable summary of the filter configuration and runtime counts."""
        return (
            f"SubThresholdMeterFilter(min_power_kwh={self.min_power_kwh}, "
            f"require_active={self.require_active_systems}, "
            f"processed={self.processed_count}, dropped={self.dropped_count})"
        )


class DataIngestionEngine(StreamProcessor):
    """
    Ingests, validates, filters, and streams raw energy telemetry data.

    Inherits from `StreamProcessor` (ABC) and implements `__iter__` and `__next__`
    for direct stream consumption as well as batch CSV ingestion.
    """

    # Class-level schema contract shared by all DataIngestionEngine instances
    REQUIRED_COLUMNS: List[str] = REQUIRED_RAW_COLUMNS

    def __init__(
        self,
        raw_file_path: Union[str, Path, None] = None,
        min_power_threshold: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name="DataIngestionEngine", logger=logger)
        self.raw_file_path = Path(raw_file_path) if raw_file_path else None
        self.min_power_threshold = float(min_power_threshold)
        self._stream_records: List[Dict[str, Any]] = []
        self._stream_index: int = 0

    def __iter__(self) -> "DataIngestionEngine":
        """Initialize iterator over the cleaned telemetry records (`__iter__`)."""
        if self.raw_file_path is None:
            raise DataValidationError("Cannot iterate DataIngestionEngine without a `raw_file_path`.")
        df = self.import_csv_batch(self.raw_file_path)
        self._stream_records = df.to_dict(orient="records")
        self._stream_index = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """Yield the next telemetry event from the stream (`__next__`)."""
        if self._stream_index >= len(self._stream_records):
            raise StopIteration
        event = dict(self._stream_records[self._stream_index])
        self._stream_index += 1
        return event

    def stream_events(
        self, file_path: Union[str, Path, None] = None
    ) -> Iterator[Dict[str, Any]]:
        """Generator (`yield`) that streams validated meter readings on the fly."""
        target = Path(file_path) if file_path else self.raw_file_path
        if target is None:
            raise DataValidationError("File path is required for streaming events.")
        df = self.import_csv_batch(target)
        for row in df.to_dict(orient="records"):
            yield row

    def process(
        self, data: Union[Dict[str, Any], pd.DataFrame, str, Path, None]
    ) -> Union[Dict[str, Any], pd.DataFrame, None]:
        """
        Uniform `StreamProcessor.process` implementation:
        - File path (str / Path): loads, cleans, and deduplicates CSV batch.
        - DataFrame: cleans, validates, and filters sub-threshold meter readings.
        - Single event (dict): validates schema, filters out sub-threshold meter
          readings (`EnergyConsumption < min_power_threshold`), and returns the
          cleaned event dict (or None if filtered out).
        """
        if data is None:
            self._record_dropped()
            return None

        if isinstance(data, (str, Path)):
            df = self.import_csv_batch(data)
            self._record_processed(len(df))
            return df

        if isinstance(data, pd.DataFrame):
            initial_len = len(data)
            cleaned = self.clean_missing_data(data)
            deduped = self.deduplicate_timestamps(cleaned)
            self._record_processed(len(deduped))
            self._record_dropped(max(0, initial_len - len(deduped)))
            return deduped

        if isinstance(data, dict):
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data]
            if missing_cols:
                self._record_dropped()
                return None
            df = pd.DataFrame([data])
            cleaned_df = self.clean_missing_data(df)
            if cleaned_df.empty:
                self._record_dropped()
                return None
            self._record_processed()
            return cleaned_df.iloc[0].to_dict()

        raise TypeError(f"Unsupported input type for DataIngestionEngine.process: {type(data)}")

    def import_csv_batch(self, file_path: Union[str, Path, None] = None) -> pd.DataFrame:
        """Load a batch CSV file, validate schema, deduplicate, and clean invalid rows."""
        target_path = Path(file_path) if file_path else self.raw_file_path
        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"CSV file not found: {target_path}")

        raw_df = pd.read_csv(target_path)
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in raw_df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns in CSV: {missing_cols}")

        cleaned_df = self.clean_missing_data(raw_df)
        deduped_df = self.deduplicate_timestamps(cleaned_df)
        self.log_audit(
            f"Ingested batch from {target_path.name}: {len(deduped_df)}/{len(raw_df)} valid rows."
        )
        return deduped_df

    def ingest_sensor_stream(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Validate and convert a single real-time JSON/dict sensor payload into a 1-row DataFrame."""
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in payload]
        if missing_cols:
            raise DataValidationError(f"Sensor payload missing required keys: {missing_cols}")

        df = pd.DataFrame([payload])
        cleaned_df = self.clean_missing_data(df)
        if cleaned_df.empty:
            raise DataValidationError("Sensor payload failed data validation checks.")
        return cleaned_df

    def clean_missing_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Filter or impute missing, negative, or out-of-range telemetry values."""
        df = raw_data.copy()

        # Parse Timestamp and drop rows with invalid timestamps
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        # Coerce numeric columns
        numeric_cols = [
            "Temperature",
            "Humidity",
            "SquareFootage",
            "Occupancy",
            "RenewableEnergy",
            "EnergyConsumption",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Interpolate or fill minor missing numeric sensor readings if any exist
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()

        # Filter out physically invalid / negative or sub-threshold values
        valid_mask = (
            (df["SquareFootage"] > 0)
            & (df["Occupancy"] >= 0)
            & (df["Humidity"] >= 0.0)
            & (df["Humidity"] <= 100.0)
            & (df["RenewableEnergy"] >= 0.0)
            & (df["RenewableEnergy"] <= 100.0)
            & (df["EnergyConsumption"] >= self.min_power_threshold)
        )
        df = df.loc[valid_mask].copy()

        # Normalize categorical usage strings
        df["HVACUsage"] = df["HVACUsage"].astype(str).str.strip().str.title()
        df["LightingUsage"] = df["LightingUsage"].astype(str).str.strip().str.title()
        df = df[df["HVACUsage"].isin(["On", "Off"]) & df["LightingUsage"].isin(["On", "Off"])].copy()

        df["Occupancy"] = df["Occupancy"].round().astype(int)
        return df.reset_index(drop=True)

    def deduplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove redundant telemetry records based on timestamps (and building_id if present)
        and sort chronologically to handle out-of-order arrivals.
        """
        subset_cols = ["Timestamp", "building_id"] if "building_id" in df.columns else ["Timestamp"]
        out = df.sort_values("Timestamp").drop_duplicates(subset=subset_cols, keep="first")
        return out.reset_index(drop=True)

    def __str__(self) -> str:
        """Return a readable summary of the ingestion engine configuration."""
        src = str(self.raw_file_path) if self.raw_file_path else "None"
        return (
            f"DataIngestionEngine(source='{src}', min_power={self.min_power_threshold}, "
            f"processed={self.processed_count}, dropped={self.dropped_count})"
        )

