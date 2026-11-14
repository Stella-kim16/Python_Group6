"""Data Ingestion Engine and Smart Meter Stream Iterator."""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import pandas as pd
from subsystem1.exceptions_subsystem1 import DataValidationError
from subsystem1.pipeline import StreamProcessor

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


class SmartMeterStream: # Reads and iterates over events from CSV or DataFrame sources.
    """
    Custom Iterator and Generator for consuming
    high-frequency telemetry readings from smart meters across buildings on the fly.
    Uses only the exact columns present in the CSV dataset.
    """

    def __init__(self, source: Union[str, Path, pd.DataFrame, Iterable[Dict[str, Any]]]) -> None:
        self._source_name: str = str(source) if isinstance(source, (str, Path)) else type(source).__name__
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Stream source file not found: {path}")
            df = pd.read_csv(path)
            self._records: List[Dict[str, Any]] = df.to_dict(orient="records") # convert df to list of dicts for iteration
        elif isinstance(source, pd.DataFrame):
            self._records = source.to_dict(orient="records")
        else:
            self._records = list(source)
        self._index: int = 0
        self.__emitted_count: int = 0

    @property
    def emitted_count(self) -> int:
        """Return count of events emitted during the current iteration cycle."""
        return self.__emitted_count

    def __len__(self) -> int:
        """Return total number of events available in the stream."""
        return len(self._records)

    def __iter__(self) -> "SmartMeterStream": # for loop will call this method first to reset the cursor and return the iterator instance
        """Reset stream cursor and return the iterator instance."""
        self._index = 0
        self.__emitted_count = 0
        return self

    def __next__(self) -> Dict[str, Any]: # then the for loop will call this method repeatedly to get the next event until StopIteration is raised
        """Return the next smart meter reading or raise StopIteration."""
        if self._index >= len(self._records):
            raise StopIteration
        raw = dict(self._records[self._index])
        self._index += 1
        self.__emitted_count += 1
        return raw

    def __str__(self) -> str:
        """Return a human-readable summary of the smart meter stream state"""
        return (
            f"SmartMeterStream(source='{self._source_name}', "
            f"cursor={self._index}/{len(self._records)}, emitted={self.__emitted_count})"
        )


class DataIngestionEngine(StreamProcessor): # first stage of the pipeline 
    """
    Ingests, validates, filters, and cleans raw energy telemetry data.
    Inherits from `StreamProcessor` for batch CSV and stream event ingestion.
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

    def process(
        self, data: Union[Dict[str, Any], pd.DataFrame, str, Path, None]
    ) -> Union[Dict[str, Any], pd.DataFrame, None]:
        """
        Uniform `StreamProcessor.process` implementation:
        - File path (str / Path): loads, cleans, and deduplicates CSV batch.
        - DataFrame: cleans, validates, and deduplicates telemetry rows.
        - Single event (dict): validates schema and physical bounds for streaming.
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
            self._record_processed(len(cleaned))
            self._record_dropped(max(0, initial_len - len(cleaned)))
            return cleaned

        if isinstance(data, dict):
            if float(data.get("EnergyConsumption", -1.0)) < self.min_power_threshold:
                self._record_dropped()
                return None
            self._record_processed()
            return data

        raise TypeError(f"Unsupported input type for DataIngestionEngine.process: {type(data)}")

    def import_csv_batch(self, file_path: Union[str, Path, None] = None) -> pd.DataFrame:
        """Load a batch CSV file, validate schema, and clean invalid or duplicate rows."""
        target_path = Path(file_path) if file_path else self.raw_file_path
        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"CSV file not found: {target_path}")

        raw_df = pd.read_csv(target_path)
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in raw_df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns in CSV: {missing_cols}")

        cleaned_df = self.clean_missing_data(raw_df)
        self.logger.debug(
            f"Ingested batch from {target_path.name}: {len(cleaned_df)}/{len(raw_df)} valid rows."
        )
        return cleaned_df

    def clean_missing_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Filter or impute missing, negative, out-of-range, or duplicate values."""
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
        df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="first")
        return df.reset_index(drop=True)

    def __str__(self) -> str:
        """Return a readable summary of the ingestion engine configuration."""
        src = str(self.raw_file_path) if self.raw_file_path else "None"
        return (
            f"DataIngestionEngine(source='{src}', min_power={self.min_power_threshold}, "
            f"processed={self.processed_count}, dropped={self.dropped_count})"
        )

