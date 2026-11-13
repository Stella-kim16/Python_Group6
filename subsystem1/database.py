"""Database Manager for SQLite persistence, time-series queries, and CSV/JSON exports."""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Union
import pandas as pd
from subsystem1.models import (
    Building,
    DatabaseOperationError,
    DataValidationError,
    EnergyLog,
)


class DatabaseManager:
    """
    Manages SQLite relational storage and file exports for buildings and energy telemetry.

    Demonstrates:
    - Encapsulation: Private `__connection_string` with `@property` getter and protected `_memory_conn`.
    - Dependency Injection: Accepts `connection_string` (including `":memory:"` for isolated testing)
      and `logger` via `__init__`.
    - Fault Tolerance & Idempotency: Uses `INSERT OR REPLACE` on composite primary key
      `(Timestamp, building_id)` for idempotent stream/batch writes and wraps SQLite errors
      in `DatabaseOperationError`.
    """

    DEFAULT_DB_PATH: str = "data/processed/smart_energy.db"

    def __init__(
        self,
        connection_string: Union[str, Path] = DEFAULT_DB_PATH,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.__connection_string: str = str(connection_string)
        self.logger: logging.Logger = logger or logging.getLogger("subsystem1.DatabaseManager")
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.__connection_string == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            Path(self.__connection_string).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def connection_string(self) -> str:
        """Read-only property exposing the configured SQLite connection string."""
        return self.__connection_string

    def _get_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.__connection_string)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        """Create buildings and energy_logs tables with foreign keys and time-series indexes."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS buildings (
                        building_id TEXT PRIMARY KEY,
                        building_type TEXT NOT NULL,
                        area_sqft REAL NOT NULL,
                        max_occupancy INTEGER NOT NULL,
                        baseline_kwh REAL NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS energy_logs (
                        Timestamp TEXT NOT NULL,
                        building_id TEXT NOT NULL,
                        building_type TEXT NOT NULL,
                        SquareFootage REAL NOT NULL,
                        max_occupancy INTEGER NOT NULL,
                        Temperature REAL NOT NULL,
                        Humidity REAL NOT NULL,
                        Occupancy INTEGER NOT NULL,
                        HVACUsage TEXT NOT NULL,
                        LightingUsage TEXT NOT NULL,
                        RenewableEnergy REAL NOT NULL,
                        DayOfWeek TEXT NOT NULL,
                        Holiday TEXT NOT NULL,
                        EnergyConsumption REAL NOT NULL,
                        cost_per_kwh REAL NOT NULL,
                        energy_cost_eur REAL NOT NULL,
                        carbon_footprint_kg REAL NOT NULL,
                        is_anomaly INTEGER NOT NULL,
                        PRIMARY KEY (Timestamp, building_id),
                        FOREIGN KEY (building_id) REFERENCES buildings (building_id)
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_logs_building_time ON energy_logs (building_id, Timestamp);"
                )
        except sqlite3.Error as exc:
            raise DatabaseOperationError(f"Failed to initialize SQLite schema: {exc}") from exc

    def save_building_record(self, building: Building) -> None:
        """Insert or update a single Building record in SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO buildings
                    (building_id, building_type, area_sqft, max_occupancy, baseline_kwh)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        building.building_id,
                        building.building_type,
                        building.area_sqft,
                        building.max_occupancy,
                        building.baseline_kwh,
                    ),
                )
        except sqlite3.Error as exc:
            raise DatabaseOperationError(
                f"Failed to save building {building.building_id}: {exc}"
            ) from exc

    def save_buildings(self, buildings: List[Building]) -> None:
        """Batch save a list of Building instances."""
        for b in buildings:
            self.save_building_record(b)
        self.logger.debug(f"Saved {len(buildings)} building records to SQLite.")

    def save_energy_log(self, log: EnergyLog) -> None:
        """Validate and save a single EnergyLog domain object into SQLite."""
        if not log.validate_data_types():
            raise DataValidationError(f"Invalid EnergyLog record at {log.timestamp}")

        cost_per_kwh = round(log.energy_cost_eur / max(log.energy_consumption_kwh, 1e-6), 4)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO energy_logs (
                        Timestamp, building_id, building_type, SquareFootage, max_occupancy,
                        Temperature, Humidity, Occupancy, HVACUsage, LightingUsage,
                        RenewableEnergy, DayOfWeek, Holiday, EnergyConsumption, cost_per_kwh,
                        energy_cost_eur, carbon_footprint_kg, is_anomaly
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        log.building_id,
                        log.building_type,
                        log.area_sqft,
                        max(log.occupancy, 1),
                        log.temperature,
                        log.humidity,
                        int(log.occupancy),
                        log.hvac_usage,
                        log.lighting_usage,
                        log.renewable_energy_pct,
                        log.day_of_week,
                        log.holiday,
                        log.energy_consumption_kwh,
                        cost_per_kwh,
                        log.energy_cost_eur,
                        log.carbon_footprint_kg,
                        int(log.is_anomaly),
                    ),
                )
        except sqlite3.Error as exc:
            raise DatabaseOperationError(
                f"Failed to save EnergyLog for {log.building_id} at {log.timestamp}: {exc}"
            ) from exc

    def save_stream_events(self, events: Iterable[Union[Dict[str, Any], EnergyLog]]) -> int:
        """
        Idempotently persist a stream of processed event dictionaries or EnergyLog instances
        into SQLite using `INSERT OR REPLACE` on `(Timestamp, building_id)`.
        Returns the number of persisted events.
        """
        rows_to_insert = []
        for ev in events:
            if isinstance(ev, EnergyLog):
                self.save_energy_log(ev)
                rows_to_insert.append(ev)
                continue
            if isinstance(ev, dict):
                raw_ts = ev.get("Timestamp", ev.get("timestamp", datetime.now()))
                ts_str = (
                    raw_ts.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(raw_ts, (datetime, pd.Timestamp))
                    else pd.to_datetime(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
                )
                kwh = float(ev.get("EnergyConsumption", ev.get("energy_consumption_kwh", 0.0)))
                cost_eur = float(ev.get("energy_cost_eur", 0.0))
                cost_per_kwh = float(
                    ev.get("cost_per_kwh", round(cost_eur / max(kwh, 1e-6), 4))
                )
                rows_to_insert.append(
                    (
                        ts_str,
                        str(ev.get("building_id", "BLD_OFF_01")),
                        str(ev.get("building_type", "Office")),
                        float(ev.get("SquareFootage", ev.get("area_sqft", 1420.0))),
                        int(ev.get("max_occupancy", 35)),
                        float(ev.get("Temperature", ev.get("temperature", 22.0))),
                        float(ev.get("Humidity", ev.get("humidity", 45.0))),
                        int(ev.get("Occupancy", ev.get("occupancy", 0))),
                        str(ev.get("HVACUsage", ev.get("hvac_usage", "Off"))),
                        str(ev.get("LightingUsage", ev.get("lighting_usage", "Off"))),
                        float(ev.get("RenewableEnergy", ev.get("renewable_energy_pct", 0.0))),
                        str(ev.get("DayOfWeek", ev.get("day_of_week", "Monday"))),
                        str(ev.get("Holiday", ev.get("holiday", "No"))),
                        kwh,
                        cost_per_kwh,
                        cost_eur,
                        float(ev.get("carbon_footprint_kg", 0.0)),
                        int(bool(ev.get("is_anomaly", False))),
                    )
                )

        dict_rows = [r for r in rows_to_insert if isinstance(r, tuple)]
        if dict_rows:
            try:
                with self._get_connection() as conn:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO energy_logs (
                            Timestamp, building_id, building_type, SquareFootage, max_occupancy,
                            Temperature, Humidity, Occupancy, HVACUsage, LightingUsage,
                            RenewableEnergy, DayOfWeek, Holiday, EnergyConsumption, cost_per_kwh,
                            energy_cost_eur, carbon_footprint_kg, is_anomaly
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        dict_rows,
                    )
            except sqlite3.Error as exc:
                raise DatabaseOperationError(f"Failed to batch insert streamed events: {exc}") from exc

        return len(rows_to_insert)

    def save_energy_logs_df(self, df: pd.DataFrame) -> None:
        """Persist an entire feature-engineered and enriched DataFrame into SQLite."""
        to_save = df.copy()
        to_save["Timestamp"] = pd.to_datetime(to_save["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        to_save["Occupancy"] = to_save["Occupancy"].astype(int)
        to_save["max_occupancy"] = to_save["max_occupancy"].astype(int)
        to_save["is_anomaly"] = to_save["is_anomaly"].astype(int)

        try:
            with self._get_connection() as conn:
                conn.execute("DROP TABLE IF EXISTS energy_logs;")
                self._init_schema()
                to_save.to_sql("energy_logs", conn, if_exists="append", index=False)
            self.logger.debug(f"Persisted {len(to_save)} energy log rows to SQLite.")
        except sqlite3.Error as exc:
            raise DatabaseOperationError(f"Failed to persist energy logs DataFrame: {exc}") from exc

    def fetch_time_series_data(
        self,
        building_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query historical energy logs filtered by optional building_id and date range."""
        query = "SELECT * FROM energy_logs WHERE 1=1"
        params: List[str] = []

        if building_id:
            query += " AND building_id = ?"
            params.append(building_id)
        if start_date:
            query += " AND Timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND Timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY Timestamp ASC;"

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except sqlite3.Error as exc:
            raise DatabaseOperationError(f"Failed to query time-series data: {exc}") from exc

        if not df.empty:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["Occupancy"] = df["Occupancy"].astype(int)
            df["max_occupancy"] = df["max_occupancy"].astype(int)
            df["is_anomaly"] = df["is_anomaly"].astype(bool)
        return df

    def get_record_count(self) -> int:
        """Return the total number of energy_logs currently stored in SQLite."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM energy_logs;")
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def export_to_csv(
        self, file_path: Union[str, Path] = "data/processed/enriched_energy_data.csv"
    ) -> Path:
        """Export all processed energy logs from SQLite to a CSV file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df = self.fetch_time_series_data()
        if not df.empty:
            df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(target, index=False)
        return target

    def export_to_json(
        self, file_path: Union[str, Path] = "data/processed/enriched_energy_data.json"
    ) -> Path:
        """Export all processed energy logs from SQLite to a standard JSON file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df = self.fetch_time_series_data()
        if not df.empty:
            df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        records = df.to_dict(orient="records")
        target.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return target

    def __str__(self) -> str:
        """Return a human-readable description of the DatabaseManager and its row count."""
        return (
            f"DatabaseManager(connection='{self.__connection_string}', "
            f"stored_logs={self.get_record_count()})"
        )

