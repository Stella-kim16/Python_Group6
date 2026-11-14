"""Metrics Calculator for energy cost, carbon footprint, and anomaly detection."""

from datetime import datetime
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from subsystem1.pipeline import StreamProcessor


class MetricsCalculator(StreamProcessor):
    """
    Computes financial costs, carbon emissions, and anomaly flags using the existing CSV/DB columns.

    Inherits from `StreamProcessor` to support polymorphic stream and batch transformations.
    """

    # Class attributes defining domain defaults across instances
    DEFAULT_EMISSION_FACTOR: float = 0.42  # kg CO2 per kWh of non-renewable grid electricity
    DEFAULT_PEAK_TARIFF: float = 0.30      # EUR per kWh during weekday peak hours (08:00 - 20:00)
    DEFAULT_OFF_PEAK_TARIFF: float = 0.15  # EUR per kWh during nights, weekends, and holidays
    STREAM_ANOMALY_KWH_THRESHOLD: float = 95.0

    def __init__(
        self,
        emission_factor: float = DEFAULT_EMISSION_FACTOR,
        peak_tariff: float = DEFAULT_PEAK_TARIFF,
        off_peak_tariff: float = DEFAULT_OFF_PEAK_TARIFF,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "MetricsCalculator", logger=logger)
        self.emission_factor = float(emission_factor)
        self.peak_tariff = float(peak_tariff)
        self.off_peak_tariff = float(off_peak_tariff)

    def is_peak_tariff(
        self, timestamp: Union[datetime, pd.Timestamp, str], holiday: str = "No"
    ) -> bool:
        """Determine whether a timestamp falls within peak-tariff hours (weekday 08:00-20:00, non-holiday)."""
        ts = pd.to_datetime(timestamp)
        is_weekend = ts.weekday() >= 5
        if is_weekend or str(holiday).strip().title() == "Yes":
            return False
        return bool(8 <= ts.hour < 20)

    def get_tariff_rate(self, timestamp: Union[datetime, pd.Timestamp], holiday: str = "No") -> float:
        """Return dynamic tariff rate (EUR/kWh) based on time-of-use and weekday/weekend schedule."""
        if self.is_peak_tariff(timestamp, holiday=holiday):
            return self.peak_tariff
        return self.off_peak_tariff

    def calculate_carbon_emission(self, estimated_kwh: float, renewable_pct: float) -> float:
        """Calculate carbon footprint (kg CO2) factoring in the building's renewable energy percentage."""
        non_renewable_share = max(0.0, min(1.0, 1.0 - (float(renewable_pct) / 100.0)))
        carbon_kg = float(estimated_kwh) * non_renewable_share * self.emission_factor
        return round(carbon_kg, 4)

    def detect_anomalies(self, df: pd.DataFrame, z_threshold: float = 2.2) -> pd.Series:
        """
        Flag anomalous telemetry records where:
        1. Consumption exceeds the building's mean baseline by > z_threshold standard deviations, OR
        2. Building is completely unoccupied (Occupancy == 0) while both HVAC and Lighting are On.
        """
        grouped = df.groupby("building_id")["EnergyConsumption"]
        b_mean = grouped.transform("mean")
        b_std = grouped.transform("std").replace(0, 1.0)
        z_scores = (df["EnergyConsumption"] - b_mean) / b_std

        unoccupied_waste = (
            (df["Occupancy"] == 0)
            & (df["HVACUsage"] == "On")
            & (df["LightingUsage"] == "On")
        )
        return (z_scores > z_threshold) | unoccupied_waste

    def enrich_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cost_per_kwh, energy_cost_eur, carbon_footprint_kg, and is_anomaly columns to DataFrame."""
        out = df.copy()
        hours = pd.to_datetime(out["Timestamp"]).dt.hour
        is_weekend = out["DayOfWeek"].isin(["Saturday", "Sunday"])
        is_holiday = out["Holiday"] == "Yes"

        is_peak = (~is_weekend) & (~is_holiday) & (hours >= 8) & (hours < 20)
        out["cost_per_kwh"] = np.where(is_peak, self.peak_tariff, self.off_peak_tariff)
        out["energy_cost_eur"] = (out["EnergyConsumption"] * out["cost_per_kwh"]).round(4)

        non_ren_share = np.clip(1.0 - (out["RenewableEnergy"] / 100.0), 0.0, 1.0)
        out["carbon_footprint_kg"] = (
            out["EnergyConsumption"] * non_ren_share * self.emission_factor
        ).round(4)

        out["is_anomaly"] = self.detect_anomalies(out)
        return out

    def process(
        self, data: Union[Dict[str, Any], pd.DataFrame, None]
    ) -> Union[Dict[str, Any], pd.DataFrame, None]:
        """
        Polymorphic `StreamProcessor.process` implementation:
        Transforms valid energy metrics into updated tariff rates (`cost_per_kwh`),
        costs (`energy_cost_eur`), carbon equivalents (`carbon_footprint_kg`),
        and anomaly flags (`is_anomaly`).
        """
        if data is None:
            self._record_dropped()
            return None

        if isinstance(data, pd.DataFrame):
            enriched_df = self.enrich_metrics(data)
            self._record_processed(len(enriched_df))
            return enriched_df

        if isinstance(data, dict):
            event = dict(data)
            ts = pd.to_datetime(event.get("Timestamp", event.get("timestamp", datetime.now())))
            holiday = str(event.get("Holiday", event.get("holiday", "No"))).strip().title()
            kwh = float(event.get("EnergyConsumption", event.get("energy_consumption_kwh", 0.0)))
            ren_pct = float(event.get("RenewableEnergy", event.get("renewable_energy_pct", 0.0)))

            rate = self.get_tariff_rate(ts, holiday=holiday)
            cost_eur = round(kwh * rate, 4)
            carbon_kg = self.calculate_carbon_emission(kwh, ren_pct)

            occ = int(event.get("Occupancy", event.get("occupancy", 0)))
            hvac = str(event.get("HVACUsage", event.get("hvac_usage", "Off")))
            light = str(event.get("LightingUsage", event.get("lighting_usage", "Off")))
            is_anom = bool(
                (occ == 0 and hvac == "On" and light == "On")
                or kwh > self.STREAM_ANOMALY_KWH_THRESHOLD
            )

            event["cost_per_kwh"] = rate
            event["energy_cost_eur"] = cost_eur
            event["carbon_footprint_kg"] = carbon_kg
            event["is_anomaly"] = is_anom
            self._record_processed()
            return event

        raise TypeError(f"Unsupported input type for MetricsCalculator.process: {type(data)}")

    def __str__(self) -> str:
        """Return a readable summary of the calculator's tariff and emission configuration."""
        return (
            f"{self.name}(peak={self.peak_tariff:.2f} EUR/kWh, "
            f"off_peak={self.off_peak_tariff:.2f} EUR/kWh, "
            f"emission_factor={self.emission_factor} kg/kWh, "
            f"processed={self.processed_count})"
        )
