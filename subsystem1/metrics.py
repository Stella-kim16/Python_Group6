"""Metrics Calculator and Stream Transformer for energy cost, carbon footprint, and anomaly detection."""

from datetime import datetime
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from subsystem1.models import EnergyLog, StreamProcessor


class MetricsCalculator(StreamProcessor):
    """
    Computes financial costs (peak/off-peak tariff), carbon emissions,
    anomaly flags, and sustainability scores using the existing CSV/DB columns.

    Inherits from `StreamProcessor` (ABC) to support polymorphic stream and batch
    transformations (Pipeline Goal 2).
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

    def process(
        self, data: Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]
    ) -> Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]:
        """
        Polymorphic `StreamProcessor.process` implementation (Pipeline Goal 2):
        Transforms valid energy metrics into updated tariff rates (`cost_per_kwh`),
        costs (`energy_cost_eur`), carbon equivalents (`carbon_footprint_kg`),
        and anomaly flags (`is_anomaly`) using only existing CSV/DB schema fields.
        """
        if data is None:
            self._record_dropped()
            return None

        if isinstance(data, pd.DataFrame):
            enriched_df = self.enrich_metrics(data)
            self._record_processed(len(enriched_df))
            return enriched_df

        if isinstance(data, EnergyLog):
            data.energy_cost_eur = self.calculate_energy_cost(
                data.energy_consumption_kwh, data.timestamp, holiday=data.holiday
            )
            data.carbon_footprint_kg = self.calculate_carbon_emission(
                data.energy_consumption_kwh, data.renewable_energy_pct
            )
            data.is_anomaly = bool(
                (data.occupancy == 0 and data.hvac_usage == "On" and data.lighting_usage == "On")
                or data.energy_consumption_kwh > self.STREAM_ANOMALY_KWH_THRESHOLD
            )
            self._record_processed()
            return data

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

    def estimate_kwh(
        self,
        hvac: str,
        lighting: str,
        sqft: float,
        temperature: float = 24.0,
        humidity: float = 45.0,
        occupancy: int = 5,
        max_occupancy: int = 35,
    ) -> float:
        """Estimate hourly energy consumption (kWh) for a single state (used by UML / stream estimator)."""
        occ_rate = min(max(occupancy / max(max_occupancy, 1), 0.0), 1.0)
        base_load = 0.026 * sqft
        thermal_load = 1.65 * (temperature - 20.0) + 0.12 * (humidity - 30.0)
        occupancy_load = 12.0 * occ_rate + 0.18 * occupancy
        hvac_load = (11.5 + 0.55 * (temperature - 20.0)) if hvac == "On" else 0.0
        lighting_load = (4.5 + 0.002 * sqft) if lighting == "On" else 0.0
        return round(max(base_load + thermal_load + occupancy_load + hvac_load + lighting_load, 25.0), 4)

    def get_tariff_rate(self, timestamp: Union[datetime, pd.Timestamp], holiday: str = "No") -> float:
        """Return dynamic tariff rate (EUR/kWh) based on time-of-use and weekday/weekend schedule."""
        if self.is_peak_tariff(timestamp, holiday=holiday):
            return self.peak_tariff
        return self.off_peak_tariff

    def calculate_energy_cost(
        self,
        estimated_kwh: float,
        timestamp: Union[datetime, pd.Timestamp],
        holiday: str = "No",
    ) -> float:
        """Calculate total energy cost (EUR) for a given consumption and timestamp."""
        rate = self.get_tariff_rate(timestamp, holiday=holiday)
        return round(float(estimated_kwh) * rate, 4)

    def calculate_carbon_emission(self, estimated_kwh: float, renewable_pct: float) -> float:
        """
        Calculate carbon footprint (kg CO2) factoring in the building's renewable energy percentage.
        """
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

    def calculate_sustainability_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute a 0-100 Sustainability Score per building based on:
        - Renewable energy utilization (40 pts)
        - Off-hours equipment discipline when unoccupied (30 pts)
        - Carbon efficiency per sq ft (30 pts)
        """
        scores: Dict[str, float] = {}
        for bid, group in df.groupby("building_id"):
            ren_score = min(40.0, (group["RenewableEnergy"].mean() / 25.0) * 40.0)

            unocc = group[group["Occupancy"] == 0]
            if len(unocc) > 0:
                waste_ratio = ((unocc["HVACUsage"] == "On") | (unocc["LightingUsage"] == "On")).mean()
                discipline_score = 30.0 * (1.0 - waste_ratio)
            else:
                discipline_score = 25.0

            carbon_per_ksqft = (group["carbon_footprint_kg"] / (group["SquareFootage"] / 1000.0)).mean()
            carbon_score = max(0.0, min(30.0, 30.0 * (1.0 - (carbon_per_ksqft - 8.0) / 22.0)))

            scores[str(bid)] = round(ren_score + discipline_score + carbon_score, 1)
        return scores

    def __str__(self) -> str:
        """Return a readable summary of the calculator's tariff and emission configuration."""
        return (
            f"{self.name}(peak={self.peak_tariff:.2f} EUR/kWh, "
            f"off_peak={self.off_peak_tariff:.2f} EUR/kWh, "
            f"emission_factor={self.emission_factor} kg/kWh, "
            f"processed={self.processed_count})"
        )


class LoadBalanceAndTariffTransformer(MetricsCalculator):
    """
    Specialized StreamProcessor subclass dedicated to transforming valid smart meter
    events into updated tariff rates, energy costs, carbon equivalents, and anomaly flags.

    Demonstrates multi-level inheritance (`StreamProcessor` -> `MetricsCalculator`
    -> `LoadBalanceAndTariffTransformer`) and `super().__init__()` delegation.
    """

    def __init__(
        self,
        emission_factor: float = MetricsCalculator.DEFAULT_EMISSION_FACTOR,
        peak_tariff: float = MetricsCalculator.DEFAULT_PEAK_TARIFF,
        off_peak_tariff: float = MetricsCalculator.DEFAULT_OFF_PEAK_TARIFF,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            emission_factor=emission_factor,
            peak_tariff=peak_tariff,
            off_peak_tariff=off_peak_tariff,
            name="LoadBalanceAndTariffTransformer",
            logger=logger,
        )

