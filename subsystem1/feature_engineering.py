"""Feature Engineering and realistic building/occupancy simulation for Subsystem 1."""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from subsystem1.models import Building, EnergyLog, StreamProcessor
from subsystem1.exceptions_subsystem1 import DataProcessingError


DEFAULT_BUILDING_CATALOG: List[Building] = [
    Building("BLD_HOM_01", "Home", 1100.0, max_occupancy=4, baseline_kwh=28.0),
    Building("BLD_HOM_02", "Home", 1250.0, max_occupancy=6, baseline_kwh=32.0),
    Building("BLD_OFF_01", "Office", 1420.0, max_occupancy=35, baseline_kwh=48.0),
    Building("BLD_OFF_02", "Office", 1580.0, max_occupancy=50, baseline_kwh=55.0),
    Building("BLD_UNI_01", "University", 1750.0, max_occupancy=90, baseline_kwh=65.0),
    Building("BLD_UNI_02", "University", 1920.0, max_occupancy=120, baseline_kwh=74.0),
]


class FeatureEngineer(StreamProcessor):
    """
    Transforms raw energy logs into multi-building, schedule-aware telemetry 
    with robust custom exception handling.
    """

    def __init__(self, buildings: List[Building] | None = None, random_seed: int = 42) -> None:
        try:
            self.buildings = buildings or DEFAULT_BUILDING_CATALOG
            self.building_map: Dict[str, Building] = {b.building_id: b for b in self.buildings}
            self.random_seed = random_seed
        except Exception as e:
            raise DataProcessingError(f"Failed to initialize FeatureEngineer: {e}") from e

    def _resolve_building_for_sqft(self, sqft: float) -> Building:
        """Map a raw SquareFootage reading to its corresponding Building instance safely."""
        try:
            thresholds = [1165.0, 1330.0, 1500.0, 1665.0, 1830.0]
            for idx, limit in enumerate(thresholds):
                if sqft < limit and idx < len(self.buildings):
                    return self.buildings[idx]
            return self.buildings[-1]
        except Exception as e:
            raise DataProcessingError(f"Error resolving building for sqft {sqft}: {e}") from e

    def process(
        self, data: Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]
    ) -> Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]:
        """
        Polymorphic `StreamProcessor.process` implementation with exception safety.
        """
        try:
            if data is None:
                return None

            if isinstance(data, pd.DataFrame):
                return self.run_pipeline(data)

            if isinstance(data, EnergyLog):
                bld = self.building_map.get(data.building_id) or self._resolve_building_for_sqft(data.area_sqft)
                data.building_id = bld.building_id
                data.building_type = bld.building_type
                data.area_sqft = bld.area_sqft
                if data.day_of_week == "Sunday":
                    data.holiday = "Yes"
                return data

            if isinstance(data, dict):
                event = dict(data)
                raw_sqft = float(event.get("SquareFootage", 1400.0))
                bld_id = event.get("building_id")
                bld = (
                    self.building_map.get(str(bld_id))
                    if bld_id in self.building_map
                    else self._resolve_building_for_sqft(raw_sqft)
                )

                event["building_id"] = bld.building_id
                event["building_type"] = bld.building_type
                event["max_occupancy"] = int(bld.max_occupancy)
                event["SquareFootage"] = float(bld.area_sqft)

                day_of_week = str(event.get("DayOfWeek", "Monday")).strip().title()
                holiday = str(event.get("Holiday", "No")).strip().title()
                if day_of_week == "Sunday":
                    holiday = "Yes"
                event["DayOfWeek"] = day_of_week
                event["Holiday"] = holiday

                occ = int(round(float(event.get("Occupancy", 0))))
                event["Occupancy"] = max(0, min(occ, bld.max_occupancy))
                return event

            raise TypeError(f"Unsupported input type for FeatureEngineer.process: {type(data)}")

        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Error processing data in FeatureEngineer: {e}") from e

    def assign_buildings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map rows to specific buildings based on SquareFootage buckets."""
        try:
            out = df.copy()
            bins = [0.0, 1165.0, 1330.0, 1500.0, 1665.0, 1830.0, float("inf")]
            labels = [b.building_id for b in self.buildings]

            out["building_id"] = pd.cut(out["SquareFootage"], bins=bins, labels=labels, right=False).astype(str)
            out["building_type"] = out["building_id"].map(lambda bid: self.building_map[bid].building_type)
            out["max_occupancy"] = out["building_id"].map(lambda bid: self.building_map[bid].max_occupancy).astype(int)
            out["SquareFootage"] = out["building_id"].map(lambda bid: self.building_map[bid].area_sqft)
            return out
        except Exception as e:
            raise DataProcessingError(f"Failed to assign buildings in dataframe: {e}") from e

    def fix_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all Sundays are marked as Holiday = 'Yes'."""
        try:
            out = df.copy()
            out["Holiday"] = out["Holiday"].astype(str).str.strip().str.title()
            out.loc[out["DayOfWeek"] == "Sunday", "Holiday"] = "Yes"
            return out
        except Exception as e:
            raise DataProcessingError(f"Failed to fix calendar features: {e}") from e

    def simulate_realistic_occupancy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale integer headcount realistically by building type and schedule."""
        try:
            out = df.copy()
            rng = np.random.default_rng(self.random_seed)
            hours = pd.to_datetime(out["Timestamp"]).dt.hour.to_numpy()

            orig_norm = np.clip(out["Occupancy"].to_numpy(dtype=float) / 9.0, 0.0, 1.0)
            jitter = rng.uniform(-0.08, 0.08, size=len(out))
            signal = np.clip(0.7 * orig_norm + 0.3 * rng.random(len(out)) + jitter, 0.0, 1.0)

            new_occupancy = np.zeros(len(out), dtype=int)

            for idx, row in out.iterrows():
                btype = row["building_type"]
                max_occ = int(row["max_occupancy"])
                hour = int(hours[idx])
                is_weekend = row["DayOfWeek"] in ("Saturday", "Sunday")
                is_holiday = row["Holiday"] == "Yes"
                off_day = bool(is_weekend or is_holiday)
                s = float(signal[idx])

                if btype == "Office":
                    if off_day:
                        frac = 0.0 if s < 0.65 else 0.05 * s
                    else:
                        if 9 <= hour <= 16:
                            frac = 0.65 + 0.30 * s
                        elif hour in (8, 17):
                            frac = 0.40 + 0.30 * s
                        elif hour in (7, 18, 19):
                            frac = 0.08 + 0.17 * s
                        else:
                            frac = 0.0 if s < 0.75 else 0.03

                elif btype == "Home":
                    if off_day:
                        if 18 <= hour <= 23 or 0 <= hour <= 8:
                            frac = 0.80 + 0.20 * s
                        else:
                            frac = 0.50 + 0.40 * s
                    else:
                        if 18 <= hour <= 23:
                            frac = 0.75 + 0.25 * s
                        elif 0 <= hour <= 6:
                            frac = 0.85 + 0.15 * s
                        elif hour in (7, 17):
                            frac = 0.45 + 0.35 * s
                        else:
                            frac = 0.05 + 0.25 * s

                else:  # University
                    if off_day:
                        if 9 <= hour <= 18:
                            frac = 0.04 + 0.11 * s
                        else:
                            frac = 0.0 if s < 0.80 else 0.02
                    else:
                        if 9 <= hour <= 16:
                            frac = 0.60 + 0.35 * s
                        elif hour in (8, 17):
                            frac = 0.35 + 0.30 * s
                        elif 18 <= hour <= 21:
                            frac = 0.18 + 0.25 * s
                        else:
                            frac = 0.0 if s < 0.75 else 0.02

                new_occupancy[idx] = int(np.clip(round(frac * max_occ), 0, max_occ))

            out["Occupancy"] = new_occupancy.astype(int)
            return out
        except Exception as e:
            raise DataProcessingError(f"Failed to simulate realistic occupancy: {e}") from e

    def align_hvac_and_lighting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align HVACUsage and LightingUsage with occupancy and schedule."""
        try:
            out = df.copy()
            rng = np.random.default_rng(self.random_seed + 7)
            hours = pd.to_datetime(out["Timestamp"]).dt.hour.to_numpy()

            hvac_list: List[str] = []
            light_list: List[str] = []

            for idx, row in out.iterrows():
                occ = int(row["Occupancy"])
                max_occ = max(int(row["max_occupancy"]), 1)
                occ_rate = occ / max_occ
                temp = float(row["Temperature"])
                hum = float(row["Humidity"])
                hour = int(hours[idx])
                btype = row["building_type"]

                if occ == 0:
                    light = "On" if rng.random() < 0.06 else "Off"
                    hvac = "On" if (temp > 28.5 or rng.random() < 0.06) else "Off"
                else:
                    if btype == "Home":
                        if 17 <= hour <= 23 or hour in (6, 7):
                            light = "On"
                        elif 0 <= hour <= 5:
                            light = "Off"
                        else:
                            light = "On" if rng.random() < 0.25 else "Off"
                    else:
                        light = "On" if (occ_rate >= 0.10 or 17 <= hour <= 21) else ("On" if rng.random() < 0.3 else "Off")

                    if temp >= 24.0 or hum >= 52.0 or occ_rate >= 0.45:
                        hvac = "On"
                    else:
                        hvac = "On" if rng.random() < 0.30 else "Off"

                hvac_list.append(hvac)
                light_list.append(light)

            out["HVACUsage"] = hvac_list
            out["LightingUsage"] = light_list
            return out
        except Exception as e:
            raise DataProcessingError(f"Failed to align HVAC and lighting states: {e}") from e

    def recalibrate_energy_consumption(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate EnergyConsumption (kWh) based on physical attributes."""
        try:
            out = df.copy()
            rng = np.random.default_rng(self.random_seed + 99)

            sqft = out["SquareFootage"].to_numpy(dtype=float)
            temp = out["Temperature"].to_numpy(dtype=float)
            hum = out["Humidity"].to_numpy(dtype=float)
            occ = out["Occupancy"].to_numpy(dtype=float)
            max_occ = np.maximum(out["max_occupancy"].to_numpy(dtype=float), 1.0)
            occ_rate = occ / max_occ
            hvac_on = (out["HVACUsage"] == "On").to_numpy(dtype=float)
            light_on = (out["LightingUsage"] == "On").to_numpy(dtype=float)

            base_load = 0.026 * sqft
            thermal_load = 1.65 * (temp - 20.0) + 0.12 * (hum - 30.0)
            occupancy_load = 12.0 * occ_rate + 0.18 * occ
            hvac_load = hvac_on * (11.5 + 0.55 * (temp - 20.0))
            lighting_load = light_on * (4.5 + 0.002 * sqft)
            noise = rng.normal(0.0, 1.25, size=len(out))

            energy_kwh = base_load + thermal_load + occupancy_load + hvac_load + lighting_load + noise
            out["EnergyConsumption"] = np.clip(energy_kwh, 25.0, None).round(4)

            ordered_cols = [
                "Timestamp",
                "building_id",
                "building_type",
                "SquareFootage",
                "max_occupancy",
                "Temperature",
                "Humidity",
                "Occupancy",
                "HVACUsage",
                "LightingUsage",
                "RenewableEnergy",
                "DayOfWeek",
                "Holiday",
                "EnergyConsumption",
            ]
            return out[ordered_cols]
        except Exception as e:
            raise DataProcessingError(f"Failed to recalibrate energy consumption: {e}") from e

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all feature engineering and data manipulation steps in sequence."""
        try:
            step1 = self.assign_buildings(df)
            step2 = self.fix_calendar_features(step1)
            step3 = self.simulate_realistic_occupancy(step2)
            step4 = self.align_hvac_and_lighting(step3)
            step5 = self.recalibrate_energy_consumption(step4)
            return step5
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Error executing feature engineering pipeline: {e}") from e