"""Domain models (`Building`, `EnergyLog`) for Subsystem 1."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict


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
