from dataclasses import dataclass, field
import numpy as np
import pandas as pd
 
 
WEIGHTS = {"waste": 0.4, "renewable": 0.3, "peak": 0.2, "efficiency": 0.1}
WASTE_LIMIT = 0.10
RENEWABLE_TARGET = 0.50
PEAK_LIMIT = 0.50
 
BUILDING_PREFIX = "Bld_"
OCCUPANCY_COL = "Occupancy"
SWITCH_OFF_COLS = ["HVACUsage", "LightingUsage"]
RAW_COLS = ["building_id", "building_type", "EnergyConsumption", "RenewableEnergy",
            "cost_per_kwh", "SquareFootage", "carbon_footprint_kg", "Timestamp"]
 
 
@dataclass
class ScoreResult:
    """Sustainability score of one building with the numbers behind it."""
    building_id: str
    building_type: str
    total_score: float
    efficiency_score: float
    sub_scores: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
 
 
def _clip(value: float) -> float:
    """Keep a score between 0 and 100."""
    return float(np.clip(value, 0, 100))
 
 
class SustainabilityScorer:
    """Score how sustainably each building uses energy."""
 
    def __init__(self, model, raw_df: pd.DataFrame, weights: dict | None = None):
        if not model.is_trained:
            raise RuntimeError("Train the AIforecast model before creating SustainabilityScorer.")
        missing = [c for c in RAW_COLS if c not in raw_df.columns]
        if missing:
            raise KeyError(f"Columns missing in raw CSV: {missing}")
 
        self.model = model
        self.raw = raw_df
        self.weights = weights or WEIGHTS
        if abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("Weights must add up to 1.")
 
        self.history_X = model.history_X()
        self.building_cols = [c for c in self.history_X.columns if c.startswith(BUILDING_PREFIX)]
        if not self.building_cols:
            raise RuntimeError("No 'Bld_' columns found — one-hot encode 'building_id' first.")
 
        self.offpeak_price = self.raw["cost_per_kwh"].min()
 
        intensity = (self.raw["EnergyConsumption"] / self.raw["SquareFootage"])
        per_building = intensity.groupby(self.raw["building_id"]).mean()
        types = self.raw.groupby("building_id")["building_type"].first()
        self.intensity = per_building
        self.best_in_type = per_building.groupby(types).min()
        self.types = types
 
    def buildings(self) -> list[str]:
        return sorted(c[len(BUILDING_PREFIX):] for c in self.building_cols)
 
    def _waste(self, building_id: str) -> tuple[float, dict]:
        """Model-estimated share of energy that would be saved by switching devices off in empty rooms."""
        hist = self.history_X[self.history_X[f"{BUILDING_PREFIX}{building_id}"] == 1]
        base_kwh = self.model.predict(hist)
 
        no_waste = hist.copy()
        empty = no_waste[OCCUPANCY_COL] == 0
        cols = [c for c in SWITCH_OFF_COLS if c in no_waste.columns]
        wasted_hours = int((empty & (no_waste[cols] == 1).any(axis=1)).sum())
        no_waste.loc[empty, cols] = 0
        rule_kwh = self.model.predict(no_waste)
 
        waste_kwh = float((base_kwh - rule_kwh).clip(lower=0).sum())
        ratio = waste_kwh / float(base_kwh.sum())
        score = _clip(100 * (1 - ratio / WASTE_LIMIT))
        return score, {"wasted_kWh": round(waste_kwh, 2),
                       "waste_ratio": round(ratio, 4),
                       "wasted_hours": wasted_hours}
 
    def _renewable(self, rows: pd.DataFrame) -> tuple[float, dict]:
        share = rows["RenewableEnergy"].sum() / rows["EnergyConsumption"].sum()
        return _clip(100 * share / RENEWABLE_TARGET), {"renewable_share": round(float(share), 4)}
 
    def _peak(self, rows: pd.DataFrame) -> tuple[float, dict]:
        is_peak = rows["cost_per_kwh"] > self.offpeak_price
        share = rows.loc[is_peak, "EnergyConsumption"].sum() / rows["EnergyConsumption"].sum()
        return _clip(100 * (1 - share / PEAK_LIMIT)), {"peak_share": round(float(share), 4)}
 
    def _efficiency(self, building_id: str) -> tuple[float, dict]:
        own = self.intensity[building_id]
        best = self.best_in_type[self.types[building_id]]
        return _clip(100 * best / own), {"kWh_per_1000_sqft": round(float(own * 1000), 2)}
 
    def score_building(self, building_id: str) -> ScoreResult:
        if f"{BUILDING_PREFIX}{building_id}" not in self.building_cols:
            raise ValueError(f"Unknown building '{building_id}'. Available: {self.buildings()}")
 
        rows = self.raw[self.raw["building_id"] == building_id]
 
        waste, d1 = self._waste(building_id)
        renewable, d2 = self._renewable(rows)
        peak, d3 = self._peak(rows)
        efficiency, d4 = self._efficiency(building_id)
        sub_scores = {"waste": round(waste, 1), "renewable": round(renewable, 1),
                      "peak": round(peak, 1), "efficiency": round(efficiency, 1)}
 
        total = sum(self.weights[k] * v for k, v in sub_scores.items())
 
        n_days = pd.to_datetime(rows["Timestamp"]).dt.date.nunique()
        details = {**d1, **d2, **d3, **d4,
                   "carbon_kg_per_day": round(float(rows["carbon_footprint_kg"].sum() / n_days), 2)}
 
        return ScoreResult(
            building_id=building_id,
            building_type=self.types[building_id],
            total_score=round(total, 1),
            efficiency_score=round(efficiency, 1),
            sub_scores=sub_scores,
            details=details,
        )
 
    def score_all(self) -> dict[str, ScoreResult]:
        """{building_id: ScoreResult} for every building."""
        results = {}
        for building_id in self.buildings():
            results[building_id] = self.score_building(building_id)
            r = results[building_id]
            print(f"  {building_id}: {r.total_score}/100  {r.sub_scores}")
        return results
 
