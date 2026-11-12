import json
from dataclasses import asdict
from pathlib import Path
 
from .AI_sustainability import ScoreResult
from .AI_savingadviser import Recommendation
from .Demandforecast import ForecastResult
 
DEFAULT_PATH = Path(__file__).resolve().parent.parent /"data" / "processed" / "output" /"subsystem2_results.json"
HIDDEN_PREFIXES = ("Bld_", "Day_")
 
 
class ResultExporter:
 
    def __init__(self, model, path: str | Path = DEFAULT_PATH, top_drivers: int = 5):
        if not model.is_trained:
            raise RuntimeError("Train the AImodel before creating ResultExporter.")
        self.model = model
        self.path = Path(path)
        self.top_drivers = top_drivers
        self.results: dict = {}
 
    def key_drivers(self) -> dict:
        importance = self.model.feature_importance()
        importance = importance[[not c.startswith(HIDDEN_PREFIXES) for c in importance.index]]
        return {name: round(float(value), 4)
                for name, value in importance.head(self.top_drivers).items()}
 
    @staticmethod
    def forecast_to_dict(forecast: ForecastResult) -> dict:
        hourly = forecast.hourly
        peak_time = hourly["predicted_kWh"].idxmax()
        return {
            "total_kwh": forecast.total_kwh,
            "total_cost_eur": forecast.total_cost_eur,
            "peak_hour": peak_time.strftime("%H:%M"),
            "peak_kwh": round(float(hourly["predicted_kWh"].max()), 2),
            "hourly": [
                {"time": ts.strftime("%Y-%m-%d %H:%M"),
                 "kwh": round(float(row["predicted_kWh"]), 3),
                 "cost_eur": round(float(row["predicted_cost_eur"]), 3)}
                for ts, row in hourly.iterrows()
            ],
        }
 
    @staticmethod
    def score_to_dict(score: ScoreResult) -> dict:
        return {
            "total_score": score.total_score,
            "efficiency_score": score.efficiency_score,
            "sub_scores": score.sub_scores,
            "details": score.details,
        }
 
    def build(self, forecasts: dict[str, ForecastResult], scores: dict[str, ScoreResult],
              tips: dict[str, list[Recommendation]], metrics: dict | None = None) -> dict:
        self.results = {
            "model": {
                "name": "Random Forest",
                "accuracy": metrics or {},
                "key_drivers": self.key_drivers(),
            },
            "buildings": {},
        }
        for building_id, score in scores.items():
            self.results["buildings"][building_id] = {
                "building_type": score.building_type,
                "forecast": self.forecast_to_dict(forecasts[building_id]),
                "score": self.score_to_dict(score),
                "recommendations": [asdict(t) for t in tips.get(building_id, [])],
            }
        return self.results
 
    def save(self) -> Path:
        if not self.results:
            raise RuntimeError("Call build() before save().")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"  Results for {len(self.results['buildings'])} buildings saved to {self.path}")
        return self.path
 
    def export(self, forecasts: dict[str, ForecastResult], scores: dict[str, ScoreResult],
               tips: dict[str, list[Recommendation]], metrics: dict | None = None) -> dict:
        self.build(forecasts, scores, tips, metrics)
        self.save()
        return self.results
 
