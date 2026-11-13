import json
from pathlib import Path
from typing import List
from subsystem3.optimisation_scenario import OptimisationScenario
from subsystem3.optimisation_result import OptimisationResult

SUBSYSTEM2_JSON = Path(__file__).resolve().parent.parent / "data" / "output" / "subsystem2_results.json"


class OptimisationEngine:
    """Coordinates scenario generation, evaluation, and optimal selection."""
    
    def __init__(self, building_id: str):
        self._building_id = building_id
        self._scenarios: List[OptimisationScenario] = []
        self._baseline_kwh, self._price_per_kwh = self.load_baseline()
        self.initialize_engine()

    def load_baseline(self) -> tuple[float, float]:
        """Reads the 24-hour forecast of this building from the Subsystem 2 JSON file."""
        if SUBSYSTEM2_JSON.exists():
            with open(SUBSYSTEM2_JSON, encoding="utf-8") as f:
                buildings = json.load(f).get("buildings", {})
            if self._building_id in buildings:
                forecast = buildings[self._building_id]["forecast"]
                return forecast["total_kwh"], forecast["total_cost_eur"] / forecast["total_kwh"]
        return 1500.0, 0.25

    def initialize_engine(self) -> None:
        """Setup default baseline operational scenarios upon instantiation."""
        self._scenarios = [
            OptimisationScenario(1, {"hvac_offset_c": 0, "lighting_dimming": 0.0}, "Baseline Operation"),
            OptimisationScenario(2, {"hvac_offset_c": 2, "lighting_dimming": 0.2}, "Moderate Energy Saving"),
            OptimisationScenario(3, {"hvac_offset_c": 4, "lighting_dimming": 0.5}, "Aggressive Eco Mode")
        ]

    # Getters (Accessor Methods)
    def get_building_id(self) -> str:
        return self._building_id

    def get_scenarios(self) -> List[OptimisationScenario]:
        return self._scenarios

    # Setters (Mutator Methods)
    def set_building_id(self, building_id: str) -> None:
        self._building_id = building_id

    def set_scenarios(self, scenarios: List[OptimisationScenario]) -> None:
        self._scenarios = scenarios

    def evaluate_scenarios(self) -> List[OptimisationResult]:
        """Evaluates all managed scenarios and returns calculated results."""
        results = []
        for sc in self._scenarios:
            params = sc.get_parameters()
            reduction_factor = 1.0 - (params.get("hvac_offset_c", 0) * 0.05 + params.get("lighting_dimming", 0.0) * 0.1)
            
            energy = round(self._baseline_kwh * reduction_factor, 2)
            carbon = round(energy * 0.4, 2)
            cost = round(energy * self._price_per_kwh, 2)
            score = 60.0 + (1.0 - reduction_factor) * 40.0
            
            res = OptimisationResult(sc.get_scenario_id(), sc.get_description(), energy, carbon, cost, score)
            results.append(res)
        return results

    def select_best_scenario(self, results: List[OptimisationResult]) -> OptimisationResult:
        """Selects the scenario with the highest sustainability score."""
        best = max(results, key=lambda r: r.get_score())
        best.set_recommended(True)
        return best