import unittest
from subsystem3.optimisation_scenario import OptimisationScenario
from subsystem3.optimisation_result import OptimisationResult
from subsystem3.optimisation_engine import OptimisationEngine
from subsystem3.results_visualiser import ResultsVisualiser
from subsystem3.exceptions_subsystem3 import (
    ScenarioGenerationError, 
    OptimisationEvaluationError, 
    VisualisationError
)

class TestSubsystem3(unittest.TestCase):

    def setUp(self):
        """Set up common test fixtures before each test method."""
        self.scenario = OptimisationScenario(
            scenario_id=1, 
            parameters={"hvac_offset_c": 2, "lighting_dimming": 0.2}, 
            description="Test Scenario"
        )
        self.engine = OptimisationEngine(building_id="Building_Test")
        self.visualiser = ResultsVisualiser(library="matplotlib")

    # --- OptimisationScenario Tests (1-3) ---
    def test_scenario_initialisation(self):
        self.assertEqual(self.scenario.get_scenario_id(), 1)
        self.assertEqual(self.scenario.get_description(), "Test Scenario")
        self.assertEqual(self.scenario.get_parameters()["hvac_offset_c"], 2)

    def test_scenario_parameter_update(self):
        self.scenario.set_parameters({"hvac_offset_c": 4})
        self.assertEqual(self.scenario.get_parameters()["hvac_offset_c"], 4)

    def test_scenario_to_dict(self):
        d = self.scenario.to_dict()
        self.assertIn("scenario_id", d)
        self.assertEqual(d["description"], "Test Scenario")

    # --- OptimisationResult Tests (4-7) ---
    def test_result_metrics_getters(self):
        result = OptimisationResult(1, "Desc", 100.0, 40.0, 25.0, 85.5)
        self.assertEqual(result.get_energy_kwh(), 100.0)
        self.assertEqual(result.get_carbon_kg(), 40.0)
        self.assertEqual(result.get_cost_eur(), 25.0)

    def test_result_recommendation_flag(self):
        result = OptimisationResult(1, "Desc", 100.0, 40.0, 25.0, 85.5)
        self.assertFalse(result.is_recommended())
        result.set_recommended(True)
        self.assertTrue(result.is_recommended())

    def test_result_summary_dictionary(self):
        result = OptimisationResult(1, "Desc", 100.0, 40.0, 25.0, 85.5)
        summary = result.summary()
        self.assertEqual(summary["score"], 85.5)

    def test_result_setters(self):
        result = OptimisationResult(1, "Desc", 100.0, 40.0, 25.0, 85.5)
        result.set_score(95.0)
        self.assertEqual(result.get_score(), 95.0)

    # --- OptimisationEngine & Error Handling Tests (8-11) ---
    def test_engine_generates_scenarios(self):
        scenarios = self.engine.get_scenarios()
        self.assertGreaterEqual(len(scenarios), 1)

    def test_engine_evaluate_scenario(self):
        res = self.engine.evaluate_scenario(self.scenario)
        self.assertIsInstance(res, OptimisationResult)
        self.assertGreater(res.get_energy_kwh(), 0)

    def test_engine_select_best_scenario(self):
        scenarios = self.engine.get_scenarios()
        results = [self.engine.evaluate_scenario(s) for s in scenarios]
        best = self.engine.select_best_scenario(results)
        self.assertTrue(best.is_recommended())

    def test_engine_select_best_empty_results_raises_error(self):
        """Test that empty results trigger evaluation error handling."""
        with self.assertRaises(OptimisationEvaluationError):
            self.engine.select_best_scenario([])

    # --- ResultsVisualiser & VisualisationError Tests (12-15) ---
    def test_visualiser_initialisation(self):
        self.assertEqual(self.visualiser.get_library(), "matplotlib")

    def test_visualiser_plot_energy_returns_figure(self):
        scenarios = self.engine.get_scenarios()
        results = [self.engine.evaluate_scenario(s) for s in scenarios]
        fig = self.visualiser.plot_energy_comparison(results)
        self.assertIsNotNone(fig)

    def test_visualiser_plot_cost_returns_figure(self):
        scenarios = self.engine.get_scenarios()
        results = [self.engine.evaluate_scenario(s) for s in scenarios]
        fig = self.visualiser.plot_cost_comparison(results)
        self.assertIsNotNone(fig)

    def test_visualiser_empty_results_raises_error(self):
        """Test that plotting empty results triggers VisualisationError."""
        with self.assertRaises(VisualisationError):
            self.visualiser.plot_energy_comparison([])


if __name__ == '__main__':
    unittest.main()