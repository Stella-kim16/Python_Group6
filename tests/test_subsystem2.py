import unittest
import pandas as pd
import numpy as np

from subsystem2.exceptions_subsystem2 import ModelTrainingError, ForecastError, ModelEvaluationError
from subsystem2.AI_model import AImodel
from subsystem2.Demandforecast import DemandForecaster
from subsystem2.Data_processor import Dataprocessor, Dataspliting
from subsystem2.Model_evaluation import ModelEvaluator
from subsystem2.Prediction_result import PredictionResult


class TestSubsystem2(unittest.TestCase):

    def setUp(self):
        # Create dummy clean DataFrame with required features for Subsystem 2
        np.random.seed(42)
        n = 50
        self.clean_df = pd.DataFrame({
            "Timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "HVACUsage": np.random.choice([0, 1], size=n),
            "LightingUsage": np.random.choice([0, 1], size=n),
            "Occupancy": np.random.randint(1, 10, size=n),
            "Temperature": np.random.uniform(18.0, 28.0, size=n),
            "Humidity": np.random.uniform(30.0, 70.0, size=n),
            "SquareFootage": np.random.choice([1200.0, 1500.0], size=n),
            "max_occupancy": np.random.choice([5, 10], size=n),
            "Hour": np.random.randint(0, 24, size=n),
            "Month": [1] * n,
            "Holiday": [0] * n,
            "RenewableEnergy": np.random.uniform(0, 50, size=n),
            "Day_Monday": np.random.choice([0, 1], size=n),
            "Bld_BLD_HOM_01": np.random.choice([0, 1], size=n),
            "cost_per_kwh": [0.25] * n,
            "EnergyConsumption": np.random.uniform(20.0, 60.0, size=n)
        })

    def test_1_ai_model_untrained_prediction(self):
        """Test that calling predict() before train() raises ModelTrainingError."""
        model = AImodel()
        with self.assertRaises(ModelTrainingError):
            model.predict(self.clean_df.drop(columns=["EnergyConsumption"]))

    def test_2_ai_model_training(self):
        """Test successful training and feature importance extraction."""
        model = AImodel(n_estimators=10)
        model.train(self.clean_df)
        self.assertTrue(model.is_trained)
        importances = model.feature_importance()
        self.assertGreater(len(importances), 0)

    def test_3_data_splitting_non_numeric(self):
        """Test that Dataspliting raises error if non-numeric columns remain."""
        df_dirty = self.clean_df.copy()
        df_dirty["UnencodedString"] = "Text"
        splitter = Dataspliting(df_dirty)
        splitter.features.append("UnencodedString")
        # Since Data_processor wraps exceptions, check for either RuntimeError or ModelTrainingError
        with self.assertRaises((RuntimeError, ModelTrainingError)):
            splitter.split()

    def test_4_demand_forecaster_unknown_building(self):
        """Test that forecasting an unknown building raises ForecastError."""
        model = AImodel(n_estimators=10)
        model.train(self.clean_df)
        
        # Pass a dataframe containing the required 'building_id' metadata column
        raw_df_with_meta = self.clean_df.copy()
        raw_df_with_meta["building_id"] = "BLD_HOM_01"
        
        forecaster = DemandForecaster(model, raw_df_with_meta)
        with self.assertRaises(ForecastError):
            forecaster.forecast_building(building_id="UNKNOWN_BLD")

    def test_5_model_evaluator_missing_actuals(self):
        """Test that ModelEvaluator raises ModelEvaluationError when actual values are missing."""
        evaluator = ModelEvaluator()
        # Initialize PredictionResult with the required arguments, providing empty/None actuals
        pred_result = PredictionResult(
            modelName="RandomForest",
            timestamps=["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
            predictions=[30.0, 40.0],
            actualValues=[None, None]
        )
        with self.assertRaises(ModelEvaluationError):
            evaluator.evaluate(pred_result)