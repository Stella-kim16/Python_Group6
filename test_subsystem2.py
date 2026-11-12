import json
from pathlib import Path

import pandas as pd

from subsystem2.Data_processor import load_clean_data
from subsystem2.AI_model import AImodel
from subsystem2.Demandforecast import DemandForecaster
from subsystem2.AI_savingadviser import SavingsAdvisor
from subsystem2.AI_sustainability import SustainabilityScorer
from subsystem2.Prediction_result import PredictionResult
from subsystem2.Model_evaluation import ModelEvaluator
from subsystem2.Result_exporter import ResultExporter

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "enriched_energy_data.csv"


def run_subsystem2_only() -> None:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"{PROCESSED_CSV} not found. Run Subsystem 1 (main.py) once first.")

    print("\n[1/6] Loading processed data...")
    clean_df = load_clean_data(PROCESSED_CSV)
    raw_df = pd.read_csv(PROCESSED_CSV)

    print("\n[2/6] Training Random Forest...")
    model = AImodel(n_estimators=200, random_state=42).train(clean_df)

    print("\n[3/6] 24-hour forecast...")
    forecasts = DemandForecaster(model, raw_df).forecast_all()

    print("\n[4/6] Savings recommendations...")
    tips = SavingsAdvisor(model, raw_df).recommend_all()

    print("\n[5/6] Sustainability scores...")
    scores = SustainabilityScorer(model, raw_df).score_all()

    print("\n[6/6] Model evaluation and JSON export...")
    test_time = pd.to_datetime(raw_df.loc[model.X_test.index, "Timestamp"])
    rf_result = PredictionResult.from_series(
        "Random Forest", model.predict(model.X_test), model.y_test, timestamps=test_time)
    metrics = ModelEvaluator().evaluate(rf_result)
    print(f"  MAE: {metrics['MAE']} kWh | MSE: {metrics['MSE']}")

    exporter = ResultExporter(model)
    exporter.export(forecasts, scores, tips, metrics)

    with open(exporter.path, encoding="utf-8") as f:
        saved = json.load(f)
    print("\n[Check] JSON file")
    print(f"  Path      : {exporter.path}")
    print(f"  Buildings : {list(saved['buildings'])}")
    print(f"  Accuracy  : {saved['model']['accuracy']}")
    print(f"  Drivers   : {saved['model']['key_drivers']}")
    first = next(iter(saved["buildings"]))
    print(f"  {first} keys: {list(saved['buildings'][first])}")
    print("\nSubsystem 2 test finished successfully.")


if __name__ == "__main__":
    run_subsystem2_only()