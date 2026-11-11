"""Unified Main Orchestration Script for Energy Management System.

Integrates:
- Subsystem 1: Event-Driven Stream Processing (ABCs, Polymorphism, Operator Overloading, Iterators/Generators),
  Ingestion, Feature Engineering, Metrics Calculation, & SQLite Storage
- Subsystem 2: Machine Learning (Random Forest), Demand Forecasting, Savings Advisor, & Sustainability Scoring
"""

from pathlib import Path
import pandas as pd

# Import Subsystem 1 modules
from subsystem1 import (
    DEFAULT_BUILDING_CATALOG,
    DatabaseManager,
    DataIngestionEngine,
    FeatureEngineer,
    LoadBalanceAndTariffTransformer,
    MetricsCalculator,
    SmartMeterStream,
    SubThresholdMeterFilter,
)

# Import Subsystem 2 modules
from subsystem2.Data_processor import load_clean_data
from subsystem2.AI_model import AImodel
from subsystem2.Demandforecast import DemandForecaster
from subsystem2.AI_savingadviser import SavingsAdvisor
from subsystem2.AI_sustainablity import SustainabilityScorer


def run_complete_energy_system(
    raw_csv_path: str | Path = "data/raw/Energy_consumption.csv",
    db_path: str | Path = "data/processed/smart_energy.db",
    processed_csv_path: str | Path = "data/processed/enriched_energy_data.csv",
    processed_json_path: str | Path = "data/processed/enriched_energy_data.json",
):
    

    # ==========================================
    # SUBSYSTEM 1: Event-Driven Stream Processing & Batch Pipeline
    # ==========================================
    print("\n[Step 1/5] Running Subsystem 1: Event-Driven Stream Processing & Data Management...")

    meter_filter = SubThresholdMeterFilter(min_power_kwh=65.0)
    engineer = FeatureEngineer(buildings=DEFAULT_BUILDING_CATALOG, random_seed=42)
    transformer = LoadBalanceAndTariffTransformer()

    stream_pipeline = meter_filter + engineer + transformer
    iot_stream = SmartMeterStream(raw_csv_path)
    streamed_events = list(stream_pipeline.process_stream(iot_stream))

    ingestion = DataIngestionEngine(raw_csv_path)
    calculator = MetricsCalculator()
    batch_pipeline = ingestion | engineer | calculator
    enriched_df = batch_pipeline.process(raw_csv_path)

    db = DatabaseManager(connection_string=db_path)
    db.save_buildings(DEFAULT_BUILDING_CATALOG)
    db.save_energy_logs_df(enriched_df)
    db.export_to_csv(processed_csv_path)
    db.export_to_json(processed_json_path)
    print(f"  -> Stream Pipeline: {stream_pipeline} (yielded {len(streamed_events)} events)")
    print(f"  -> Batch Pipeline:  {batch_pipeline}")
    print(f"  -> Storage Status:  {db}")
    print(f"  -> Subsystem 1 complete. Enriched data saved to: {processed_csv_path}")

    # ==========================================
    # SUBSYSTEM 2: AI Modeling & Forecasting
    # ==========================================
    print("\n[Step 2/5] Training Subsystem 2: Random Forest AI Model...")

    # Load fully processed clean data for modeling
    clean_df = load_clean_data(processed_csv_path)
    raw_df = pd.read_csv(processed_csv_path)

    # Train Random Forest Regressor
    model = AImodel(n_estimators=200, random_state=42)
    model.train(clean_df)
    model.save("trained_energy_model.pkl")
    print("  -> AI Model successfully trained and saved as 'trained_energy_model.pkl'.")

    # ==========================================
    # SUBSYSTEM 2: Demand Forecasting
    # ==========================================
    print("\n[Step 3/5] Running Next 24-Hour Demand Forecast...")
    forecaster = DemandForecaster(model, raw_df)
    forecast_results = forecaster.forecast_all()

    print("\n--- 24-Hour Forecast Summary ---")
    for bld_id, res in forecast_results.items():
        print(f"  Building {bld_id:12s} | 24h Total: {res.total_kwh:8.2f} kWh | Estimated Cost: EUR {res.total_cost_eur:6.2f}")

    # ==========================================
    # SUBSYSTEM 2: Savings Advisor
    # ==========================================
    print("\n[Step 4/5] Evaluating Energy Savings Recommendations...")
    advisor = SavingsAdvisor(model, raw_df)
    recommendations = advisor.recommend_all()

    print("\n--- Top Saving Opportunities ---")
    for bld_id, recs in recommendations.items():
        if recs:
            print(f"  Building: {bld_id}")
            for rec in recs:
                print(f"    * [{rec.title}] Wasted Hours: {rec.wasted_hours}h | Save: ~EUR {rec.saved_eur_per_month:.2f}/month")

    # ==========================================
    # SUBSYSTEM 2: Sustainability Scoring
    # ==========================================
    print("\n[Step 5/5] Calculating Building Sustainability Scores...")
    scorer = SustainabilityScorer(model, raw_df)
    scores = scorer.score_all()

    print("\n--- Sustainability Score Summary (0-100) ---")
    for bld_id, score_res in scores.items():
        print(f"  Building {bld_id:12s} | Total Score: {score_res.total_score:5.1f}/100 | Efficiency: {score_res.efficiency_score:5.1f}/100 | Sub-scores: {score_res.sub_scores}")

    print("\n" + "=" * 80)
    print("INTEGRATED PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_complete_energy_system()
