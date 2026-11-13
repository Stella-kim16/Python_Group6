import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import pandas as pd

from subsystem1.models import Building, EnergyLog
from subsystem1.exceptions_subsystem1 import (
    EnergyPipelineError,
    DataValidationError,
    StreamProcessingError,
    DatabaseOperationError,
)
from subsystem1.pipeline import StreamProcessor, StreamPipeline
from subsystem1.ingestion import (
    SmartMeterStream,
    DataIngestionEngine,
)
from subsystem1.feature_engineering import FeatureEngineer, DEFAULT_BUILDING_CATALOG
from subsystem1.metrics import MetricsCalculator
from subsystem1.database import DatabaseManager


class TestSubsystem1(unittest.TestCase):

    def setUp(self):
        """Set up reusable test fixtures before each test method."""
        self.sample_raw_events = [
            {
                "Timestamp": "2026-01-05 10:00:00",
                "Temperature": 23.5,
                "Humidity": 45.0,
                "SquareFootage": 1450.0,
                "Occupancy": 5,
                "HVACUsage": "On",
                "LightingUsage": "On",
                "RenewableEnergy": 20.0,
                "DayOfWeek": "Monday",
                "Holiday": "No",
                "EnergyConsumption": 55.0,
            },
            {
                "Timestamp": "2026-01-05 11:00:00",
                "Temperature": 24.0,
                "Humidity": 50.0,
                "SquareFootage": 1120.0,
                "Occupancy": 0,
                "HVACUsage": "On",
                "LightingUsage": "On",
                "RenewableEnergy": 10.0,
                "DayOfWeek": "Sunday",
                "Holiday": "No",
                "EnergyConsumption": 62.0,
            },
            {
                "Timestamp": "2026-01-05 12:00:00",
                "Temperature": 21.0,
                "Humidity": 40.0,
                "SquareFootage": 1700.0,
                "Occupancy": 2,
                "HVACUsage": "Off",
                "LightingUsage": "Off",
                "RenewableEnergy": 30.0,
                "DayOfWeek": "Monday",
                "Holiday": "No",
                "EnergyConsumption": 15.0,  # Below default 25.0 kWh threshold
            },
        ]
        self.raw_df = pd.DataFrame(self.sample_raw_events)

    # --- 1. Domain Models & Validation Tests ---
    def test_1_building_model(self):
        """Test Building domain model metadata dictionary and string representation."""
        bld = Building("BLD_TEST_01", "Office", 1500.0, max_occupancy=40, baseline_kwh=50.0)
        info = bld.get_building_info()
        self.assertEqual(info["building_id"], "BLD_TEST_01")
        self.assertEqual(info["building_type"], "Office")
        self.assertEqual(info["max_occupancy"], 40)
        self.assertIn("BLD_TEST_01", str(bld))

    def test_2_energy_log_validation_and_serialization(self):
        """Test EnergyLog data validation rules and dictionary serialization."""
        valid_log = EnergyLog(
            timestamp=datetime(2026, 1, 5, 10, 0, 0),
            building_id="BLD_OFF_01",
            building_type="Office",
            area_sqft=1420.0,
            temperature=22.5,
            humidity=45.0,
            occupancy=10,
            hvac_usage="On",
            lighting_usage="Off",
            renewable_energy_pct=25.0,
            day_of_week="Monday",
            holiday="No",
            energy_consumption_kwh=48.0,
        )
        self.assertTrue(valid_log.validate_data_types())
        serialized = valid_log.to_dict()
        self.assertEqual(serialized["timestamp"], "2026-01-05 10:00:00")
        self.assertEqual(serialized["building_id"], "BLD_OFF_01")

        # Invalid humidity (> 100) should fail validation
        invalid_log = EnergyLog(
            timestamp=datetime(2026, 1, 5, 10, 0, 0),
            building_id="BLD_OFF_01",
            building_type="Office",
            area_sqft=1420.0,
            temperature=22.5,
            humidity=120.0,
            occupancy=10,
            hvac_usage="On",
            lighting_usage="Off",
            renewable_energy_pct=25.0,
            day_of_week="Monday",
            holiday="No",
            energy_consumption_kwh=48.0,
        )
        self.assertFalse(invalid_log.validate_data_types())

    # --- 2. SmartMeterStream & DataIngestionEngine Tests ---
    def test_3_smart_meter_stream_iterator(self):
        """Test SmartMeterStream iterator protocol (__iter__, __next__, __len__)."""
        stream = SmartMeterStream(self.sample_raw_events)
        self.assertEqual(len(stream), 3)

        emitted = list(stream)
        self.assertEqual(len(emitted), 3)
        self.assertEqual(stream.emitted_count, 3)

        # Iterating again resets the cursor and the emitted counter
        self.assertEqual(len(list(stream)), 3)
        self.assertEqual(stream.emitted_count, 3)

        # Missing CSV source raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            SmartMeterStream("does_not_exist.csv")

    def test_4_ingestion_engine_threshold_filtering(self):
        """Test DataIngestionEngine drops sub-threshold events and keeps valid ones."""
        engine = DataIngestionEngine(min_power_threshold=25.0)

        # Event with 55.0 kWh should pass
        passed = engine.process(self.sample_raw_events[0])
        self.assertIsNotNone(passed)

        # Event with 15.0 kWh (< 25.0) should be dropped (returns None)
        dropped = engine.process(self.sample_raw_events[2])
        self.assertIsNone(dropped)

        self.assertEqual(engine.processed_count, 1)
        self.assertEqual(engine.dropped_count, 1)

        # DataFrame batch filtering
        filtered_df = engine.process(self.raw_df)
        self.assertEqual(len(filtered_df), 2)

    def test_5_ingestion_engine_validation_and_deduplication(self):
        """Test DataIngestionEngine cleans data, deduplicates timestamps, and raises DataValidationError."""
        engine = DataIngestionEngine(min_power_threshold=20.0)

        # Missing required columns in a CSV file should raise DataValidationError
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_csv = Path(tmp_dir) / "bad.csv"
            pd.DataFrame([{"Timestamp": "2026-01-05 10:00:00"}]).to_csv(bad_csv, index=False)
            with self.assertRaises(DataValidationError):
                engine.import_csv_batch(bad_csv)

            # Valid CSV is loaded and cleaned
            good_csv = Path(tmp_dir) / "good.csv"
            self.raw_df.to_csv(good_csv, index=False)
            loaded = engine.import_csv_batch(good_csv)
            self.assertEqual(len(loaded), 2)  # 15.0 kWh row removed

        # No file path configured and none provided -> FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            engine.import_csv_batch()

        # Duplicate timestamps should be deduplicated and sub-threshold row (15.0 kWh) removed
        dup_df = pd.concat([self.raw_df, self.raw_df.iloc[[0]]], ignore_index=True)
        processed_df = engine.process(dup_df)
        self.assertEqual(len(processed_df), 2)

    # --- 3. FeatureEngineer Tests ---
    def test_6_feature_engineer_pipeline_and_single_event(self):
        """Test FeatureEngineer assigns buildings, fixes Sunday holiday flag, and processes events."""
        fe = FeatureEngineer(random_seed=42)

        # Single event on Sunday should automatically get Holiday = 'Yes' and building metadata
        sunday_event = fe.process(self.sample_raw_events[1])
        self.assertEqual(sunday_event["Holiday"], "Yes")
        self.assertIn("building_id", sunday_event)
        self.assertIn("building_type", sunday_event)

        # Batch DataFrame pipeline
        clean_df = self.raw_df.iloc[:2].copy()
        enriched_df = fe.run_pipeline(clean_df)
        self.assertEqual(len(enriched_df), 2)
        self.assertIn("building_id", enriched_df.columns)
        self.assertTrue((enriched_df["EnergyConsumption"] >= 25.0).all())

    # --- 4. MetricsCalculator Tests ---
    def test_7_metrics_calculator_tariffs_and_emissions(self):
        """Test peak/off-peak tariff rates and carbon emission calculations."""
        calc = MetricsCalculator(emission_factor=0.42, peak_tariff=0.30, off_peak_tariff=0.15)

        # Monday at 10:00 (non-holiday) -> Peak tariff (0.30)
        self.assertTrue(calc.is_peak_tariff("2026-01-05 10:00:00", holiday="No"))
        self.assertEqual(calc.get_tariff_rate("2026-01-05 10:00:00", holiday="No"), 0.30)

        # Monday at 10:00 on a Holiday -> Off-peak tariff (0.15)
        self.assertFalse(calc.is_peak_tariff("2026-01-05 10:00:00", holiday="Yes"))
        self.assertEqual(calc.get_tariff_rate("2026-01-05 10:00:00", holiday="Yes"), 0.15)

        # Carbon emission for 100 kWh with 50% renewable energy: 100 * 0.5 * 0.42 = 21.0 kg
        self.assertAlmostEqual(calc.calculate_carbon_emission(100.0, 50.0), 21.0, places=4)

    def test_8_metrics_calculator_anomaly_detection_and_enrichment(self):
        """Test MetricsCalculator flags unoccupied active HVAC+Lighting as anomaly."""
        calculator = MetricsCalculator()
        fe = FeatureEngineer(random_seed=42)

        # Unoccupied event with both HVAC and Lighting 'On' should be flagged as an anomaly
        unoccupied_waste_event = fe.process(self.sample_raw_events[1])
        unoccupied_waste_event["Occupancy"] = 0
        unoccupied_waste_event["HVACUsage"] = "On"
        unoccupied_waste_event["LightingUsage"] = "On"

        transformed = calculator.process(unoccupied_waste_event)
        self.assertTrue(transformed["is_anomaly"])
        self.assertIn("energy_cost_eur", transformed)
        self.assertIn("carbon_footprint_kg", transformed)

        # Batch DataFrame enrichment adds the metric columns
        enriched_df = calculator.process(fe.run_pipeline(self.raw_df.iloc[:2].copy()))
        for col in ("cost_per_kwh", "energy_cost_eur", "carbon_footprint_kg", "is_anomaly"):
            self.assertIn(col, enriched_df.columns)

    # --- 5. StreamPipeline Composition & Delivery Guarantees Tests ---
    def test_9_stream_pipeline_composition_deduplication_and_watermarks(self):
        """Test StreamPipeline composition (+ and |), duplicate suppression, and out-of-order detection."""
        pipeline = (
            DataIngestionEngine(min_power_threshold=25.0)
            | FeatureEngineer(random_seed=42)
            | MetricsCalculator()
        )
        self.assertIsInstance(pipeline, StreamPipeline)
        self.assertEqual(len(pipeline.processors), 3)

        # Stream with a duplicate event and an out-of-order timestamp for the same building
        e1 = dict(self.sample_raw_events[0], Timestamp="2026-01-05 12:00:00")
        e1_dup = dict(e1)  # Duplicate of e1
        e2_out_of_order = dict(self.sample_raw_events[0], Timestamp="2026-01-05 09:00:00")
        e3_sub_threshold = dict(self.sample_raw_events[2])  # 15.0 kWh -> filtered out

        output_events = list(pipeline.process_stream([e1, e1_dup, e2_out_of_order, e3_sub_threshold]))

        self.assertEqual(len(output_events), 2)
        self.assertEqual(pipeline.duplicate_count, 1)
        self.assertEqual(pipeline.out_of_order_count, 1)

    def test_10_stream_pipeline_errors_and_exception_hierarchy(self):
        """Test StreamPipeline error handling (StreamProcessingError) and exception inheritance."""
        self.assertTrue(issubclass(DataValidationError, (EnergyPipelineError, ValueError)))
        self.assertTrue(issubclass(StreamProcessingError, (EnergyPipelineError, RuntimeError)))
        self.assertTrue(issubclass(DatabaseOperationError, (EnergyPipelineError, RuntimeError)))

        pipeline = StreamPipeline([DataIngestionEngine()])
        self.assertIsInstance(pipeline, StreamProcessor)

        # Passing an unsupported type (int) triggers a stage error wrapped in StreamProcessingError
        with self.assertRaises(StreamProcessingError):
            pipeline.process(12345)

    # --- 6. DatabaseManager In-Memory Persistence Tests ---
    def _build_enriched_df(self) -> pd.DataFrame:
        pipeline = (
            DataIngestionEngine(min_power_threshold=25.0)
            + FeatureEngineer(random_seed=42)
            + MetricsCalculator()
        )
        return pipeline.process(self.raw_df)

    def test_11_database_manager_in_memory_crud(self):
        """Test DatabaseManager in-memory SQLite persistence, idempotency, and time-series queries."""
        db = DatabaseManager(connection_string=":memory:")
        db.save_buildings(DEFAULT_BUILDING_CATALOG)

        enriched_df = self._build_enriched_df()
        self.assertEqual(len(enriched_df), 2)

        # First insert
        db.save_energy_logs_df(enriched_df)
        self.assertEqual(db.get_record_count(), 2)

        # Re-saving the same DataFrame should be idempotent (table is rebuilt)
        db.save_energy_logs_df(enriched_df)
        self.assertEqual(db.get_record_count(), 2)

        # Query time-series data
        df = db.fetch_time_series_data()
        self.assertEqual(len(df), 2)
        self.assertIn("energy_cost_eur", df.columns)

        # Filter by building_id
        bld_id = enriched_df.iloc[0]["building_id"]
        filtered = db.fetch_time_series_data(building_id=bld_id)
        self.assertTrue((filtered["building_id"] == bld_id).all())

    def test_12_database_manager_operation_errors(self):
        """Test DatabaseManager raises DatabaseOperationError on foreign key violations."""
        db = DatabaseManager(connection_string=":memory:")
        db.save_buildings(DEFAULT_BUILDING_CATALOG)

        # Saving logs with an unknown building_id violates SQLite foreign key -> DatabaseOperationError
        bad_df = self._build_enriched_df()
        bad_df["building_id"] = "NON_EXISTENT_BLD"
        with self.assertRaises(DatabaseOperationError):
            db.save_energy_logs_df(bad_df)


if __name__ == "__main__":
    unittest.main()
