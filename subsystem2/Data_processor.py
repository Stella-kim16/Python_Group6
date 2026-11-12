import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from subsystem2.exceptions_subsystem2 import ModelTrainingError


class Dataloader:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overview(self) -> None:
        try:
            print(f"shape: {self.df.shape[0]}rows, {self.df.shape[1]}columns")
            print("column data type:", self.df.dtypes)
            print("first five head: ", self.df.head())
        except Exception as e:
            raise ModelTrainingError(f"Error displaying dataset overview: {e}") from e

    def missing_report(self) -> pd.DataFrame:
        try:
            missing_counts = self.df.isnull().sum()
            missing_pct = (missing_counts / len(self.df) * 100).round(2)
            report = pd.DataFrame({
                "missing_counts:": missing_counts,
                "missing_pct:": missing_pct
            })
            report = report[report["missing_counts:"] > 0]

            if report.empty:
                print("no missing values")
            else:
                print(report)
            return report
        except Exception as e:
            raise ModelTrainingError(f"Error generating missing value report: {e}") from e


class Dataprocessor:
    def __init__(self, df: pd.DataFrame, target: str = "EnergyConsumption"):
        try:
            self.df = df.copy()
            self.target = target
        except Exception as e:
            raise ModelTrainingError(f"Failed to initialize Dataprocessor: {e}") from e

    def add_time_features(self, time_col: str = "Timestamp",
                          fmt: str = "%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
        try:
            self.df[time_col] = pd.to_datetime(self.df[time_col], format=fmt)
            self.df = self.df.sort_values(time_col)
            self.df["Hour"] = self.df[time_col].dt.hour
            self.df["Month"] = self.df[time_col].dt.month
            self.df = self.df.drop(columns=[time_col])

            return self.df
        except Exception as e:
            raise ModelTrainingError(f"Failed to add time features from column '{time_col}': {e}") from e

    def encode_categoricals(self) -> pd.DataFrame:
        try:
            binary_maps = {
                "HVACUsage": {"On": 1, "Off": 0},
                "LightingUsage": {"On": 1, "Off": 0},
                "Holiday": {"Yes": 1, "No": 0},
            }
            for col, mapping in binary_maps.items():
                if col not in self.df.columns:
                    print(f"  Warning: Column '{col}' not found for binary encoding.")
                    continue
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    print(f"  '{col}' is already numeric, skipped.")
                    continue
                self.df[col] = self.df[col].map(mapping)
                if self.df[col].isnull().any():
                    raise ValueError(f"'{col}' has values not in {list(mapping)}")
                print(f"  Encoded '{col}' as binary (1/0)")

            if "DayOfWeek" in self.df.columns:
                self.df = pd.get_dummies(self.df, columns=["DayOfWeek"], prefix="Day",
                                         drop_first=True, dtype=int)
                print("  One-hot encoded 'DayOfWeek' (drop_first=True to avoid redundancy)")
            else:
                print("  Warning: 'DayOfWeek' column not found for one-hot encoding.")

            self.features = [c for c in self.df.columns if c != self.target]
            if "building_id" in self.df.columns:
                self.df = pd.get_dummies(self.df, columns=["building_id"], prefix="Bld",
                                         drop_first=False, dtype=int)
                print("  One-hot encoded 'building_id'")
            return self.df
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to encode categorical variables: {e}") from e

    def process(self) -> pd.DataFrame:
        try:
            self.add_time_features()
            self.encode_categoricals()
            return self.df
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Error executing Dataprocessor pipeline: {e}") from e


class Dataspliting:
    def __init__(self, df: pd.DataFrame, target: str = "EnergyConsumption"):
        try:
            self.df = df.copy()
            self.target = target
            self.features = [c for c in self.df.columns if c != self.target]
            self.X_train = self.X_test = None
            self.y_train = self.y_test = None
        except Exception as e:
            raise ModelTrainingError(f"Failed to initialize Dataspliting: {e}") from e

    def split(self, test_size: float = 0.2):
        try:
            non_numeric = self.df[self.features].select_dtypes(exclude="number").columns
            if len(non_numeric) > 0:
                raise RuntimeError(f"Non-numeric columns left: {list(non_numeric)}. "
                                   "Run Dataprocessor.process() first.")

            X = self.df[self.features]
            y = self.df[self.target]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False)

            print(f"  Train: {len(self.X_train)} rows, Test: {len(self.X_test)} rows")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            if isinstance(e, ModelTrainingError):
                raise
            raise ModelTrainingError(f"Failed to split dataset for training and testing: {e}") from e


CSV_PATH = "source/enriched_energy_data.csv"


def load_clean_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load data from the CSV file and return the fully processed DataFrame safely."""
    try:
        raw_df = pd.read_csv(csv_path)
        processor = Dataprocessor(raw_df)
        clean_df = processor.process()
        return clean_df
    except Exception as e:
        if isinstance(e, ModelTrainingError):
            raise
        raise ModelTrainingError(f"Failed to load and clean data from '{csv_path}': {e}") from e