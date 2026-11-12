from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from subsystem2.exceptions_subsystem2 import ModelTrainingError





# Calculator : calculate "how many kWh" for given conditions 

TARGET = "EnergyConsumption"


FEATURE_COLS = [
    "HVACUsage", "LightingUsage", "Occupancy",        
    "Temperature", "Humidity",                        
    "SquareFootage", "max_occupancy",                 
    "Hour", "Month", "Holiday",                       
    "RenewableEnergy",                            
]

FEATURE_PREFIXES = ["Day_", "Bld_"]


class AImodel:
    """Select features, split, train a Random Forest and predict."""

    def __init__(self, n_estimators: int = 200, max_depth: int | None = None,
                 random_state: int = 42, test_size: float = 0.2):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,        
            random_state=random_state, 
            n_jobs=-1,                   
        )
        self.test_size = test_size
        self.feature_names = None        
        self.is_trained = False
        
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    # Pick the necessary columns
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        cols += [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"  Warning: feature columns not found: {missing}")

        X = df[cols]
        non_numeric = X.select_dtypes(exclude="number").columns
        if len(non_numeric) > 0:
            raise RuntimeError(f"Non-numeric features: {list(non_numeric)}. "
                               "Run Dataprocessor.process() first.")
        return X

    
    # Training model with clean_df
    def train(self, clean_df: pd.DataFrame) -> "AImodel":
        X = self.select_features(clean_df)
        y = clean_df[TARGET]

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False)

        self.feature_names = list(X.columns)
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print(f"  Train: {len(self.X_train)} rows, Test: {len(self.X_test)} rows, "
              f"{len(self.feature_names)} features")
        return self

   
    # Prediction part 
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise ModelTrainingError("Call train() before predict().")
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return pd.Series(self.model.predict(X), index=X.index, name="predicted_kWh")

    def history_X(self) -> pd.DataFrame:
        """All feature rows (train + test) — input for forecasting and savings."""
        return pd.concat([self.X_train, self.X_test])

    def feature_importance(self) -> pd.Series:
        return (pd.Series(self.model.feature_importances_, index=self.feature_names)
                  .sort_values(ascending=False))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "AImodel":
        return joblib.load(path)
