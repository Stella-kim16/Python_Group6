from dataclasses import dataclass
import pandas as pd


DAY_PREFIX = "Day_"
BUILDING_PREFIX = "Bld_"
HOLIDAY_COL = "Holiday"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
META_COLS = ["Timestamp", "building_id", "cost_per_kwh", "EnergyConsumption"]


@dataclass
class ForecastResult:
    """Everything one building's Demand Forecast needs to show."""
    building_id: str
    hourly: pd.DataFrame
    total_kwh: float
    total_cost_eur: float
    recent_actual: pd.DataFrame


class DemandForecaster:
    """Predict the next 24 hours of energy use for each building."""

    def __init__(self, model, raw_df: pd.DataFrame, min_rows: int = 3):
        if not model.is_trained:
            raise RuntimeError("Train the AIforecast model before creating DemandForecaster.")

        self.model = model
        self.min_rows = min_rows

        self.history_X = model.history_X()

        self.meta = self._build_meta(raw_df)

        self.day_cols = [c for c in self.history_X.columns if c.startswith(DAY_PREFIX)]
        self.building_cols = [c for c in self.history_X.columns if c.startswith(BUILDING_PREFIX)]
        if not self.building_cols:
            raise RuntimeError(
                "No 'Bld_' columns found. In Data_processor.encode_categoricals(), "
                "one-hot encode 'building_id' with prefix='Bld' and drop_first=False.")
        self.binary_cols = [c for c in self.history_X.columns
                            if set(self.history_X[c].dropna().unique()) <= {0, 1}]

        self.hourly_price = (self.meta.groupby(self.meta["Timestamp"].dt.hour)["cost_per_kwh"]
                                      .mean())

    def _build_meta(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Side information for the same rows as history_X (anomaly rows are already gone)."""
        missing = [c for c in META_COLS if c not in raw_df.columns]
        if missing:
            raise KeyError(f"Columns missing in raw CSV: {missing}")
        meta = raw_df.loc[self.history_X.index, META_COLS].copy()
        meta["Timestamp"] = pd.to_datetime(meta["Timestamp"], format=TIME_FORMAT)
        return meta

    def _is_day(self, X: pd.DataFrame, day_name: str) -> pd.Series:
        """True for rows on day_name. The day dropped by drop_first has all Day_ columns = 0."""
        col = f"{DAY_PREFIX}{day_name}"
        if col in X.columns:
            return X[col] == 1
        return (X[self.day_cols] == 0).all(axis=1)

    def _typical_row(self, hist: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        """
        Typical input values for this building at time ts.
        Search widens step by step until at least min_rows past rows are found:
        same weekday + hour → same hour → hour ± 1 → whole building history.
        """
        candidates = [
            (hist["Hour"] == ts.hour) & self._is_day(hist, ts.day_name()),
            hist["Hour"] == ts.hour,
            hist["Hour"].isin([(ts.hour - 1) % 24, ts.hour, (ts.hour + 1) % 24]),
        ]
        rows = hist
        for mask in candidates:
            if mask.sum() >= self.min_rows:
                rows = hist[mask]
                break

        typical = rows.median()
        typical[self.binary_cols] = typical[self.binary_cols].round()
        return typical

    def buildings(self) -> list[str]:
        """Building IDs that can be forecast, e.g. ['BLD_HOM_01', ...]."""
        return sorted(c[len(BUILDING_PREFIX):] for c in self.building_cols)

    def forecast_building(self, building_id: str, start: pd.Timestamp | None = None,
                          holidays=(), hours: int = 24) -> ForecastResult:
        bld_col = f"{BUILDING_PREFIX}{building_id}"
        if bld_col not in self.building_cols:
            raise ValueError(f"Unknown building '{building_id}'. Available: {self.buildings()}")
        hist = self.history_X[self.history_X[bld_col] == 1]

        if start is None:
            start = self.meta["Timestamp"].max() + pd.Timedelta(hours=1)

        future_hours = pd.date_range(start=pd.Timestamp(start).floor("h"),
                                     periods=hours, freq="h")
        holiday_dates = {pd.Timestamp(d).date() for d in holidays}

        rows = []
        for ts in future_hours:
            row = self._typical_row(hist, ts)

            row["Hour"] = ts.hour
            if "Month" in row.index:
                row["Month"] = ts.month
            row[self.day_cols] = 0
            if f"{DAY_PREFIX}{ts.day_name()}" in row.index:
                row[f"{DAY_PREFIX}{ts.day_name()}"] = 1
            if HOLIDAY_COL in row.index:
                row[HOLIDAY_COL] = int(ts.date() in holiday_dates)
            rows.append(row)

        X_next = pd.DataFrame(rows, index=future_hours)

        kwh = self.model.predict(X_next).values
        price = self.hourly_price.reindex(future_hours.hour).values
        hourly = pd.DataFrame({
            "predicted_kWh": kwh,
            "price_per_kWh": price,
            "predicted_cost_eur": kwh * price,
        }, index=future_hours)

        bld_meta = self.meta[self.meta["building_id"] == building_id]
        recent = bld_meta[bld_meta["Timestamp"] > future_hours[0] - pd.Timedelta(days=2)]
        recent_actual = recent.set_index("Timestamp")[["EnergyConsumption"]]

        return ForecastResult(
            building_id=building_id,
            hourly=hourly,
            total_kwh=round(float(kwh.sum()), 2),
            total_cost_eur=round(float((kwh * price).sum()), 2),
            recent_actual=recent_actual,
        )

    def forecast_all(self, start: pd.Timestamp | None = None,
                     holidays=()) -> dict[str, ForecastResult]:
        """{building_id: ForecastResult} for every building."""
        results = {}
        for building_id in self.buildings():
            results[building_id] = self.forecast_building(building_id, start, holidays)
            r = results[building_id]
            print(f"  {building_id}: {r.total_kwh:.1f} kWh, {r.total_cost_eur:.2f} €")
        return results
