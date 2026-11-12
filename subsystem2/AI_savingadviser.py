from dataclasses import dataclass
import pandas as pd


BUILDING_PREFIX = "Bld_"
DAYS_PER_MONTH = 30
META_COLS = ["Timestamp", "building_id", "cost_per_kwh"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class Recommendation:
    """One saving tip for one building, with the money behind it."""
    building_id: str
    title: str
    message: str
    wasted_hours: int
    saved_kwh: float
    saved_eur: float
    saved_eur_per_month: float
    saved_pct: float


class SavingScenario:
    title = "Base scenario"
    switch_off: list[str] = []

    def waste_mask(self, X: pd.DataFrame) -> pd.Series:
        """True for hours where energy is wasted. Child classes define the condition."""
        raise NotImplementedError

    def apply(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Copy of X with the devices switched off in wasted hours."""
        mask = self.waste_mask(X)
        X_new = X.copy()
        X_new.loc[mask, self.switch_off] = 0
        return X_new, mask

    def message(self, saved_eur_per_month: float, wasted_hours: int) -> str:
        return f"{self.title}: save about {saved_eur_per_month:.2f} € per month."


class HVACOffWhenEmpty(SavingScenario):
    """Nobody in the room but the HVAC is on → switch the HVAC off."""
    title = "Switch off HVAC in empty rooms"
    switch_off = ["HVACUsage"]

    def waste_mask(self, X: pd.DataFrame) -> pd.Series:
        return (X["Occupancy"] == 0) & (X["HVACUsage"] == 1)

    def message(self, saved_eur_per_month: float, wasted_hours: int) -> str:
        return (f"The HVAC was running in {wasted_hours} hours with nobody inside. "
                f"Switching it off saves about {saved_eur_per_month:.2f} € per month.")


class LightsOffWhenEmpty(SavingScenario):
    """Nobody in the room but the lights are on → switch the lights off."""
    title = "Switch off lights in empty rooms"
    switch_off = ["LightingUsage"]

    def waste_mask(self, X: pd.DataFrame) -> pd.Series:
        return (X["Occupancy"] == 0) & (X["LightingUsage"] == 1)

    def message(self, saved_eur_per_month: float, wasted_hours: int) -> str:
        return (f"The lights were on in {wasted_hours} hours with nobody inside. "
                f"Switching them off saves about {saved_eur_per_month:.2f} € per month.")


class SavingsAdvisor:
    """Estimate how much money each saving scenario would save per building."""

    def __init__(self, model, raw_df: pd.DataFrame, scenarios: list[SavingScenario] | None = None):
        if not model.is_trained:
            raise RuntimeError("Train the AIforecast model before creating SavingsAdvisor.")

        self.model = model
        self.scenarios = scenarios or [HVACOffWhenEmpty(), LightsOffWhenEmpty()]

        self.history_X = model.history_X()
        meta = raw_df.loc[self.history_X.index, META_COLS].copy()
        meta["Timestamp"] = pd.to_datetime(meta["Timestamp"], format=TIME_FORMAT)
        self.price = meta["cost_per_kwh"]
        self.meta = meta

        self.building_cols = [c for c in self.history_X.columns if c.startswith(BUILDING_PREFIX)]
        if not self.building_cols:
            raise RuntimeError("No 'Bld_' columns found — one-hot encode 'building_id' first.")
        self.n_days = (meta["Timestamp"].max() - meta["Timestamp"].min()) / pd.Timedelta(days=1)

    def buildings(self) -> list[str]:
        return sorted(c[len(BUILDING_PREFIX):] for c in self.building_cols)

    def evaluate(self, building_id: str, scenario: SavingScenario) -> Recommendation:
        bld_col = f"{BUILDING_PREFIX}{building_id}"
        if bld_col not in self.building_cols:
            raise ValueError(f"Unknown building '{building_id}'. Available: {self.buildings()}")

        hist = self.history_X[self.history_X[bld_col] == 1]
        price = self.price.loc[hist.index]

        base_kwh = self.model.predict(hist)
        changed_X, mask = scenario.apply(hist)
        new_kwh = self.model.predict(changed_X)

        saved_kwh = (base_kwh - new_kwh).clip(lower=0)
        saved_eur = float((saved_kwh * price).sum())
        base_eur = float((base_kwh * price).sum())

        per_month = saved_eur / self.n_days * DAYS_PER_MONTH
        wasted_hours = int(mask.sum())

        return Recommendation(
            building_id=building_id,
            title=scenario.title,
            message=scenario.message(per_month, wasted_hours),
            wasted_hours=wasted_hours,
            saved_kwh=round(float(saved_kwh.sum()), 2),
            saved_eur=round(saved_eur, 2),
            saved_eur_per_month=round(per_month, 2),
            saved_pct=round(100 * saved_eur / base_eur, 2) if base_eur else 0.0,
        )

    def recommend(self, building_id: str) -> list[Recommendation]:
        recs = [self.evaluate(building_id, s) for s in self.scenarios]
        recs = [r for r in recs if r.wasted_hours > 0]
        return sorted(recs, key=lambda r: r.saved_eur_per_month, reverse=True)

    def recommend_all(self) -> dict[str, list[Recommendation]]:
        """{building_id: [Recommendation, ...]} for every building."""
        results = {}
        for building_id in self.buildings():
            results[building_id] = self.recommend(building_id)
            total = sum(r.saved_eur_per_month for r in results[building_id])
            print(f"  {building_id}: {len(results[building_id])} tips, "
                  f"up to {total:.2f} € / month")
        return results
