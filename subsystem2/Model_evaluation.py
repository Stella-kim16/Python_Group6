import numpy as np

from subsystem2.Prediction_result import PredictionResult
from subsystem2.exceptions_subsystem2 import ModelEvaluationError

class ModelEvaluator:
    """Evaluate a PredictionResult with MAE and MSE."""

    def __init__(self):
        self.mae = None
        self.mse = None

    def evaluate(self, prediction: PredictionResult) -> dict:
        
        pairs = [(p, a) for p, a in zip(prediction.getPredictions(), prediction.getActualValues())
                 if a is not None]
        if not pairs:
            raise ModelEvaluationError("No actual values to compare. Use addActualValues() first.")

        self._y_pred = np.array([p for p, _ in pairs])   
        self._y_true = np.array([a for _, a in pairs])   

        self.mae = self.calculateMAE()
        self.mse = self.calculateMSE()
        return self.getMetrics()

    def calculateMAE(self) -> float:
        return float(np.mean(np.abs(self._y_true - self._y_pred)))

    def calculateMSE(self) -> float:
        return float(np.mean((self._y_true - self._y_pred) ** 2))

    def getMetrics(self) -> dict:
        return {"MAE": round(self.mae, 3), "MSE": round(self.mse, 3)}
