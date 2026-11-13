from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd




@dataclass
class PredictionResult:
    modelName: str                       
    timestamps: list[str]               
    predictions: list[float]             
    actualValues: list[float | None]     
    createdAt: datetime = field(default_factory=datetime.now)  

    @classmethod
    def from_series(cls, modelName: str, predicted: pd.Series,
                    actual: pd.Series | None = None, timestamps=None) -> "PredictionResult":
       
        times = predicted.index if timestamps is None else timestamps
        preds = [round(float(v), 3) for v in predicted]
        if actual is None:
            actuals = [None] * len(preds)
        else:
            actuals = [round(float(v), 3) for v in actual.loc[predicted.index]]
        return PredictionResult(modelName, [str(t) for t in times], preds, actuals)

    
    def getPredictions(self) -> list[float]:
        return self.predictions

    def getActualValues(self) -> list[float | None]:
        return self.actualValues
