from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


class ModelTrainer:
    def __init__(self, model_dir: str = "data/models") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            return pd.DataFrame(json.loads(path.read_text()))
        raise ValueError("Only CSV/JSON datasets are supported")

    def train(self, dataset_path: str, feature_columns: List[str], label_column: str) -> Dict[str, object]:
        df = self._load_dataset(dataset_path)
        missing = [c for c in feature_columns + [label_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = df[feature_columns].fillna(0.0)
        y = df[label_column].astype(int)

        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        pred = model.predict(X)

        metrics = {
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "samples": float(len(df)),
        }

        out_path = self.model_dir / "fraud_calibrator.pkl"
        with out_path.open("wb") as f:
            pickle.dump({"model": model, "features": feature_columns}, f)

        return {"model_path": str(out_path), "metrics": metrics}
