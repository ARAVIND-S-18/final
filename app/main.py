from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import AlgorithmOption, ProcessRequest, ProcessResponse, TrainRequest, TrainResponse
from .pipeline import TemplateEngine, algorithm_catalog
from .training import ModelTrainer

app = FastAPI(title="Similar Document Template Matching API", version="1.0.0")
engine = TemplateEngine()
trainer = ModelTrainer()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/options/algorithms", response_model=list[AlgorithmOption])
def list_algorithms() -> list[AlgorithmOption]:
    return [AlgorithmOption(**row) for row in algorithm_catalog()]


@app.post("/process", response_model=ProcessResponse)
def process_document(payload: ProcessRequest) -> ProcessResponse:
    signature, matches, flag, reason = engine.compare(payload)
    return ProcessResponse(
        doc_id=payload.doc_id,
        document_type=payload.doc_type,
        template_signature=signature,
        top_matches=matches,
        fraud_flag=flag,
        risk_summary=reason,
    )


@app.post("/train", response_model=TrainResponse)
def train_model(payload: TrainRequest) -> TrainResponse:
    try:
        out = trainer.train(payload.dataset_path, payload.feature_columns, payload.label_column)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TrainResponse(
        trained=True,
        model_path=out["model_path"],
        message="Training complete. Fraud calibration model updated from uploaded dataset.",
        metrics=out["metrics"],
    )
