from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AlgorithmOption(BaseModel):
    name: str
    category: str
    summary: str
    details: str
    selectable: bool = True


class ProcessingOptions(BaseModel):
    image_rescaling: bool = True
    deskew_method: Literal["hough", "projection_profile", "none"] = "hough"
    threshold_method: Literal["adaptive", "otsu", "none"] = "adaptive"
    denoise_method: Literal["non_local_means", "gaussian", "none"] = "non_local_means"
    ocr_engine: Literal["tesseract", "transformer_seq_label", "none"] = "tesseract"
    layout_detector: Literal["rule_based", "faster_rcnn", "mask_rcnn", "none"] = "rule_based"
    text_embedding: Literal["tfidf", "sentence_transformer", "bm25"] = "tfidf"
    visual_similarity: Literal["phash", "ssim", "cnn_embedding"] = "phash"
    local_features: Literal["sift", "orb", "akaze", "none"] = "orb"
    ann_index: Literal["hnsw", "ivf", "lsh", "none"] = "hnsw"
    clustering: Literal["hdbscan", "agglomerative", "kmeans", "none"] = "hdbscan"
    calibration: Literal["logistic_regression", "rule_based"] = "logistic_regression"


class ProcessRequest(BaseModel):
    doc_id: str
    doc_type: Literal["invoice", "prescription", "lab_report"]
    provider_name: Optional[str] = None
    pages: List[str] = Field(default_factory=list, description="OCR text per page")
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)


class SimilarityBreakdown(BaseModel):
    visual: float
    layout: float
    text: float
    schema: float


class MatchResult(BaseModel):
    candidate_doc_id: str
    candidate_template_id: str
    score: float
    breakdown: SimilarityBreakdown
    page_line_matches: List[Dict[str, object]] = Field(default_factory=list)


class ProcessResponse(BaseModel):
    doc_id: str
    document_type: str
    template_signature: Dict[str, object]
    top_matches: List[MatchResult]
    fraud_flag: Literal["NONE", "AMBER", "RED"]
    risk_summary: str


class TrainRequest(BaseModel):
    dataset_path: str = Field(description="CSV/JSON path containing labeled samples")
    label_column: str = "is_fraud"
    feature_columns: List[str] = Field(default_factory=lambda: ["visual", "layout", "text", "schema"])


class TrainResponse(BaseModel):
    trained: bool
    model_path: str
    message: str
    metrics: Dict[str, float]
