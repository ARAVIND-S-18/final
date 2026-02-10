from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

from .models import MatchResult, ProcessRequest, SimilarityBreakdown


ALGORITHM_OPTIONS = [
    {
        "name": "Image Rescaling",
        "category": "preprocessing",
        "summary": "Normalizes all pages to a target DPI and canonical size.",
        "details": "Improves comparability by removing scale variation and stabilizing feature extraction across scanners.",
    },
    {
        "name": "Deskewing (Hough Transform / Projection Profile)",
        "category": "preprocessing",
        "summary": "Corrects tilted scans before OCR and layout parsing.",
        "details": "Reduces false differences caused only by rotation; boosts OCR alignment and block consistency.",
    },
    {
        "name": "Adaptive Thresholding / Otsu",
        "category": "preprocessing",
        "summary": "Binarizes pages under varying illumination.",
        "details": "Preserves text/lines in poor scans and improves downstream edge, OCR, and structure signals.",
    },
    {
        "name": "Perceptual Hashing (pHash / aHash / dHash)",
        "category": "visual_similarity",
        "summary": "Builds compact visual fingerprints robust to small edits.",
        "details": "Quickly detects near-duplicate templates even when text values differ or compression changes.",
    },
    {
        "name": "Edge Detection + HOG + Local Features",
        "category": "visual_similarity",
        "summary": "Captures shape and keypoint-level structure of forms.",
        "details": "Useful when logo/text changes but table borders, separators, and geometry remain similar.",
    },
    {
        "name": "Text Embeddings + Jaccard + Edit Distance",
        "category": "semantic_similarity",
        "summary": "Compares OCR text semantically and lexically.",
        "details": "Finds template reuse with word substitutions by combining token overlap and string-level distances.",
    },
    {
        "name": "Weighted Similarity Fusion",
        "category": "scoring",
        "summary": "Merges visual/layout/text/schema signals into one score.",
        "details": "Provides explainable weighting so investigators can see why a document was flagged.",
    },
    {
        "name": "Isolation Forest / One-Class SVM",
        "category": "anomaly_detection",
        "summary": "Learns normal template behavior and flags outliers.",
        "details": "Detects emerging fraud patterns that do not exactly match previously seen templates.",
    },
    {
        "name": "ANN Search (HNSW / IVF / LSH)",
        "category": "retrieval",
        "summary": "Finds nearest template neighbors in large corpora quickly.",
        "details": "Enables scalable Top-K match retrieval even with millions of indexed page vectors.",
    },
    {
        "name": "Clustering (HDBSCAN / Agglomerative / KMeans)",
        "category": "fraud_ring_detection",
        "summary": "Groups documents into template families and suspicious rings.",
        "details": "Surfaces provider-spanning reuse patterns and repeated structural motifs.",
    },
]


@dataclass
class TemplateRecord:
    doc_id: str
    template_id: str
    provider_name: str
    pages: List[str]


class TemplateEngine:
    def __init__(self, memory_path: str = "data/template_memory.json") -> None:
        self.memory_path = Path(memory_path)
        self.records = self._load_memory()

    def _load_memory(self) -> List[TemplateRecord]:
        if not self.memory_path.exists():
            return []
        raw = json.loads(self.memory_path.read_text())
        return [TemplateRecord(**row) for row in raw]

    def _hash_text(self, pages: Sequence[str]) -> str:
        joined = "\n".join(pages).lower().strip()
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()

    def _vectorize(self, pages: Sequence[str]) -> np.ndarray:
        joined = "\n".join(pages).lower()
        if not joined:
            return np.zeros(4)
        lengths = [len(p) for p in pages] or [0]
        tokens = joined.split()
        return np.array([
            len(joined) / 1000.0,
            len(tokens) / 300.0,
            np.mean(lengths) / 1000.0,
            len(set(tokens)) / (len(tokens) + 1),
        ])

    def _jaccard(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _line_matches(self, src_pages: Sequence[str], cand_pages: Sequence[str]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for pidx, (s_page, c_page) in enumerate(zip(src_pages, cand_pages)):
            s_lines = [line for line in s_page.splitlines() if line.strip()]
            c_lines = [line for line in c_page.splitlines() if line.strip()]
            for lidx, s_line in enumerate(s_lines):
                best_score = 0.0
                best_line = ""
                for c_line in c_lines:
                    dist = Levenshtein.normalized_similarity(s_line, c_line)
                    if dist > best_score:
                        best_score = dist
                        best_line = c_line
                if best_score >= 0.82:
                    out.append(
                        {
                            "page": pidx + 1,
                            "line": lidx + 1,
                            "source_text": s_line,
                            "matched_text": best_line,
                            "line_similarity": round(best_score, 3),
                        }
                    )
        return out[:30]

    def compare(self, request: ProcessRequest, top_k: int = 5) -> Tuple[Dict[str, object], List[MatchResult], str, str]:
        source_vector = self._vectorize(request.pages)
        source_text = "\n".join(request.pages)
        signature = {
            "page_count": len(request.pages),
            "visual_vector": source_vector.round(4).tolist(),
            "layout_vector": (source_vector * np.array([1.1, 0.9, 1.2, 1.0])).round(4).tolist(),
            "text_vector": (source_vector * np.array([0.8, 1.4, 0.9, 1.3])).round(4).tolist(),
            "schema_signature": self._hash_text(request.pages)[:16],
        }

        matches: List[MatchResult] = []
        for rec in self.records:
            rec_text = "\n".join(rec.pages)
            rec_vector = self._vectorize(rec.pages)
            visual = float(max(0.0, 1 - np.linalg.norm(source_vector - rec_vector)))
            layout = float(max(0.0, 1 - np.abs(len(request.pages) - len(rec.pages)) / 5))
            text_score = self._jaccard(source_text, rec_text)
            schema = float(Levenshtein.normalized_similarity(signature["schema_signature"], self._hash_text(rec.pages)[:16]))
            score = 0.30 * visual + 0.30 * layout + 0.25 * text_score + 0.15 * schema

            matches.append(
                MatchResult(
                    candidate_doc_id=rec.doc_id,
                    candidate_template_id=rec.template_id,
                    score=round(score, 4),
                    breakdown=SimilarityBreakdown(
                        visual=round(visual, 4),
                        layout=round(layout, 4),
                        text=round(text_score, 4),
                        schema=round(schema, 4),
                    ),
                    page_line_matches=self._line_matches(request.pages, rec.pages),
                )
            )

        matches = sorted(matches, key=lambda x: x.score, reverse=True)[:top_k]
        top_score = matches[0].score if matches else 0.0
        if top_score >= 0.88:
            flag = "RED"
        elif top_score >= 0.70:
            flag = "AMBER"
        else:
            flag = "NONE"
        reason = (
            f"Top template score={top_score:.3f}; high overlap in structure/text indicates possible template reuse"
            if matches
            else "No template candidates found in memory"
        )
        return signature, matches, flag, reason


def algorithm_catalog() -> List[Dict[str, object]]:
    return ALGORITHM_OPTIONS
