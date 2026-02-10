# Similar Document Template Matching Algorithm

This repository provides a runnable **FastAPI prototype** for medical invoice/prescription/lab-report template comparison with fraud flagging.
It supports configurable algorithm options, explainable similarity breakdown, page-line level matching (plagiarism-checker style), and user dataset training.

## What is implemented

- Document processing endpoint with configurable options (`/process`).
- Algorithm catalog endpoint with user-selectable methods and concise explanations (`/options/algorithms`).
- Dataset-driven training endpoint to update fraud calibration from user-uploaded datasets (`/train`).
- Template memory store (`data/template_memory.json`) for baseline nearest match retrieval.
- Explainable output with:
  - Top-K similar documents.
  - Visual/layout/text/schema score breakdown.
  - Page + line-level textual similarity evidence.
  - Fraud flag (`NONE`, `AMBER`, `RED`) and reason summary.

## Algorithm options users can choose

Each option below is exposed via `GET /options/algorithms` and can be mapped to pipeline settings:

1. **Image Rescaling**  
   Standardizes document resolution and page size for consistent feature extraction.  
   Helps avoid false mismatch caused by scanner DPI differences.

2. **Deskewing (Hough / Projection Profile)**  
   Fixes rotated scans before OCR and layout analysis.  
   Improves line alignment and block detection reliability.

3. **Adaptive Thresholding / Otsu**  
   Converts noisy scans into cleaner foreground/background separation.  
   Improves OCR quality and edge/shape extraction.

4. **Perceptual Hashing (pHash/aHash/dHash)**  
   Creates compact fingerprints robust to compression and minor edits.  
   Useful for rapid near-duplicate template detection.

5. **Edge + HOG + Local Features (SIFT/ORB/AKAZE)**  
   Captures structure-level traits like boxes, lines, and keypoints.  
   Strong when template geometry is reused with changed text.

6. **Text Embeddings + Jaccard + Edit Distance**  
   Compares OCR content semantically and lexically.  
   Detects reused templates with word replacements or minor edits.

7. **Weighted Similarity Fusion**  
   Combines visual, layout, text, and schema scores into final risk score.  
   Maintains explainability with per-component contribution.

8. **Isolation Forest / One-Class SVM**  
   Learns normal document behavior and flags anomalies.  
   Catches suspicious templates with no exact historical match.

9. **ANN Search (HNSW / IVF / LSH)**  
   Speeds retrieval of nearest templates in large archives.  
   Supports scale while keeping query latency low.

10. **Clustering (HDBSCAN / Agglomerative / KMeans)**  
    Groups records into template families and suspicious rings.  
    Useful for provider-level and cross-provider fraud patterns.

## API quickstart

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 1) View selectable options

```bash
curl http://127.0.0.1:8000/options/algorithms
```

### 2) Process one document and get similarity report

```bash
curl -X POST http://127.0.0.1:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id":"doc_uploaded_001",
    "doc_type":"invoice",
    "provider_name":"City Care Hospital",
    "pages":["Invoice\nCity Care Hospital\nPatient Name: XXXX\nInvoice No: INV-1007\nItem Description Amount\nConsultation 1200\nLab Test 3500\nTotal 4700"]
  }'
```

### 3) Upload your own dataset and train model

Prepare CSV with columns: `visual,layout,text,schema,is_fraud`

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path":"data/sample_training.csv",
    "label_column":"is_fraud",
    "feature_columns":["visual","layout","text","schema"]
  }'
```

The trained calibration model is saved to `data/models/fraud_calibrator.pkl`.

## Plagiarism-checker style output (example)

```json
{
  "doc_id": "doc_uploaded_001",
  "fraud_flag": "RED",
  "top_matches": [
    {
      "candidate_doc_id": "doc_seed_001",
      "candidate_template_id": "invoice_tpl_a",
      "score": 0.956,
      "breakdown": {
        "visual": 0.94,
        "layout": 1.0,
        "text": 0.93,
        "schema": 0.95
      },
      "page_line_matches": [
        {
          "page": 1,
          "line": 1,
          "source_text": "Invoice",
          "matched_text": "Invoice",
          "line_similarity": 1.0
        },
        {
          "page": 1,
          "line": 6,
          "source_text": "Consultation 1200",
          "matched_text": "Consultation 1200",
          "line_similarity": 1.0
        }
      ]
    }
  ],
  "risk_summary": "Top template score=0.956; high overlap in structure/text indicates possible template reuse"
}
```

## Notes

- This is a production-oriented scaffold: algorithm switches are exposed and explainability/reporting are implemented.
- Heavy CV/DL modules (Detectron2, CLIP/ViT, GNN, Milvus/Pinecone) are represented as selectable options and can be plugged into `app/pipeline.py` as next iteration.
