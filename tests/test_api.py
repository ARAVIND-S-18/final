from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_options_endpoint():
    r = client.get('/options/algorithms')
    assert r.status_code == 200
    body = r.json()
    assert any(item['name'].startswith('Image Rescaling') for item in body)


def test_process_endpoint():
    payload = {
        'doc_id': 'doc_x',
        'doc_type': 'invoice',
        'pages': ['Invoice\nCity Care Hospital\nConsultation 1200\nTotal 4700'],
    }
    r = client.post('/process', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 'top_matches' in body
    assert body['fraud_flag'] in {'NONE', 'AMBER', 'RED'}


def test_train_endpoint():
    payload = {
        'dataset_path': 'data/sample_training.csv',
        'label_column': 'is_fraud',
        'feature_columns': ['visual', 'layout', 'text', 'schema'],
    }
    r = client.post('/train', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body['trained'] is True
