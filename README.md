# 📰 Fake News Detector

A small web app and REST API that classifies a news article as **real** or **fake**. It uses a TF-IDF text representation and a Logistic Regression classifier, served with Flask.

## Features

- Text classification with TF-IDF (top 10,000 features, English stop words removed) and Logistic Regression
- REST API with `/predict` and `/health` endpoints
- Simple web UI for testing predictions in the browser
- Model is trained automatically when the Docker container starts
- Docker and Docker Compose support

## Project Structure

```
.
├── app.py              # Flask API + serves the web UI
├── train_model.py      # Trains the model and saves it to model.pkl
├── index.html          # Web interface
├── requirements.txt    # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── True.csv            # Dataset: real news (download separately)
└── Fake.csv            # Dataset: fake news (download separately)
```

## Dataset

The model is trained on a real/fake news dataset split into two CSV files:

- `True.csv`: real news articles
- `Fake.csv`: fake news articles

Download the dataset here: **[https://tinyurl.com/4emsn7xe](https://tinyurl.com/4emsn7xe)**

Place `True.csv` and `Fake.csv` in the project root (next to `train_model.py`). The `text` column is used as the model input.

## Model

| Step | Details |
|------|---------|
| Vectorizer | `TfidfVectorizer(stop_words="english", max_features=10000)` |
| Classifier | `LogisticRegression(max_iter=1000)` |
| Labels | `1` = real, `0` = fake |
| Split | 80% train / 20% test (`random_state=42`) |

Training prints the test accuracy and a classification report, then saves the full pipeline to `model.pkl`.

## Getting Started

### Option 1: Docker (recommended)

1. Download the dataset and put `True.csv` and `Fake.csv` in the project root.
2. Build and run:

```bash
   docker compose up --build
```

   The container trains the model on startup, then serves the app with Gunicorn.

3. Open **http://localhost:5000**.

### Option 2: Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates model.pkl)
python train_model.py

# 3. Start the server
python app.py
```

Then open **http://localhost:5000**.

## API Reference

### `GET /health`

Checks that the service is running.

```json
{ "status": "ok", "model": "LogisticRegression + TF-IDF" }
```

### `POST /predict`

Classifies a piece of text.

**Request**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here..."}'
```

**Response**

```json
{
  "prediction": "real",
  "confidence": 97.42,
  "status": "sukses"
}
```

| Field | Description |
|-------|-------------|
| `prediction` | `"real"` or `"fake"` |
| `confidence` | Model confidence for the predicted class, in percent |
| `status` | `"sukses"` on success |

**Errors**

| Code | Cause |
|------|-------|
| `400` | `text` is missing or empty |
| `500` | Unexpected server error |

## Tech Stack

Python 3.11 · Flask · Gunicorn · scikit-learn · pandas · NumPy · Docker

## Limitations

- The model learns the writing style of the training dataset, so it may not generalize well to news from other sources, topics, or languages. It was trained on English text.
- Predictions are statistical estimates and should not be treated as fact-checking.
