# Sentinel — Real-Time ML Inference Pipeline

End-to-end market signal prediction engine on GCP. Ingests stock data from Twelve Data, computes technical indicators, runs a trained classifier, and serves BULLISH / BEARISH / NEUTRAL signals with confidence scores. Supports both synchronous and async (Pub/Sub) inference paths.

**[Live Demo](https://adsouza5.github.io/portfolio-react/projects/sentinel)**

## Architecture

```
Twelve Data API
      │  OHLCV + indicators (RSI, MACD, Bollinger Bands, ATR)
      ▼
  FastAPI (Cloud Run)
      │
      ├─▶ /predict        ──▶ scikit-learn classifier ──▶ JSON response
      │
      └─▶ /predict/async  ──▶ Pub/Sub topic
                                   │
                              Push subscriber
                                   │
                           BigQuery (partitioned)
                           Firestore (session state)
```

## Features

- **Synchronous inference** — sub-second predictions via `/predict`
- **Async pipeline** — `/predict/async` publishes to Pub/Sub; results land in BigQuery and Firestore
- **Technical indicators** — RSI, MACD, Bollinger Bands, ATR computed from 252-day OHLCV windows
- **Trained classifier** — scikit-learn model (`model.joblib`) trained on historical market data
- **Live analytics** — `/analytics` endpoint queries BigQuery for rolling prediction breakdowns
- **Infrastructure as code** — full GCP stack provisioned with Terraform

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Python |
| ML | scikit-learn, pandas, numpy |
| Async pipeline | GCP Pub/Sub |
| Storage | BigQuery (partitioned), Firestore |
| Deployment | Cloud Run, Docker |
| IaC | Terraform |
| Market data | Twelve Data API |

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Synchronous ML inference |
| POST | `/predict/async` | Async via Pub/Sub |
| GET | `/results/{session_id}` | Poll async result |
| GET | `/analytics` | Rolling BigQuery analytics |
| GET | `/health` | Health check |

## Local Development

```bash
cd backend
pip install -r requirements.txt
export GCP_PROJECT=your-project
export TWELVE_DATA_API_KEY=your-key
uvicorn main:app --reload
```

## Deployment

```bash
gcloud builds submit backend/ --tag <IMAGE>
gcloud run deploy ml-pipeline-api --image <IMAGE> --allow-unauthenticated
```

Infrastructure:
```bash
cd terraform
terraform init && terraform apply
```
