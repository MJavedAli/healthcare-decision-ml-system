# clinical-mlops-platform

ML platform for healthcare decision systems with real-time inference, schema validation, observability (logs & metrics), drift detection, and safe deployment/rollback strategies.



## Overview

This repository demonstrates how to take ML from notebook → **reliable production service** in a healthcare context.

It implements a shared platform that supports multiple use cases while enforcing engineering best practices:
- contracts
- reproducibility
- observability
- safe releases

### Included use cases

1. **Appointment No-Show Risk**  
   Predicts the probability a patient will miss an appointment to enable reminders and overbooking strategies.

2. **Patient Readmission Risk**  
   Predicts likelihood of 30-day readmission to trigger follow-up care and reduce penalties.



## High-level architecture

Data → Training Pipeline → Model Artifact (+ metadata)
↓
Model Loader
↓
Client → FastAPI Inference → Logs (JSON) + Metrics (Prometheus)
↓
Drift Checks → Retrain / Rollback


## Tech stack

- Python, pandas, scikit-learn
- FastAPI (serving), Pydantic (schemas)
- MLflow (experiment tracking)
- Prometheus client (metrics)
- Docker (packaging)



## Repository structure

src/
common/        # schemas, shared utils
training/      # training pipeline
serving/       # FastAPI app, model loader
monitoring/    # metrics, drift checks

models/          # saved artifacts + metadata
mlruns/          # MLflow runs
tests/

Dockerfile
requirements.txtHere’s your clean, properly formatted GitHub-ready Markdown 👇 (copy-paste directly)

## High-level architecture

Data → Training Pipeline → Model Artifact (+ metadata)
↓
Model Loader
↓
Client → FastAPI Inference → Logs (JSON) + Metrics (Prometheus)
↓
Drift Checks → Retrain / Rollback




## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt

2) Train (creates model artifact)

python -m src.training.train
```

3) Serve
```
uvicorn src.serving.app:app --reload
```


Test prediction
```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "prior_no_shows": 2,
    "lead_time_hours": 72,
    "num_previous_appointments": 8,
    "is_weekend": false,
    "distance_km": 12.5
  }'

```

Health & metrics
```
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

### Operational scenarios

Happy path
```
Valid request → schema validation → model inference → response + logs + metrics.
```
### Bad Input
Invalid payload rejected by schema (no model invocation).

### Drift
Feature distribution shifts (e.g., scheduling policy change) → drift check flags → retrain/rollback decision.

### Failure
Model load error or spike in errors/latency → alert → rollback to previous version.

## Key Concepts

- **Training–serving skew** → ensure identical feature transformations  
- **Artifact vs code** → deploy model artifact, not training code  
- **Observability** → logs (traceability) + metrics (SLOs: latency, errors)  
- **Drift detection** → monitor input distribution vs training baseline  
- **Safe rollout** → shadow/canary before full promotion  
- **Rollback** → revert on SLO breach or business metric degradation  


## Example SLOs

- p95 latency < 100 ms  
- error rate < 1%  
- 100% requests validated by schema  
- model version attached to every prediction  

## Extending to LLM-era

This platform generalizes to LLM services by adding:

- prompt/schema validation  
- guardrails (safety, PII)  
- evals (quality, hallucination checks)  
- trace logging for prompts/responses  

## Roadmap

- Docker Compose with Prometheus + Grafana dashboards  
- CI (lint/tests) + CD (build/push image)  
- Model registry + automated promotion  
- Canary/shadow routing  
- Feature store (offline/online parity)  

## License

MIT
