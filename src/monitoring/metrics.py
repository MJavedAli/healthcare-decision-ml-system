from prometheus_client import Counter, Histogram

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total", "Total prediction requests"
)
PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total", "Total prediction errors"
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
