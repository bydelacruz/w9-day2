from prometheus_client import Counter, Histogram

# Metrics definition
# Counter: Counts total number of requests
REQUEST_COUNT = Counter(
    "http_request_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

# Histogram: Measures distribution of request duration
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
