"""Typed errors specific to the data-ingestion layer.

Why typed classes instead of a generic `Exception`?
Interview answer: in production, "an error happened" is not enough; the
action taken depends on the TYPE of error:
  - Transient network error (ApiTransientError)  -> retry / failover source
  - Permanent error, e.g. wrong API key          -> retry is POINTLESS,
    raise an alert
  - Data arrived but is corrupt (DataQualityError) -> stop the pipeline,
    never feed corrupt data to the model ("fail loudly, not silently").
"""


class IngestionError(Exception):
    """Common ancestor of all errors in the ingestion layer."""


class ApiTransientError(IngestionError):
    """Transient error: timeout, 5xx, rate-limit. Retryable."""


class ApiPermanentError(IngestionError):
    """Permanent error: 401/403 (bad key), 400 (malformed request). Not retryable."""


class DataQualityError(IngestionError):
    """The API responded but the data is unusable (empty, broken schema, etc.)."""
