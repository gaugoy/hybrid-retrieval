"""
Project-specific exceptions.

WHY
  Catching a bare ``Exception`` hides bugs. A small hierarchy lets callers (the CLI, the API,
  the eval runner) handle "your data isn't downloaded yet" differently from "the LLM provider
  is down". Messages should tell the reader what to run to fix the problem.
"""


class FinsightError(Exception):
    """Base class for every error raised on purpose by finsight."""


class ConfigError(FinsightError):
    """Configuration is missing, malformed, or has invalid values."""


class DataNotFoundError(FinsightError):
    """An expected data artifact (raw file, parquet) is missing. The message names the fix."""


class IndexNotBuiltError(FinsightError):
    """A Qdrant collection or BM25 index is missing or out of date."""


class ProviderError(FinsightError):
    """An external provider (Ollama, Anthropic, Cohere, Gemini, ...) failed after retries."""


class RateLimitedError(ProviderError):
    """Still rate-limited (HTTP 429) after every retry. Lower the preset's rpm/tpm and re-run."""


class QuotaExhaustedError(ProviderError):
    """A daily/monthly quota or the run's token budget is used up. Work done so far is cached:
    re-run after the quota resets and it continues where it stopped."""


class MissingAPIKeyError(ConfigError):
    """An online provider was selected but its API key is not in the environment / .env."""


class PaidProviderBlockedError(ConfigError):
    """A paid-only provider was selected while embedding.allow_paid is false."""
