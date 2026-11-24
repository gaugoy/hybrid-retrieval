"""
FallbackLLM: try the primary LLM; on a provider failure, try the next preset in llm.fallbacks.

WHY
  A hosted model can be down, rate-limited or out of free quota. A degraded answer from a
  local model, clearly flagged, beats an error page. Each fallback is built only when first
  needed (no extra model loaded while the primary works), and the response says which model
  answered (`fallback_from`), so the pipeline adds "llm_fallback" to Answer.degraded.

Only runtime provider errors trigger a fallback (ProviderError, timeouts, connection errors).
Config errors (a missing key, a paid provider blocked) are raised: they need a human, not a retry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from finsight.errors import ProviderError
from finsight.generation.llm.base import LLM, LLMResponse
from finsight.observability.logging import get_logger

log = get_logger(__name__)

RUNTIME_ERRORS: tuple[type[BaseException], ...] = (ProviderError, TimeoutError, ConnectionError)


class FallbackLLM(LLM):
    def __init__(self, primary: LLM, fallbacks: list[str], build: Callable[[str], LLM]) -> None:
        self.primary, self.names, self._build = primary, fallbacks, build
        self._built: dict[str, LLM] = {}
        self.provider, self.model = primary.provider, primary.model

    def _chain(self) -> Iterator[tuple[str, LLM]]:
        yield f"{self.primary.provider}:{self.primary.model}", self.primary
        for name in self.names:
            if name not in self._built:
                try:
                    self._built[name] = self._build(name)
                except Exception as exc:
                    log.warning(
                        "fallback llm unavailable", extra={"preset": name, "error": str(exc)}
                    )
                    continue
            yield name, self._built[name]

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        first, errors = None, []
        for name, llm in self._chain():
            first = first or name
            try:
                resp = llm.generate(system, messages, max_tokens, json_mode)
            except RUNTIME_ERRORS as exc:
                log.warning("llm failed, trying the next", extra={"llm": name, "error": str(exc)})
                errors.append(f"{name}: {exc}")
                continue
            if llm is not self.primary:
                resp = resp.model_copy(update={"fallback_from": first})
            return resp
        raise ProviderError("every LLM failed: " + " | ".join(errors))

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        errors = []
        for name, llm in self._chain():
            started = False
            try:
                for delta in llm.stream(system, messages, max_tokens):
                    started = True
                    yield delta
                return
            except RUNTIME_ERRORS as exc:
                if started:  # half an answer was already sent: don't splice in another model
                    raise
                errors.append(f"{name}: {exc}")
        raise ProviderError("every LLM failed: " + " | ".join(errors))
