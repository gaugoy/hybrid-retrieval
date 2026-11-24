"""
Build the configured LLM. Provider "auto" picks the first preset that can run right now.

    auto_order (default): ollama-qwen7b, claude-opus, gemini-flash, mistral-small, groq-llama,
                          hf-qwen1.5b
    ollama   -> the server answers and the model is pulled
    online   -> its API key is set (and, for paid-only providers, allow_paid is true)
    hf       -> always (downloads the model once)
The choice is logged and visible in Answer.provider / Answer.model.
"""

from __future__ import annotations

from pathlib import Path

from finsight.config import LLMCfg, Secrets, load_llm_presets, resolve_llm_preset
from finsight.errors import ConfigError, MissingAPIKeyError, PaidProviderBlockedError
from finsight.generation.llm.base import LLM
from finsight.observability.logging import get_logger

log = get_logger(__name__)


def can_run(cfg: LLMCfg, secrets: Secrets | None) -> tuple[bool, str]:
    """(ok, reason) without building anything heavy."""
    from finsight.generation.llm.remote_llm import PAID_ONLY

    if cfg.provider in ("fake", "hf"):
        return True, "local"
    if cfg.provider == "ollama":
        from finsight.generation.llm.ollama import ollama_ready

        ok = ollama_ready(cfg.base_url or cfg.ollama_base_url, cfg.model)
        return ok, "ready" if ok else "Ollama not running or model not pulled"
    if cfg.provider in PAID_ONLY and not cfg.allow_paid:
        return False, "paid only (llm.allow_paid is false)"
    key = secrets.for_provider(cfg.provider) if secrets else None
    return (key is not None), ("key set" if key else f"no {cfg.provider.upper()}_API_KEY")


def resolve_auto(cfg: LLMCfg, root: Path, secrets: Secrets | None) -> LLMCfg:
    presets = load_llm_presets(root)
    tried = []
    for name in cfg.auto_order:
        if name not in presets:
            continue
        candidate = LLMCfg.model_validate(
            resolve_llm_preset(
                root,
                {
                    "preset": name,
                    "allow_paid": cfg.allow_paid,
                    "ollama_base_url": cfg.ollama_base_url,
                },
            )
        )
        ok, why = can_run(candidate, secrets)
        if ok:
            log.info("llm auto-selected", extra={"preset": name, "model": candidate.model})
            return candidate
        tried.append(f"{name}: {why}")
    raise ConfigError("llm.provider=auto found nothing to run: " + "; ".join(tried))


def build_llm(
    cfg: LLMCfg,
    *,
    root: Path,
    secrets: Secrets | None = None,
    usage_dir: Path | None = None,
    device: str = "auto",
) -> LLM:
    if cfg.provider == "auto":
        cfg = resolve_auto(cfg, root, secrets)
    if cfg.provider == "fake":
        from finsight.generation.llm.fake import FakeLLM

        return FakeLLM(model=cfg.model)
    if cfg.provider == "ollama":
        from finsight.generation.llm.ollama import OllamaLLM

        return OllamaLLM(cfg)
    if cfg.provider == "hf":
        from finsight.generation.llm.hf_local import HFLocalLLM

        return HFLocalLLM(cfg, device=device)

    from finsight.generation.llm.remote_llm import KEY_URLS, PAID_ONLY

    if cfg.provider in PAID_ONLY and not cfg.allow_paid:
        raise PaidProviderBlockedError(
            f"{cfg.provider} has no free tier; set llm.allow_paid: true to spend money on purpose. "
            "Free options: LLM=gemini-flash, mistral-small, groq-llama, hf-qwen1.5b, ollama-qwen7b."
        )
    key = secrets.for_provider(cfg.provider) if secrets else None
    if key is None:
        url = KEY_URLS.get(cfg.provider, "https://console.anthropic.com/")
        raise MissingAPIKeyError(
            f"LLM {cfg.preset or cfg.model!r} needs {cfg.provider.upper()}_API_KEY in .env ({url})."
        )
    if cfg.provider == "anthropic":
        from finsight.generation.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(cfg, key.get_secret_value())
    from finsight.generation.llm.remote_llm import GeminiLLM, OpenAIChatLLM

    cls = GeminiLLM if cfg.provider == "gemini" else OpenAIChatLLM
    return cls(cfg, key.get_secret_value(), usage_dir=usage_dir)
