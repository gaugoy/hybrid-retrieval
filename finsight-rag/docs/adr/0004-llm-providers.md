# ADR 0004: LLM presets with "auto" resolution, free tiers only by default, fallback chain

- **Status:** accepted (P6, P10) · **Date:** 2026-11-24

## Context
It must run with zero spend and no keys, yet use Claude or free hosted tiers when available. Free tiers have
strict and changing limits, and hosted models fail (outages, 429, quota).

## Decision
- Models are **presets** in `configs/llms.yaml`, chosen with `LLM=<name>`. `auto` resolves at startup, in order:
  Ollama → Claude → Gemini → Mistral → Groq → local Hugging Face Qwen2.5-1.5B (always available).
- **Paid-only providers are refused** unless `allow_paid: true`. Client-side rate limits (rpm/tpm/rpd/rpmonth)
  are persisted across runs. A per-day/month quota 429 stops instead of retrying.
- Claude through the official SDK (`effort`, server-side refusal fallback; a refusal becomes an abstention).
- **P10 fallback chain:** `llm.fallbacks` (e.g. claude-opus → ollama-qwen7b → hf-qwen1.5b). Only runtime provider errors
  trigger it; config errors raise. The answer is flagged `degraded: ["llm_fallback"]` and never cached.
- The judge (P8) should be from a different model family than the generator: a same-model local judge rated 90/90 "correct".

## Consequences
- ✅ `git clone && make ...` works offline at $0. Switching provider is an env var.
- ⚠️ The default local model is small: answers are weaker (numeric match 0.31 on 30 Q) and requests are serialized on one GPU.
- ⚠️ Free-tier limits in the presets are conservative guesses dated 2026-11-24 and need occasional updates.
