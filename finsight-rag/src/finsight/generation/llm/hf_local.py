"""
A small instruction model running inside this Python process (transformers, Apple GPU / MPS).

The always-available fallback of provider "auto": no server, no key, no cost. Qwen2.5-1.5B
answers simple extraction questions; it is much weaker than a 7B Ollama model or Claude on
calculations, so its eval numbers are a floor, not the project's best. Greedy decoding.
The prompt is cut from the LEFT of the context if it exceeds num_ctx (the question stays).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from functools import cached_property
from typing import Any

from finsight.config import LLMCfg
from finsight.generation.llm.base import LLM, LLMResponse


class HFLocalLLM(LLM):
    provider = "hf"

    def __init__(self, cfg: LLMCfg, device: str = "auto") -> None:
        self.cfg, self.model, self._device = cfg, cfg.model, device

    @cached_property
    def _loaded(self) -> tuple[Any, Any, str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from finsight.embedding.sentence_transformer import resolve_device

        device = resolve_device(self._device)
        tok = AutoTokenizer.from_pretrained(self.model)
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype)
        model = model.to(device).eval()  # type: ignore[arg-type]
        return tok, model, device

    def _inputs(self, system: str, messages: list[dict], max_tokens: int) -> Any:
        tok, _, device = self._loaded
        chat = [{"role": "system", "content": system}, *messages]
        ids = tok.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        budget = self.cfg.num_ctx - max_tokens
        if ids["input_ids"].shape[1] > budget:  # keep the end (question + instructions)
            ids = {k: v[:, -budget:] for k, v in ids.items()}
        return {k: v.to(device) for k, v in ids.items()}

    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        import torch

        tok, model, _ = self._loaded
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        inputs = self._inputs(system, messages, max_tokens)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id
            )
        n_in = inputs["input_ids"].shape[1]
        new = out[0, n_in:]
        return LLMResponse(
            text=tok.decode(new, skip_special_tokens=True).strip(),
            model=self.model,
            input_tokens=int(n_in),
            output_tokens=int(new.shape[0]),
            stop_reason="end_turn" if new.shape[0] < max_tokens else "max_tokens",
        )

    def stream(self, system: str, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        tok, model, _ = self._loaded
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = {
            **self._inputs(system, messages, max_tokens),
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "streamer": streamer,
            "pad_token_id": tok.eos_token_id,
        }
        threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
        yield from streamer
