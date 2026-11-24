"""
AgenticRAGPipeline: the LLM drives retrieval with tools, at most agent.max_steps steps.

  search(query, company?, year?) -> top-5 chunk ids + previews (hybrid + rerank, filtered)
  read(chunk_id)                 -> the chunk's full text
  answer(text, citations)        -> finish; citations = chunk ids it read

Two protocols:
  Claude      native tool use: tools=[...], tool_choice auto (Claude Opus 5.5 rejects forced
              tool_choice), loop while stop_reason == "tool_use", return tool_result blocks
  everything  a JSON-action protocol: one {"tool": ..., "args": ...} object per step, validated
  else        with pydantic; invalid twice in a row -> fall back to plain hybrid_rerank
The step cap stops runaway loops; the trace (thought/tool/args/result preview) is kept in
`self.trace` for the step script and the UI.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel

from finsight.generation import prompt_loader
from finsight.generation.jsonout import ask_json
from finsight.pipelines.base import Retrieved
from finsight.pipelines.hybrid_rerank import HybridRerankPipeline
from finsight.pipelines.registry import register_pipeline
from finsight.schemas import Answer, Citation, QueryFilters, RetrievedChunk


class Action(BaseModel):
    tool: Literal["search", "read", "answer"]
    args: dict[str, Any] = {}


TOOLS = [
    {
        "name": "search",
        "description": "Search SEC filing chunks. Use filing vocabulary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "company": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read",
        "description": "Read the full text of a chunk returned by search.",
        "input_schema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "answer",
        "description": "Finish with the final answer and the chunk ids you read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "citations"],
            "additionalProperties": False,
        },
    },
]


@register_pipeline("agentic")
class AgenticRAGPipeline(HybridRerankPipeline):
    # ------------------------------------------------------------------ tools
    def _search(self, args: dict[str, Any], seen: dict[str, RetrievedChunk]) -> str:
        flt = QueryFilters(
            companies=[args["company"]] if args.get("company") else [],
            doc_periods=[int(args["year"])] if args.get("year") else [],
        )
        has = flt.companies or flt.doc_periods
        res = self.retrieval.retrieve(
            str(args.get("query", "")), filters=flt if has else None, top_k=5
        )
        for rc in res.final:
            seen[rc.chunk.chunk_id] = rc
        return (
            "\n".join(
                f"{rc.chunk.chunk_id} | {rc.chunk.doc_name} p{rc.chunk.page_start + 1} | "
                f"{' '.join(rc.chunk.text.split())[:160]}"
                for rc in res.final
            )
            or "(no results)"
        )

    def _read(self, args: dict[str, Any], seen: dict[str, RetrievedChunk], read: list[str]) -> str:
        cid = str(args.get("chunk_id", ""))
        if cid not in seen:
            return f"unknown chunk_id {cid!r}: use an id returned by search"
        if cid not in read:
            read.append(cid)
        return seen[cid].chunk.text[:3000]

    # ------------------------------------------------------------------ loops
    def _loop_json(self, question: str) -> tuple[str, list[str], dict[str, RetrievedChunk], int]:
        cfg, g = self.settings.agent, self.settings.generation
        system = prompt_loader.render(
            "agent_system", max_steps=cfg.max_steps, abstain_phrase=g.abstain_phrase
        )
        seen: dict[str, RetrievedChunk] = {}
        read: list[str] = []
        history: list[str] = []
        calls, failures = 0, 0
        for step in range(cfg.max_steps):
            prompt = prompt_loader.render(
                "agent_json", question=question, history="\n".join(history) or "(none)"
            )
            act, n = ask_json(self.llm, prompt, Action, system=system, max_tokens=400, retries=0)
            calls += n
            if act is None:
                failures += 1
                if failures >= 2:
                    raise ValueError("agent produced invalid JSON twice")
                history.append(f"{step + 1}. (invalid JSON; reply with one JSON object)")
                continue
            if act.tool == "answer":
                cites = [c for c in act.args.get("citations", []) if c in read]
                self.trace.append({"step": step + 1, "tool": "answer", "args": act.args})
                return str(act.args.get("text", "")), cites, seen, calls
            result = (
                self._search(act.args, seen)
                if act.tool == "search"
                else self._read(act.args, seen, read)
            )
            self.trace.append(
                {"step": step + 1, "tool": act.tool, "args": act.args, "result": result[:200]}
            )
            history.append(f"{step + 1}. {act.tool}({json.dumps(act.args)}) ->\n{result[:1200]}")
        return g.abstain_phrase, [], seen, calls

    def _loop_claude(self, question: str) -> tuple[str, list[str], dict[str, RetrievedChunk], int]:
        cfg, g = self.settings.agent, self.settings.generation
        system = prompt_loader.render(
            "agent_system", max_steps=cfg.max_steps, abstain_phrase=g.abstain_phrase
        )
        llm: Any = self.llm
        seen: dict[str, RetrievedChunk] = {}
        read: list[str] = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        for step in range(cfg.max_steps):
            params = llm._params(system, messages, 4096)
            resp = llm._api().create(**params, tools=TOOLS, tool_choice={"type": "auto"})
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return text or g.abstain_phrase, read, seen, step + 1
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "answer":
                    cites = [c for c in block.input.get("citations", []) if c in read]
                    self.trace.append({"step": step + 1, "tool": "answer", "args": block.input})
                    return str(block.input.get("text", "")), cites, seen, step + 1
                out = (
                    self._search(block.input, seen)
                    if block.name == "search"
                    else self._read(block.input, seen, read)
                )
                self.trace.append(
                    {"step": step + 1, "tool": block.name, "args": block.input, "result": out[:200]}
                )
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            messages.append({"role": "user", "content": results})
        return g.abstain_phrase, [], seen, cfg.max_steps

    # ------------------------------------------------------------------ entry point
    def answer(
        self, question: str, filters: QueryFilters | None = None, request_id: str | None = None
    ) -> Answer:
        t_all = time.perf_counter()
        self.trace: list[dict[str, Any]] = []
        g = self.settings.generation
        try:
            loop = self._loop_claude if self.llm.provider == "anthropic" else self._loop_json
            text, cites, seen, calls = loop(question)
        except ValueError:  # the model can't follow the protocol: plain hybrid_rerank instead
            ans = super().answer(question, filters, request_id)
            ans.pipeline, ans.degraded = self.name, [*ans.degraded, "agent_fell_back"]
            return ans
        self.llm_calls = calls
        citations = [
            Citation(
                marker=i + 1,
                chunk_id=c,
                doc_name=seen[c].chunk.doc_name,
                page_idx=seen[c].chunk.page_start,
                snippet=seen[c].chunk.text[:300],
            )
            for i, c in enumerate(cites)
        ]
        abstained = text.strip().lower().startswith(g.abstain_phrase.lower().rstrip("."))
        warnings = [] if citations or abstained else ["uncited_answer"]
        return Answer(
            question=question,
            answer=text,
            citations=citations,
            abstained=abstained,
            retrieved=list(seen.values())[:20],
            pipeline=self.name,
            provider=self.llm.provider,
            model=self.llm.model,
            timings_ms={"total": round((time.perf_counter() - t_all) * 1000, 1)},
            usage={"llm_calls": calls},
            request_id=request_id or "",
            warnings=warnings,
        )

    def _retrieve(self, question: str, filters: QueryFilters | None) -> Retrieved:
        return super()._retrieve(question, filters)
