# FinSight RAG — agent instructions

This folder is an independent project. Don't import from, or modify, the parent `hybrid-retrieval/` experiment.

- **Read `docs/DEVELOPMENT_GUIDE.md` before writing any code.** It is the source of truth: the rules in §0, the verified data facts in §1, the contracts in §4, and the phases in §5.
- Build **one phase at a time**, in order. A phase is finished only when every box in its "Done when" list is ticked.
- Each phase ships a hand-run learning script, `scripts/NN_*.py`, that prints a "what just happened" summary. Library code goes in `src/finsight/`.
- Run everything with `uv run ...`. Python 3.12. Local-first: no API key may be required for any phase.
- No LangChain or LlamaIndex. No module may load models or data at import time. Never commit `.env`, `data/` or `indexes/`.
- At the end of each phase, update `docs/DEVELOPMENT_GUIDE.md` §6 (progress) and §7 (decisions), then propose a commit message. Commit or push only when the user asks.
- The concept explanations for interviews live in `docs/INTERVIEW_GUIDE.md`. When a phase produces new measured numbers, offer to fill its 📌 placeholders.
