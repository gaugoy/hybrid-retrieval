# FinSight RAG — agent instructions (keep this file short)

Independent project. Don't import from or modify the parent `hybrid-retrieval/` experiment.

## Start here (cheap → expensive)
1. `docs/CODEMAP.md`: where everything is, plus change recipes. Read it FIRST for any code change.
2. Then only the files the CODEMAP points to. Never read all of `src/`.
3. `docs/DEVELOPMENT_GUIDE.md` (~26k tokens): read ONLY the section you need (grep the heading), never the whole file.
4. Decisions: `docs/adr/`. Current state and open items: DEVELOPMENT_GUIDE §6.1 "Resume notes".

## Hard rules
- $0: local models or free tiers only. Never enable paid providers (`allow_paid` stays false).
- No Docker. The app runs as independent processes: `ollama serve` (optional), `make serve`, `make ui`.
- Run everything via `uv run` or Makefile targets. Python 3.12. No LangChain/LlamaIndex. Library code goes in `src/finsight/`.
- No models or data loaded at import time. Never commit `.env`, `data/` or `indexes/`.
- Close the loop: every change gets a test. Run `make lint` and `make test`, and show the summary line.
- Numbers in docs must come from a `results/` report.
- Warn me BEFORE anything that takes > 5 min, downloads > 1 GB, or uses a lot of CPU/RAM.
- Embedded Qdrant allows one process. If it's locked, ask me (notebook kernels may be open); never kill them.

## Conventions
- Learning scripts `scripts/NN_*.py` use `# %%` cells, `script_args()` and `finsight.notebook.show()`. Run `make notebooks` after changing one.
- After a change, update DEVELOPMENT_GUIDE §6 (progress) and §7 (decisions) briefly. Update `docs/CODEMAP.md` if files or entry points changed.
- New measured numbers: offer to update `docs/evaluation.md` and the 📌 lines in `docs/INTERVIEW_GUIDE.md`.

## Git
- One branch per task. Commit author: Gaurav Goyal. Commit/push only when I ask. I merge PRs in the GitHub UI.

## Output style
- Short answers. Show evidence (test summary, file:line, results path). Say what you did NOT verify.
- End of task: update §6.1 resume notes (≤ 20 lines), then stop.
