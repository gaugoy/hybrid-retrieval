# CODEMAP — read this instead of the whole repo

Flow: data → ingestion → chunking → embedding/indexing → retrieval → generation → pipelines → api/ui.
evaluation/ measures each stage. container.py wires everything from config.py `Settings`;
the CLI, scripts/, evaluation and the API all use the same Container.

| Area | Files (src/finsight/…) | Entry points | Tests |
|---|---|---|---|
| Config | config.py, configs/*.yaml (base, local, cloud, multihop, llms, embedders, experiments/) | load_settings(), Settings.fingerprint() | unit/test_config.py |
| Wiring | container.py | Container.pipeline / .retrieval / .llm / .answer_cache | integration/* |
| Data | data/financebench.py, multihop.py, datasets.py (dispatch by settings.dataset), download.py | eval_records(), document_meta(), pages_df() | unit/test_multihop_loader.py |
| Ingestion | ingestion/pdf_parser.py, cleaner.py, quality.py, pipeline.py | ingest_all() | integration/test_ingest_pipeline.py |
| Chunking | chunking/fixed, recursive, page_aware, parent_child, contextual, registry, tokenizer, pipeline | make_chunker(), chunk_corpus() | integration/test_chunk_pipeline.py |
| Embedding | embedding/sentence_transformer.py, remote.py, catalog.py, cache.py, fake.py | make_embedder() (indexing/pipeline.py), embed_with_cache(), cached_query_vectors() | unit/test_caching_repro.py |
| Indexing | indexing/qdrant_store.py, bm25_store.py, manifest.py, pipeline.py | index_corpus(), collection_name() | integration/test_index_roundtrip.py |
| Retrieval | retrieval/dense, sparse, hybrid (RRF), filters, reranker (+ScoreCache), query_transform, service | RetrievalService.retrieve() | integration/test_retrieval_service.py, test_regression_gate.py |
| Generation | generation/context.py, citations.py, prompt_loader.py, prompts/*.md, llm/{factory,fallback,hf_local,ollama,remote_llm,anthropic_llm,fake}.py | build_llm(), FallbackLLM, parse_citations() | unit/test_fallback_llm.py, integration/test_pipeline_fake_llm.py |
| Pipelines | pipelines/base.py, naive, hybrid_rerank, corrective, agentic, registry | RAGPipeline.answer() | integration/test_advanced_rag.py |
| Evaluation | evaluation/runner.py, retrieval_metrics.py, e2e.py, generation_metrics.py, judge.py, report.py, pipeline_sample.py, embedder_compare.py | run_retrieval_eval(), run_end_to_end_eval() | unit/test_judge.py, unit/test_generation_metrics.py |
| Serving | api/main.py (create_app), routes.py, models.py, middleware.py, deps.py; cache.py; ui/streamlit_app.py | /v1/query, /v1/query/stream | integration/test_api.py, test_degradation.py, unit/test_streamlit_ui.py |
| Providers | providers/http.py (retry/backoff), providers/ratelimit.py (rpm/tpm/rpd) | post_json(), RateLimiter | — |
| Ops | observability/logging.py, timing.py, cost.py; diagnostics.py; cli.py | finsight <command> | unit/test_diagnostics_and_cli.py |

## Change recipes (touch only these)
- New LLM provider → generation/llm/<new>.py (subclass LLM: generate, stream) + llm/factory.py (build_llm branch)
  + config.py (LLMProvider literal, API_KEY_ENV) + configs/llms.yaml preset + a unit test with a fake HTTP client.
- New model of an existing provider → configs/llms.yaml or configs/embedders.yaml ONLY (then `make index` for embedders).
- New chunker → chunking/<new>.py (subclass Chunker) + chunking/registry.py + config strategy literal.
- New retrieval step → retrieval/<new>.py + retrieval/service.py + a variant in configs/experiments/*.yaml.
- New pipeline → pipelines/<new>.py + pipelines/registry.py + api/models.py `Pipeline` literal + config `pipeline` literal.
- New API endpoint → api/routes.py + api/models.py + integration/test_api.py.
- UI change → ui/streamlit_app.py (it only calls the API) + unit/test_streamlit_ui.py.
- New dataset → data/<new>.py + data/datasets.py dispatch + configs/<dataset>.yaml.
- New config field → config.py (strict models: unknown YAML keys fail) + configs/base.yaml;
  add the section to FINGERPRINT_FIELDS if it changes answers.
- New metric → evaluation/retrieval_metrics.py or generation_metrics.py + evaluation/report.py.

## Gotchas
- Embedded Qdrant = one process (notebook kernels lock indexes/qdrant).
- Local models are serialized by locks (hf_local._GPU_LOCK, sentence_transformer._ENCODE_LOCK).
- Caches live in indexes/embed_cache/ (query vectors, rerank scores, answers). Keys include model/text/config fingerprint.
- Tests use fakes (FakeEmbedder, FakeLLM) and the `tiny` fixture: no downloads, ~8 s.
