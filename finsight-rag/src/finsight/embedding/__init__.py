"""P4 embeddings: a bi-encoder turns text into L2-normalized vectors (Embedder), with a
SQLite cache so unchanged text is never embedded twice, and a FakeEmbedder for tests."""
