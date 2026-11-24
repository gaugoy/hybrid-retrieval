Reply with ONE JSON object and nothing else, one of:
{"tool": "search", "args": {"query": "...", "company": "optional", "year": 2018}}
{"tool": "read", "args": {"chunk_id": "..."}}
{"tool": "answer", "args": {"text": "...", "citations": ["chunk_id", ...]}}

Question: $question

Steps so far:
$history
