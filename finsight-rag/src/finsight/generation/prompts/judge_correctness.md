You grade answers to financial questions against a reference answer.
Question: $question
Reference answer: $gold
Candidate answer: $prediction

Grade the candidate:
- "correct": same facts/numbers as the reference (rounding and units may differ; extra correct detail is fine)
- "partially_correct": some key facts right, others missing or wrong
- "incorrect": wrong, contradicts the reference, or says the information was not found
Return JSON only: {"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one sentence>"}
