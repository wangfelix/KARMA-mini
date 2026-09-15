"""Development-selected defaults for the final RAG comparison.

These values were selected on the 20-question development split under the
open-corpus benchmark (generator v3). The selection rule was fixed before the
results were seen: highest fully-correct rate, then fewest incorrect answers,
then better reference-item recall, then smaller ``k``. BM25 at ``k=15`` won on
the first two criteria outright.

The held-out test split must not be used to revise these values.
"""

FINAL_ANSWER_MODEL = "kit.mistral-small-4-119b-a8b"
FINAL_JUDGE_MODEL = "kit.gpt-oss-120b"
FINAL_RETRIEVAL_MODE = "bm25"
FINAL_DENSE_WEIGHT = 0.0
FINAL_TOP_K = 15
