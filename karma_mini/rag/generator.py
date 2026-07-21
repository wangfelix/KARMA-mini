"""
Generation: answer the query from the retrieved excerpts.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about a collection of NLP research papers.

You are given numbered excerpts retrieved from the papers. Rules:
- Answer using only the information in the excerpts. Do not use outside knowledge.
- Cite every claim with the excerpt tag(s) it comes from, e.g. [machine-translation/0:15-18].
- If the excerpts do not contain the answer, say exactly that.
- Be concise: a short paragraph, or bullets when listing."""


def answer(client, model: str, query: str, hits: List[Dict],
           temperature: float = 0.1) -> str:
    """Ask the LLM to answer ``query`` from the retrieved ``hits``."""
    blocks = []
    for h in hits:
        c = h["chunk"]
        section = f" ({c['section']})" if c.get("section") else ""
        blocks.append(f"[{c['id']}]{section}\n{c['text']}")
    context = "\n\n".join(blocks)

    prompt = f"""EXCERPTS:
{context}

QUESTION: {query}

Answer from the excerpts only, with citations."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
