"""
RAG CLI — the plain-RAG baseline for the GraphRAG comparison.

    python rag.py index                    # chunk + embed the corpus once
    python rag.py search "your query"      # retrieval only (shows scores)
    python rag.py ask "your question"      # retrieval + LLM answer

Corpus: the *-Stanza-out.txt files under --data. Retrieval: BM25 + embedding
cosine, min-max normalized, combined by averaging.
"""

import os
import argparse
import logging

from dotenv import load_dotenv
from openai import OpenAI

from karma_mini.rag import (
    Embedder, build_index, load_index, hybrid_search, answer,
)
from karma_mini.rag.embedder import DEFAULT_EMBED_MODEL

load_dotenv()

API_KEY = os.getenv("KIT_API_KEY")
BASE_URL = os.getenv("KIT_BASE_URL")

DEFAULT_CHAT_MODEL = "kit.mistral-small-4-119b-a8b"


def main():
    parser = argparse.ArgumentParser(description="RAG over the NCG trial papers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="chunk + embed the corpus")
    p_index.add_argument("--data", default="data/ncg/trial-data")
    p_index.add_argument("--out", default="data/rag")
    p_index.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)

    p_search = sub.add_parser("search", help="hybrid retrieval only")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--index", default="data/rag")

    p_ask = sub.add_parser("ask", help="retrieval + LLM answer")
    p_ask.add_argument("query")
    p_ask.add_argument("-k", type=int, default=5)
    p_ask.add_argument("--index", default="data/rag")
    p_ask.add_argument("--model", default=DEFAULT_CHAT_MODEL,
                       help="chat model for generation (any id on the endpoint)")

    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    if not API_KEY or not BASE_URL:
        print("\n[ERROR] KIT_API_KEY or KIT_BASE_URL is not set!")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=args.timeout)

    if args.cmd == "index":
        idx = build_index(client, args.data, args.out, embed_model=args.embed_model)
        print(f"\nIndexed {idx.meta['n_chunks']} chunks "
              f"(dim {idx.meta['dim']}) -> {args.out}\n")
        return

    idx = load_index(args.index)
    embedder = Embedder(client, model=idx.meta["embed_model"])
    hits = hybrid_search(idx, embedder, args.query, k=args.k)

    if args.cmd == "search":
        print(f"\nTop {len(hits)} for: {args.query!r}\n")
        for rank, h in enumerate(hits, 1):
            c = h["chunk"]
            section = f" | {c['section']}" if c.get("section") else ""
            print(f"[{rank}] combined={h['score']:.3f}  "
                  f"(bm25={h['bm25']:.2f}, cosine={h['cosine']:.3f})")
            print(f"    {c['id']}{section}")
            text = c["text"]
            print(f"    {text[:220]}{'...' if len(text) > 220 else ''}\n")
        return

    print(f"\nQ: {args.query}\n")
    print(answer(client, args.model, args.query, hits))
    print("\nSources:")

    for h in hits:
        c = h["chunk"]
        print(f"  [{c['id']}] combined={h['score']:.3f} "
              f"(bm25={h['bm25']:.2f}, cosine={h['cosine']:.3f})")
    print()


if __name__ == "__main__":
    main()
