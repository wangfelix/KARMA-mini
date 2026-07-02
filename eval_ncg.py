#!/usr/bin/env python
"""
Score NCG predictions against the gold trial data using the OFFICIAL scorer.

This reuses the exact matching logic from ``scoring/evaluation.py`` (the
SemEval-2021 Task 11 scoring program) — the per-line string matching in
``evaluate()`` — but discovers whichever papers are present under the
predictions root, so it works whether you ran the pipeline on one paper or all
of them.

Usage:
    python eval_ncg.py
    python eval_ncg.py --gold data/ncg/trial-data --pred data/ncg/predictions

Setup (one-time): clone the official scorer into ./scoring (gitignored):
    git clone https://github.com/ncg-task/scoring-program.git scoring
"""

import os
import sys
import types
import argparse
import importlib.util
from os import walk

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER_PATH = os.path.join(HERE, "scoring", "evaluation.py")

# Info-unit keys reported individually, mirroring the official scorer's order.
IU_KEYS = [
    "research-problem", "approach", "model", "code", "dataset",
    "experimental-setup", "hyperparameters", "baselines", "results",
    "tasks", "experiments", "ablation-analysis",
]


def load_official_scorer():
    """Import scoring/evaluation.py, stubbing its unused scipy/numpy imports."""
    if not os.path.exists(SCORER_PATH):
        sys.exit(
            "Official scorer not found at scoring/evaluation.py.\n"
            "Clone it (gitignored):\n"
            "  git clone https://github.com/ncg-task/scoring-program.git scoring"
        )

    # evaluation.py imports scipy.stats and numpy at module top but never uses
    # them; stub them out so the import works without those heavy deps.
    if "scipy" not in sys.modules:
        try:
            import scipy.stats  # noqa: F401
        except Exception:
            scipy = types.ModuleType("scipy")
            stats = types.ModuleType("scipy.stats")
            scipy.stats = stats
            sys.modules["scipy"] = scipy
            sys.modules["scipy.stats"] = stats
    if "numpy" not in sys.modules:
        try:
            import numpy  # noqa: F401
        except Exception:
            sys.modules["numpy"] = types.ModuleType("numpy")

    spec = importlib.util.spec_from_file_location("ncg_evaluation", SCORER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_papers(pred_root):
    """Relative paper paths (<task>/<n>) present under the predictions root."""
    papers = []
    for dirpath, dirnames, filenames in walk(pred_root):
        if "sentences.txt" in filenames and "triples" in dirnames:
            papers.append(os.path.relpath(dirpath, pred_root))
    return sorted(papers)


def f1(p, r):
    return 0.0 if (p + r) == 0 else (2.0 * p * r) / (p + r)


def main():
    ap = argparse.ArgumentParser(description="Score NCG predictions (official scorer)")
    ap.add_argument("--gold", default="data/ncg/trial-data", help="gold trial-data root")
    ap.add_argument("--pred", default="data/ncg/predictions", help="predictions root")
    args = ap.parse_args()

    ev = load_official_scorer()
    evaluate = ev.evaluate
    compute_total = ev.compute_total
    prf = ev.compute_recall_precision_fscore

    papers = discover_papers(args.pred)
    if not papers:
        sys.exit(f"No prediction papers found under {args.pred}. Run main.py first.")

    # Aggregate counters (same structure as the official main()).
    agg = {k: {"tp": 0, "fp": 0, "fn": 0, "total": 0}
           for k in ("sent", "phr", "iu", "trip")}
    iu_total, iu_tp, iu_fp = {}, {}, {}

    scored = 0
    for rel in papers:
        gold_dir = os.path.join(args.gold, rel)
        pred_dir = os.path.join(args.pred, rel)
        if not os.path.isdir(gold_dir):
            print(f"[skip] no gold for {rel}")
            continue
        scored += 1

        # sentences + phrases (entities) — exact per-line match
        for field, fname in (("sent", "sentences.txt"), ("phr", "entities.txt")):
            g = os.path.join(gold_dir, fname)
            p = os.path.join(pred_dir, fname)
            if not (os.path.exists(g) and os.path.exists(p)):
                continue
            total, tp, fp, fn = evaluate(g, p)
            agg[field]["total"] += total
            agg[field]["tp"] += tp
            agg[field]["fp"] += fp
            agg[field]["fn"] += fn

        # triples + info-units
        gold_tri = os.path.join(gold_dir, "triples")
        pred_tri = os.path.join(pred_dir, "triples")
        gold_files = set(os.listdir(gold_tri)) if os.path.isdir(gold_tri) else set()
        pred_files = set(os.listdir(pred_tri)) if os.path.isdir(pred_tri) else set()

        for f in gold_files:
            key = f.replace(".txt", "")
            gpath = os.path.join(gold_tri, f)
            ppath = os.path.join(pred_tri, f)
            if os.path.exists(ppath):
                agg["iu"]["total"] += 1
                agg["iu"]["tp"] += 1
                total, tp, fp, fn = evaluate(gpath, ppath)
                agg["trip"]["total"] += total
                agg["trip"]["tp"] += tp
                agg["trip"]["fp"] += fp
                agg["trip"]["fn"] += fn
                iu_total[key] = iu_total.get(key, 0) + total
                iu_tp[key] = iu_tp.get(key, 0) + tp
                iu_fp[key] = iu_fp.get(key, 0) + fp
            else:
                agg["iu"]["total"] += 1
                agg["iu"]["fn"] += 1
                total = compute_total(gpath)
                agg["trip"]["total"] += total
                agg["trip"]["fn"] += total
                iu_total[key] = iu_total.get(key, 0) + total

        # predicted IU files with no gold counterpart -> false positives
        for f in pred_files - gold_files:
            key = f.replace(".txt", "")
            agg["iu"]["fp"] += 1
            fp_temp = compute_total(os.path.join(pred_tri, f))
            agg["trip"]["fp"] += fp_temp
            iu_fp[key] = iu_fp.get(key, 0) + fp_temp

    # --- report -----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f" NCG SCORES  ({scored} paper(s)) ".center(60, "="))
    print("=" * 60)
    header = f"{'metric':<16}{'P':>10}{'R':>10}{'F1':>10}"
    print(header)
    print("-" * 60)

    labels = {"sent": "SENTENCES", "phr": "PHRASES", "iu": "INFO_UNITS", "trip": "TRIPLES"}
    f1s = []
    for key in ("sent", "phr", "iu", "trip"):
        c = agg[key]
        r, p, fscore = prf(c["tp"], c["fp"], c["total"]) if c["total"] else (0.0, 0.0, 0.0)
        f1s.append(fscore)
        print(f"{labels[key]:<16}{p:>10.4f}{r:>10.4f}{fscore:>10.4f}")

    print("-" * 60)
    print(f"{'AVERAGE_F1':<16}{'':>10}{'':>10}{sum(f1s) / 4:>10.4f}")

    # Per-info-unit triple F1 (only those that appear in gold/pred)
    keys_present = [k for k in IU_KEYS if k in iu_total]
    if keys_present:
        print("\nPer-info-unit triple F1:")
        for key in keys_present:
            tp = iu_tp.get(key, 0)
            fp = iu_fp.get(key, 0)
            total = iu_total.get(key, 0)
            r, p, fscore = prf(tp, fp, total) if total else (0.0, 0.0, 0.0)
            print(f"  {key:<20} P={p:.3f} R={r:.3f} F1={fscore:.3f}")
    print()


if __name__ == "__main__":
    main()
