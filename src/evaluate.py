import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.recsys import ModelHyperParams, load_movielens_100k, train_test_split_interactions, train_engine


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def rmse(engine, test_df: pd.DataFrame, strategy: str) -> float:
    errors: List[float] = []
    for row in test_df.itertuples(index=False):
        pred = engine.predict_rating(int(row.user_id), int(row.item_id), strategy=strategy)
        errors.append((pred - float(row.rating)) ** 2)
    return float(np.sqrt(np.mean(errors))) if errors else float("nan")


def precision_recall_at_k(
    engine,
    test_df: pd.DataFrame,
    strategy: str,
    k: int,
    relevance_threshold: float,
) -> Tuple[float, float]:
    # Build relevant sets per user
    grouped = test_df.groupby("user_id")
    precisions: List[float] = []
    recalls: List[float] = []

    for user_id, u_df in grouped:
        relevant_items = set(u_df[u_df["rating"] >= relevance_threshold]["item_id"].astype(int).tolist())
        if not relevant_items:
            continue

        recs = engine.recommend(int(user_id), k=k, strategy=strategy, exclude_seen=True)
        rec_item_ids = [int(r["item_id"]) for r in recs]

        hits = len(relevant_items.intersection(rec_item_ids))
        precisions.append(hits / float(k))
        recalls.append(hits / float(len(relevant_items)))

    if not precisions:
        return float("nan"), float("nan")
    return float(np.mean(precisions)), float(np.mean(recalls))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommendation strategies.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="reports/eval.json")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-user-ratings", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(os.path.dirname(args.output) or ".")

    hyperparams = ModelHyperParams()
    engine, report, train_df, test_df = train_engine(
        data_dir=args.data_dir,
        hyperparams=hyperparams,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_user_ratings=args.min_user_ratings,
    )

    strategies = ["user_cf", "item_cf", "content", "svd", "hybrid"]
    results: Dict[str, dict] = {"meta": report, "metrics": {}}

    for strat in strategies:
        logging.getLogger(__name__).info("Evaluating %s...", strat)
        rmse_val = rmse(engine, test_df, strategy=strat)
        precision_val, recall_val = precision_recall_at_k(
            engine,
            test_df,
            strategy=strat,
            k=args.k,
            relevance_threshold=hyperparams.relevance_rating_threshold,
        )
        results["metrics"][strat] = {
            "rmse": rmse_val,
            f"precision@{args.k}": precision_val,
            f"recall@{args.k}": recall_val,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.getLogger(__name__).info("Saved evaluation to %s", args.output)


if __name__ == "__main__":
    main()

