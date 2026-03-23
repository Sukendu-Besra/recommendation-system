import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.recsys import load_movielens_100k


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    data_dir = os.environ.get("DATA_DIR", "data")
    reports_dir = os.environ.get("REPORTS_DIR", "reports")
    ensure_dir(reports_dir)

    ratings_df, items_df = load_movielens_100k(data_dir)

    # 1) Rating distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(ratings_df["rating"], bins=20, kde=False)
    plt.title("MovieLens 100K: Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "eda_rating_distribution.png"), dpi=160)
    plt.close()

    # 2) User activity
    user_activity = ratings_df.groupby("user_id")["item_id"].count().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    sns.histplot(user_activity, bins=30)
    plt.title("User Activity (number of ratings per user)")
    plt.xlabel("Ratings per user")
    plt.ylabel("Number of users")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "eda_user_activity.png"), dpi=160)
    plt.close()

    # 3) Item popularity
    item_popularity = ratings_df.groupby("item_id")["user_id"].count().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    sns.histplot(item_popularity, bins=30)
    plt.title("Item Popularity (number of ratings per item)")
    plt.xlabel("Ratings per item")
    plt.ylabel("Number of items")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "eda_item_popularity.png"), dpi=160)
    plt.close()

    # 4) Sparsity estimate
    n_users = ratings_df["user_id"].nunique()
    n_items = ratings_df["item_id"].nunique()
    n_ratings = len(ratings_df)
    sparsity = 1.0 - (n_ratings / float(n_users * n_items))
    with open(os.path.join(reports_dir, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"n_users={n_users}\n")
        f.write(f"n_items={n_items}\n")
        f.write(f"n_ratings={n_ratings}\n")
        f.write(f"sparsity≈{sparsity:.4f}\n")

    # 5) Genre popularity
    genre_cols = [c for c in items_df.columns if c not in {"item_id", "title", "release_date", "video_release_date", "imdb_url"}]
    genre_counts = items_df[genre_cols].sum(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.xticks(rotation=60, ha="right")
    plt.title("MovieLens: Genre presence (count of movies per genre)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "eda_genre_presence.png"), dpi=160)
    plt.close()

    print("EDA finished. Plots saved to:", reports_dir)


if __name__ == "__main__":
    main()

