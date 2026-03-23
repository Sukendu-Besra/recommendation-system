import json
import logging
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


MOVIELENS_100K_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass(frozen=True)
class ModelHyperParams:
    # Neighborhood sizes for cosine-similarity CF
    user_cf_k: int = 50
    item_cf_k: int = 50
    content_k: int = 50

    # Hybrid weights
    w_cf: float = 0.6
    w_content: float = 0.4

    # SVD settings
    svd_components: int = 50
    svd_random_state: int = 42

    # Ranking relevance threshold
    relevance_rating_threshold: float = 4.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download(url: str, dest_path: str) -> None:
    logger.info("Downloading dataset: %s", url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def download_movielens_100k(data_dir: str) -> str:
    """
    Downloads MovieLens 100K and returns the extracted folder path.
    """
    ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_root = data_dir
    extracted_folder = os.path.join(extract_root, "ml-100k")

    if os.path.isdir(extracted_folder):
        return extracted_folder

    if not os.path.exists(zip_path):
        _download(MOVIELENS_100K_ZIP_URL, zip_path)

    logger.info("Extracting MovieLens 100K to %s", extract_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    if not os.path.isdir(extracted_folder):
        raise RuntimeError("MovieLens extraction failed: missing ml-100k folder")
    return extracted_folder


def load_movielens_100k(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      ratings_df: columns [user_id, item_id, rating, timestamp]
      items_df: columns [item_id, title, genres...]
    """
    root = download_movielens_100k(data_dir)
    ratings_path = os.path.join(root, "u.data")
    items_path = os.path.join(root, "u.item")

    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    # u.item format:
    # movie id | movie title | release date | video release date | IMDb URL | genres...
    item_cols = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ]
    genre_names = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    item_cols.extend(genre_names)

    items_df = pd.read_csv(
        items_path,
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python",
    )

    # Basic cleaning
    ratings_df = ratings_df.dropna(subset=["user_id", "item_id", "rating"])
    items_df = items_df.dropna(subset=["item_id", "title"])
    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    ratings_df["item_id"] = ratings_df["item_id"].astype(int)

    for g in genre_names:
        items_df[g] = pd.to_numeric(items_df[g], errors="coerce").fillna(0.0).astype(float)

    return ratings_df, items_df


def train_test_split_interactions(
    ratings_df: pd.DataFrame, test_fraction: float = 0.2, min_user_ratings: int = 10, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple split:
      - Keep only users with at least `min_user_ratings`
      - Sample test interactions per-user
    """
    rng = np.random.default_rng(seed)
    user_counts = ratings_df["user_id"].value_counts()
    keep_users = user_counts[user_counts >= min_user_ratings].index
    df = ratings_df[ratings_df["user_id"].isin(keep_users)].copy()

    train_parts = []
    test_parts = []
    for user_id, u_df in df.groupby("user_id"):
        idx = u_df.index.to_numpy()
        n = len(idx)
        n_test = max(1, int(round(test_fraction * n)))
        test_idx = rng.choice(idx, size=n_test, replace=False)
        mask = np.isin(idx, test_idx)
        train_parts.append(u_df.loc[idx[~mask]])
        test_parts.append(u_df.loc[test_idx])
    train_df = pd.concat(train_parts, axis=0)
    test_df = pd.concat(test_parts, axis=0)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_mappings(train_df: pd.DataFrame, items_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Returns mappings:
      user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id
    """
    user_ids = sorted(train_df["user_id"].unique().tolist())
    item_ids = sorted(train_df["item_id"].unique().tolist())

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}

    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}
    return user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id


def build_interaction_matrix(
    train_df: pd.DataFrame,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    n_users: int,
    n_items: int,
) -> sparse.csr_matrix:
    rows = train_df["user_id"].map(user_id_to_idx).to_numpy()
    cols = train_df["item_id"].map(item_id_to_idx).to_numpy()
    vals = train_df["rating"].to_numpy(dtype=float)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return mat


def build_user_ratings(
    train_df: pd.DataFrame,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
) -> Dict[int, Dict[int, float]]:
    """
    user_ratings[user_idx][item_idx] = rating
    """
    user_ratings: Dict[int, Dict[int, float]] = {}
    for row in train_df.itertuples(index=False):
        u = user_id_to_idx[int(row.user_id)]
        i = item_id_to_idx[int(row.item_id)]
        user_ratings.setdefault(u, {})[i] = float(row.rating)
    return user_ratings


def build_item_features(items_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    genre_cols = [c for c in items_df.columns if c not in {"item_id", "title", "release_date", "video_release_date", "imdb_url"}]
    feature_mat = items_df[genre_cols].to_numpy(dtype=float)
    return feature_mat, genre_cols


def _topk_from_neighbors(
    indices: np.ndarray, distances: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NearestNeighbors returns n_neighbors >= k. We keep the first k.
    For cosine distance, similarity = 1 - distance.
    """
    indices_k = indices[:, :k]
    d_k = distances[:, :k]
    sims_k = 1.0 - d_k
    return indices_k.astype(int), sims_k.astype(float)


def compute_cosine_neighbors(
    vectors: Union[sparse.csr_matrix, np.ndarray],
    k: int,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row vector, find top-k nearest neighbors by cosine similarity.
    Returns:
      neighbors: (n_entities, k) indices
      sims: (n_entities, k) similarity scores in [-1,1]
    """
    n_entities = vectors.shape[0]
    k_eff = min(k, n_entities - 1) if exclude_self else min(k, n_entities)
    if k_eff <= 0:
        # Degenerate case
        return np.zeros((n_entities, 0), dtype=int), np.zeros((n_entities, 0), dtype=float)

    nn = NearestNeighbors(n_neighbors=k_eff + (1 if exclude_self else 0), metric="cosine", algorithm="brute")
    nn.fit(vectors)

    distances, indices = nn.kneighbors(vectors, return_distance=True)
    if exclude_self:
        # First neighbor is usually self for duplicate vectors. Drop it.
        # To be safe, drop the first column if indices match row indices for most rows.
        if indices.shape[1] > 0:
            drop = 0
            if np.mean(indices[:, 0] == np.arange(n_entities)) > 0.8:
                drop = 1
            indices = indices[:, drop:]
            distances = distances[:, drop:]

    indices_top, sims_top = _topk_from_neighbors(indices, distances, k_eff)
    return indices_top, sims_top


def minmax_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)


class RealtimeOverlay:
    """
    In-memory simulation of new user-item interactions.
    This is intentionally not persisted (meant for demo / showcasing).
    """

    def __init__(self) -> None:
        self.user_extra_ratings: Dict[int, Dict[int, float]] = {}

    def add(self, user_idx: int, item_idx: int, rating: float) -> None:
        self.user_extra_ratings.setdefault(user_idx, {})[item_idx] = float(rating)

    def get_user_items(self, user_idx: int) -> Dict[int, float]:
        return self.user_extra_ratings.get(user_idx, {})


class RecSysEngine:
    def __init__(
        self,
        *,
        hyperparams: ModelHyperParams,
        user_id_to_idx: Dict[int, int],
        idx_to_user_id: Dict[int, int],
        item_id_to_idx: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        item_titles: Dict[int, str],
        train_matrix: sparse.csr_matrix,
        user_ratings: Dict[int, Dict[int, float]],
        user_cf_neighbors: np.ndarray,
        user_cf_sims: np.ndarray,
        item_cf_neighbors: np.ndarray,
        item_cf_sims: np.ndarray,
        content_neighbors: np.ndarray,
        content_sims: np.ndarray,
        svd_user_factors: np.ndarray,
        svd_item_factors: np.ndarray,
        popularity: np.ndarray,
        global_mean_rating: float,
    ) -> None:
        self.hyperparams = hyperparams
        self.user_id_to_idx = user_id_to_idx
        self.idx_to_user_id = idx_to_user_id
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = idx_to_item_id
        self.item_titles = item_titles

        self.train_matrix = train_matrix
        self.user_ratings = user_ratings

        self.user_cf_neighbors = user_cf_neighbors
        self.user_cf_sims = user_cf_sims
        self.item_cf_neighbors = item_cf_neighbors
        self.item_cf_sims = item_cf_sims
        self.content_neighbors = content_neighbors
        self.content_sims = content_sims

        self.svd_user_factors = svd_user_factors
        self.svd_item_factors = svd_item_factors

        self.popularity = popularity
        self.global_mean_rating = float(global_mean_rating)

        self.overlay = RealtimeOverlay()

    @classmethod
    def from_artifacts(cls, artifacts: dict) -> "RecSysEngine":
        return cls(
            hyperparams=artifacts["hyperparams"],
            user_id_to_idx=artifacts["user_id_to_idx"],
            idx_to_user_id=artifacts["idx_to_user_id"],
            item_id_to_idx=artifacts["item_id_to_idx"],
            idx_to_item_id=artifacts["idx_to_item_id"],
            item_titles=artifacts["item_titles"],
            train_matrix=artifacts["train_matrix"],
            user_ratings=artifacts["user_ratings"],
            user_cf_neighbors=artifacts["user_cf_neighbors"],
            user_cf_sims=artifacts["user_cf_sims"],
            item_cf_neighbors=artifacts["item_cf_neighbors"],
            item_cf_sims=artifacts["item_cf_sims"],
            content_neighbors=artifacts["content_neighbors"],
            content_sims=artifacts["content_sims"],
            svd_user_factors=artifacts["svd_user_factors"],
            svd_item_factors=artifacts["svd_item_factors"],
            popularity=artifacts["popularity"],
            global_mean_rating=artifacts["global_mean_rating"],
        )

    def _get_user_idx(self, user_id: int) -> Optional[int]:
        return self.user_id_to_idx.get(int(user_id))

    def _user_seen_items(self, user_idx: int) -> Dict[int, float]:
        base = self.user_ratings.get(user_idx, {})
        extra = self.overlay.get_user_items(user_idx)
        if not extra:
            return base
        merged = dict(base)
        merged.update(extra)
        return merged

    def _seen_item_mask(self, user_seen: Dict[int, float]) -> np.ndarray:
        seen = np.fromiter(user_seen.keys(), dtype=int, count=len(user_seen))
        mask = np.zeros(self.train_matrix.shape[1], dtype=bool)
        mask[seen] = True
        return mask

    def _recommend_from_scores(self, scores: np.ndarray, k: int, seen_mask: np.ndarray) -> List[Tuple[int, float]]:
        scores = scores.copy()
        scores[seen_mask] = -np.inf
        if np.all(scores == -np.inf):
            return []
        if k <= 0:
            return []
        n_items = len(scores)
        k_eff = min(k, n_items)
        if k_eff <= 0:
            return []
        # Efficient top-k
        topk_idx = np.argpartition(scores, -k_eff)[-k_eff:]
        topk_sorted = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in topk_sorted]

    def score_user_cf(self, user_idx: int, user_seen: Dict[int, float]) -> np.ndarray:
        """
        User-based collaborative filtering using cosine neighbors.
        For unseen items: sum(sim(u,v) * rating(v,i)).
        """
        neighbors = self.user_cf_neighbors[user_idx]
        sims = self.user_cf_sims[user_idx]
        if neighbors.size == 0:
            return np.full(self.train_matrix.shape[1], self.global_mean_rating, dtype=float)

        # Weighted sum across neighbor users (sparse row selection)
        neigh_mat = self.train_matrix[neighbors]  # (k, n_items)
        weighted = neigh_mat.multiply(sims[:, None])
        scores = np.asarray(weighted.sum(axis=0)).ravel()
        # If user has no neighbors rating for an item, it will remain 0; add a small global mean prior.
        scores = scores + (scores == 0).astype(float) * self.global_mean_rating * 0.05
        return scores

    def score_item_cf(self, user_idx: int, user_seen: Dict[int, float]) -> np.ndarray:
        """
        Item-based CF using cosine neighbors.
        For candidate item i: sum(sim(i,j) * rating(u,j)).
        """
        n_items = self.train_matrix.shape[1]
        scores = np.zeros(n_items, dtype=float)
        seen_items = list(user_seen.items())  # (item_idx, rating)

        for j_idx, r_uj in seen_items:
            neigh_items = self.item_cf_neighbors[j_idx]
            neigh_sims = self.item_cf_sims[j_idx]
            for i_idx, sim in zip(neigh_items, neigh_sims):
                scores[i_idx] += float(sim) * float(r_uj)

        # Basic smoothing
        scores = scores + (scores == 0).astype(float) * self.global_mean_rating * 0.02
        return scores

    def score_content_based(self, user_idx: int, user_seen: Dict[int, float]) -> np.ndarray:
        """
        Content-based recommendations using cosine similarity between item genre vectors.
        For candidate item i: sum(sim_content(i,j) * rating(u,j)).
        """
        n_items = self.train_matrix.shape[1]
        scores = np.zeros(n_items, dtype=float)
        seen_items = list(user_seen.items())

        for j_idx, r_uj in seen_items:
            neigh_items = self.content_neighbors[j_idx]
            neigh_sims = self.content_sims[j_idx]
            for i_idx, sim in zip(neigh_items, neigh_sims):
                scores[i_idx] += float(sim) * float(r_uj)

        scores = scores + (scores == 0).astype(float) * self.global_mean_rating * 0.02
        return scores

    def score_svd(self, user_idx: int) -> np.ndarray:
        """
        Matrix factorization via SVD: predicted_rating(u,i) ~ U_u dot V_i.
        """
        u_vec = self.svd_user_factors[user_idx]  # (k,)
        scores = u_vec @ self.svd_item_factors  # (n_items,)
        return scores.astype(float)

    def score_hybrid(self, user_idx: int, user_seen: Dict[int, float]) -> np.ndarray:
        cf_scores = 0.5 * (self.score_user_cf(user_idx, user_seen) + self.score_item_cf(user_idx, user_seen))
        content_scores = self.score_content_based(user_idx, user_seen)
        cf_norm = minmax_normalize(cf_scores)
        content_norm = minmax_normalize(content_scores)
        return (self.hyperparams.w_cf * cf_norm) + (self.hyperparams.w_content * content_norm)

    def predict_rating(self, user_id: int, item_id: int, strategy: str = "hybrid") -> float:
        user_idx = self._get_user_idx(user_id)
        item_idx = self.item_id_to_idx.get(int(item_id))
        if user_idx is None or item_idx is None:
            return float(self.global_mean_rating)

        user_seen = self._user_seen_items(user_idx)
        if strategy == "user_cf":
            return float(self.score_user_cf(user_idx, user_seen)[item_idx])
        if strategy == "item_cf":
            return float(self.score_item_cf(user_idx, user_seen)[item_idx])
        if strategy == "content":
            return float(self.score_content_based(user_idx, user_seen)[item_idx])
        if strategy == "svd":
            return float(self.score_svd(user_idx)[item_idx])
        if strategy == "hybrid":
            return float(self.score_hybrid(user_idx, user_seen)[item_idx])
        raise ValueError(f"Unknown strategy: {strategy}")

    def popularity_recommendations(self, k: int) -> List[Tuple[int, float]]:
        scores = self.popularity.copy()
        # For unknown users there is no seen mask.
        seen_mask = np.zeros_like(scores, dtype=bool)
        return self._recommend_from_scores(scores, k=k, seen_mask=seen_mask)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        strategy: str = "hybrid",
        exclude_seen: bool = True,
    ) -> List[Dict[str, float | int | str]]:
        user_idx = self._get_user_idx(user_id)

        if user_idx is None:
            recs = self.popularity_recommendations(k)
            return [{"item_id": self.idx_to_item_id[i], "title": self.item_titles[i], "score": float(s)} for i, s in recs]

        user_seen = self._user_seen_items(user_idx)
        seen_mask = self._seen_item_mask(user_seen) if exclude_seen else np.zeros(self.train_matrix.shape[1], dtype=bool)

        if strategy == "user_cf":
            scores = self.score_user_cf(user_idx, user_seen)
        elif strategy == "item_cf":
            scores = self.score_item_cf(user_idx, user_seen)
        elif strategy == "content":
            scores = self.score_content_based(user_idx, user_seen)
        elif strategy == "svd":
            scores = self.score_svd(user_idx)
        elif strategy == "hybrid":
            scores = self.score_hybrid(user_idx, user_seen)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        recs = self._recommend_from_scores(scores, k=k, seen_mask=seen_mask)
        return [{"item_id": self.idx_to_item_id[i], "title": self.item_titles[i], "score": float(s)} for i, s in recs]

    def add_interaction_for_demo(self, user_id: int, item_id: int, rating: float) -> None:
        user_idx = self._get_user_idx(user_id)
        item_idx = self.item_id_to_idx.get(int(item_id))
        if user_idx is None or item_idx is None:
            # Cold-start demo: if user/item not known, just store nothing
            return
        self.overlay.add(user_idx, item_idx, rating)


def compute_popularity(train_df: pd.DataFrame, item_id_to_idx: Dict[int, int], n_items: int) -> np.ndarray:
    """
    Popularity-based score:
      score = mean_rating + log(1 + count)
    """
    g = train_df.groupby("item_id")["rating"].agg(["mean", "count"])
    popularity = np.zeros(n_items, dtype=float)
    for item_id, row in g.iterrows():
        if int(item_id) in item_id_to_idx:
            idx = item_id_to_idx[int(item_id)]
            popularity[idx] = float(row["mean"]) + np.log1p(float(row["count"]))
    # Smoothing: replace zeros with global average
    zeros = popularity == 0
    if np.any(zeros):
        popularity[zeros] = float(train_df["rating"].mean())
    return popularity


def train_engine(
    data_dir: str,
    hyperparams: ModelHyperParams,
    test_fraction: float = 0.2,
    seed: int = 42,
    min_user_ratings: int = 10,
) -> Tuple[RecSysEngine, dict, pd.DataFrame, pd.DataFrame]:
    """
    Trains models and returns (engine, training_report).
    """
    logger.info("Loading MovieLens...")
    ratings_df, items_df = load_movielens_100k(data_dir)
    train_df, test_df = train_test_split_interactions(
        ratings_df, test_fraction=test_fraction, min_user_ratings=min_user_ratings, seed=seed
    )

    user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id = build_mappings(train_df, items_df)
    n_users = len(user_id_to_idx)
    n_items = len(item_id_to_idx)

    logger.info("Building interaction matrix: %d users x %d items", n_users, n_items)
    train_matrix = build_interaction_matrix(train_df, user_id_to_idx, item_id_to_idx, n_users, n_items).astype(float)

    user_ratings = build_user_ratings(train_df, user_id_to_idx, item_id_to_idx)

    # Item features (genres)
    item_title_map: Dict[int, str] = {}
    filtered_items_df = items_df[items_df["item_id"].isin(item_id_to_idx.keys())].copy()
    # Ensure consistent order with idx mapping
    filtered_items_df = filtered_items_df.set_index("item_id").loc[item_id_to_idx.keys()].reset_index()
    feature_mat, _ = build_item_features(filtered_items_df)

    for item_id, idx in item_id_to_idx.items():
        title = items_df.loc[items_df["item_id"] == item_id, "title"].values[0]
        item_title_map[idx] = str(title)

    # Popularity fallback
    popularity = compute_popularity(train_df, item_id_to_idx, n_items)
    global_mean_rating = float(train_df["rating"].mean())

    # Neighborhoods by cosine similarity
    logger.info("Computing user neighbors (cosine)...")
    user_cf_neighbors, user_cf_sims = compute_cosine_neighbors(train_matrix, k=hyperparams.user_cf_k, exclude_self=True)

    logger.info("Computing item neighbors (cosine)...")
    item_cf_neighbors, item_cf_sims = compute_cosine_neighbors(train_matrix.T.tocsr(), k=hyperparams.item_cf_k, exclude_self=True)

    logger.info("Computing content neighbors (cosine on genres)...")
    content_neighbors, content_sims = compute_cosine_neighbors(sparse.csr_matrix(feature_mat), k=hyperparams.content_k, exclude_self=True)

    # SVD matrix factorization
    logger.info("Training SVD (matrix factorization)...")
    svd = TruncatedSVD(n_components=min(hyperparams.svd_components, n_users - 1, n_items - 1), random_state=hyperparams.svd_random_state)
    svd_user_factors = svd.fit_transform(train_matrix)
    # components_: (n_components, n_items)
    svd_item_factors = svd.components_

    engine = RecSysEngine(
        hyperparams=hyperparams,
        user_id_to_idx=user_id_to_idx,
        idx_to_user_id=idx_to_user_id,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=idx_to_item_id,
        item_titles=item_title_map,
        train_matrix=train_matrix,
        user_ratings=user_ratings,
        user_cf_neighbors=user_cf_neighbors,
        user_cf_sims=user_cf_sims,
        item_cf_neighbors=item_cf_neighbors,
        item_cf_sims=item_cf_sims,
        content_neighbors=content_neighbors,
        content_sims=content_sims,
        svd_user_factors=svd_user_factors,
        svd_item_factors=svd_item_factors,
        popularity=popularity,
        global_mean_rating=global_mean_rating,
    )

    training_report = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_users": n_users,
        "n_items": n_items,
        "hyperparams": hyperparams.__dict__,
        "test_interactions": int(len(test_df)),
        "train_interactions": int(len(train_df)),
    }

    return engine, training_report, train_df, test_df


def save_artifacts(engine: RecSysEngine, model_path: str, extra: Optional[dict] = None) -> None:
    ensure_dir(os.path.dirname(model_path))
    artifacts = {
        "hyperparams": engine.hyperparams,
        "user_id_to_idx": engine.user_id_to_idx,
        "idx_to_user_id": engine.idx_to_user_id,
        "item_id_to_idx": engine.item_id_to_idx,
        "idx_to_item_id": engine.idx_to_item_id,
        "item_titles": engine.item_titles,
        "train_matrix": engine.train_matrix,
        "user_ratings": engine.user_ratings,
        "user_cf_neighbors": engine.user_cf_neighbors,
        "user_cf_sims": engine.user_cf_sims,
        "item_cf_neighbors": engine.item_cf_neighbors,
        "item_cf_sims": engine.item_cf_sims,
        "content_neighbors": engine.content_neighbors,
        "content_sims": engine.content_sims,
        "svd_user_factors": engine.svd_user_factors,
        "svd_item_factors": engine.svd_item_factors,
        "popularity": engine.popularity,
        "global_mean_rating": engine.global_mean_rating,
    }
    if extra:
        artifacts["extra"] = extra
    joblib.dump(artifacts, model_path, compress=3)
    logger.info("Saved artifacts to %s", model_path)


def load_engine(model_path: str) -> RecSysEngine:
    artifacts = joblib.load(model_path)
    return RecSysEngine.from_artifacts(artifacts)


def build_engine_and_artifacts(
    data_dir: str,
    model_path: str,
    hyperparams: Optional[ModelHyperParams] = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    min_user_ratings: int = 10,
) -> Tuple[RecSysEngine, dict]:
    if hyperparams is None:
        hyperparams = ModelHyperParams()

    engine, report, _, _ = train_engine(
        data_dir=data_dir,
        hyperparams=hyperparams,
        test_fraction=test_fraction,
        seed=seed,
        min_user_ratings=min_user_ratings,
    )
    save_artifacts(engine, model_path, extra={"report": report})
    return engine, report


def _format_recommendation_payload(engine: RecSysEngine, items: List[Tuple[int, float]]) -> List[dict]:
    return [{"item_id": engine.idx_to_item_id[i], "title": engine.item_titles[i], "score": float(s)} for i, s in items]


def recommend_for_unknown_user(engine: RecSysEngine, k: int) -> List[dict]:
    recs = engine.popularity_recommendations(k)
    return _format_recommendation_payload(engine, recs)


def artifacts_to_jsonable(model_path: str, out_path: str) -> None:
    """
    Convenience for debugging: extracts only JSON-serializable bits.
    """
    artifacts = joblib.load(model_path)
    jsonable = {k: v for k, v in artifacts.items() if k not in {"train_matrix"}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, indent=2, default=str)

