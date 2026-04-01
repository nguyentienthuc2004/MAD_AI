import os
import json
from dataclasses import asdict, dataclass, replace
from typing import Dict, List

# Avoid OpenBLAS over-threading warning/perf issues when training implicit ALS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from dotenv import load_dotenv


def _load_env() -> None:
    # Load .env next to this file so running recommend.py directly works.
    env_path = os.getenv("DOTENV_PATH") or os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)


@dataclass
class RecommenderConfig:
    database: str = "blog_db"
    collection: str = "user_activities"
    decay_days: int = 30
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 30
    alpha: float = 40.0
    eval_k: int = 10
    random_state: int = 42
    activity_weight: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.activity_weight is None:
            self.activity_weight = {
                "view": 1.0,
                "like": 3.0,
                "comment": 6.0,
            }


class ActivityRepository:
    def __init__(self, mongo_uri: str, config: RecommenderConfig) -> None:
        self.client = MongoClient(mongo_uri)
        self.config = config

    def fetch_activities(self) -> pd.DataFrame:
        collection = self.client[self.config.database][self.config.collection]
        data = list(collection.find())
        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame(columns=["userId", "postId", "activity_type", "createdAt"])

        if "isDeleted" in df.columns:
            df = df[df["isDeleted"] == False]

        required_columns = ["userId", "postId", "activity_type", "createdAt"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        return df[required_columns].copy()


class ALSRecommender:
    def __init__(self, config: RecommenderConfig) -> None:
        self.config = config
        self.model = AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            random_state=self.config.random_state,
        )
        self.matrix = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_post: Dict[int, str] = {}
        self.user_seen_posts: Dict[str, set[str]] = {}
        self.popular_post_ids: List[str] = []

    def fit(self, df: pd.DataFrame) -> None:
        interactions = self._build_interactions(df)
        if interactions.empty:
            raise ValueError("No interaction data available to train recommender")

        # Cache seen and popularity views so we can backfill recommendations on sparse data.
        self.user_seen_posts = (
            interactions.groupby("userId")["postId"].apply(lambda x: set(x.astype(str))).to_dict()
        )
        popularity = (
            interactions.groupby("postId", as_index=False)["weight"]
            .sum()
            .sort_values("weight", ascending=False)
        )
        self.popular_post_ids = popularity["postId"].astype(str).tolist()

        user_ids = interactions["userId"].astype(str).unique()
        post_ids = interactions["postId"].astype(str).unique()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        post_to_idx = {post_id: idx for idx, post_id in enumerate(post_ids)}
        self.idx_to_post = {idx: post_id for post_id, idx in post_to_idx.items()}

        interactions["user_idx"] = interactions["userId"].astype(str).map(self.user_to_idx)
        interactions["post_idx"] = interactions["postId"].astype(str).map(post_to_idx)

        rows = interactions["user_idx"].to_numpy()
        cols = interactions["post_idx"].to_numpy()
        values = interactions["weight"].to_numpy()

        self.matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(post_ids)))
        self.model.fit(self.matrix * self.config.alpha)

    

    def _build_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        data = df.copy()
        data["base_weight"] = data["activity_type"].map(self.config.activity_weight).fillna(1.0)

        if "createdAt" in data.columns:
            data["timestamp"] = pd.to_datetime(data["createdAt"], errors="coerce")
            now = data["timestamp"].max()
            if pd.isna(now):
                data["time_weight"] = 1.0
            else:
                data["days_ago"] = (now - data["timestamp"]).dt.days.fillna(0)
                data["time_weight"] = np.exp(-data["days_ago"] / self.config.decay_days)
        else:
            data["time_weight"] = 1.0

        data["weight"] = np.log1p(data["base_weight"] * data["time_weight"])

        return (
            data.groupby(["userId", "postId"], as_index=False)["weight"]
            .sum()
            .dropna(subset=["userId", "postId"])
        )
    def recommend_post_ids(self, user_id: str, k: int = 20) -> List[str]:
      if self.matrix is None:
          raise RuntimeError("Recommender is not trained yet")

      user_key = str(user_id)
      seen_post_ids = self.user_seen_posts.get(user_key, set())
      user_idx = self.user_to_idx.get(user_key)

      post_ids: List[str] = []
      used: set[str] = set()

      if user_idx is not None:
          seen_items = set(self.matrix[user_idx].indices)
          ids, _scores = self.model.recommend(
              userid=user_idx,
              user_items=self.matrix[user_idx],
              N=max(k * 3, k),
          )

          for post_idx in ids:
              if post_idx in seen_items:
                  continue

              post_id = self.idx_to_post.get(int(post_idx))
              if post_id is None or post_id in used or post_id in seen_post_ids:
                  continue

              post_ids.append(post_id)
              used.add(post_id)
              if len(post_ids) >= k:
                  break

      # Popularity fallback keeps quality reasonable for tiny demo datasets.
      for post_id in self.popular_post_ids:
          if post_id in used or post_id in seen_post_ids:
              continue

          post_ids.append(post_id)
          used.add(post_id)
          if len(post_ids) >= k:
              break

      return post_ids

def build_recommender(mongo_uri: str) -> ALSRecommender:
    config = RecommenderConfig()
    repository = ActivityRepository(mongo_uri, config)
    model = ALSRecommender(config)
    model.fit(repository.fetch_activities())
    return model


def _temporal_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-item-out split per user.

    For each user, pick the most recent post as validation target and remove *all*
    interactions of that user-post from training. This avoids leakage where the
    held-out item was already seen in train for the same user.
    """
    if df.empty:
        return df.copy(), df.copy()

    data = df.copy()
    data["createdAt"] = pd.to_datetime(data["createdAt"], errors="coerce")
    data = data.dropna(subset=["userId", "postId", "createdAt"]).sort_values("createdAt")

    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    for _user_id, group in data.groupby("userId", sort=False):
        if len(group) < 2:
            continue

        group_sorted = group.sort_values("createdAt")
        val_post_id = str(group_sorted.iloc[-1]["postId"])

        # Remove all user interactions of the held-out post from train.
        train_group = group_sorted[group_sorted["postId"].astype(str) != val_post_id]
        if train_group.empty:
            continue

        val_parts.append(group_sorted.tail(1))
        train_parts.append(train_group)

    if not train_parts or not val_parts:
        return data.copy(), pd.DataFrame(columns=data.columns)

    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def _average_precision_at_k(predictions: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    hit_count = 0
    score = 0.0
    for rank, post_id in enumerate(predictions[:k], start=1):
        if post_id in relevant:
            hit_count += 1
            score += hit_count / rank

    denom = min(len(relevant), k)
    return score / denom if denom > 0 else 0.0


def _evaluate_grouped(
    grouped: Dict[str, set[str]],
    predict_fn,
    k: int = 10,
) -> Dict[str, float]:
    if not grouped:
        return {
            "users_evaluated": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    map_scores: List[float] = []
    ndcg_scores: List[float] = []

    for user_id, relevant_items in grouped.items():
        predictions = predict_fn(str(user_id), k)
        if not predictions:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            map_scores.append(0.0)
            ndcg_scores.append(0.0)
            continue

        pred_top_k = predictions[:k]
        hits = sum(1 for post_id in pred_top_k if post_id in relevant_items)

        precision_scores.append(hits / k)
        recall_scores.append(hits / len(relevant_items))
        map_scores.append(_average_precision_at_k(pred_top_k, relevant_items, k))

        dcg = 0.0
        for idx, post_id in enumerate(pred_top_k):
            rel = 1.0 if post_id in relevant_items else 0.0
            dcg += rel / np.log2(idx + 2)

        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
        ndcg_scores.append((dcg / idcg) if idcg > 0 else 0.0)

    return {
        "users_evaluated": len(grouped),
        "precision_at_k": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_at_k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "map_at_k": float(np.mean(map_scores)) if map_scores else 0.0,
        "ndcg_at_k": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
    }


def evaluate_recommender(model: ALSRecommender, val_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
    """Evaluate model using ranking metrics on held-out interactions."""
    if val_df.empty:
        return {
            "users_evaluated": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    grouped = val_df.groupby("userId")["postId"].apply(lambda x: set(x.astype(str))).to_dict()
    return _evaluate_grouped(grouped, model.recommend_post_ids, k)


def evaluate_popularity_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    k: int,
) -> Dict[str, float]:
    if train_df.empty or val_df.empty:
        return {
            "users_evaluated": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    interactions = ALSRecommender(RecommenderConfig())._build_interactions(train_df)
    popularity = (
        interactions.groupby("postId", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)["postId"]
        .astype(str)
        .tolist()
    )

    seen_by_user = (
        train_df.groupby("userId")["postId"].apply(lambda x: set(x.astype(str))).to_dict()
    )
    grouped = val_df.groupby("userId")["postId"].apply(lambda x: set(x.astype(str))).to_dict()

    def _predict(user_id: str, top_k: int) -> List[str]:
        seen = seen_by_user.get(user_id, set())
        out: List[str] = []
        for post_id in popularity:
            if post_id in seen:
                continue
            out.append(post_id)
            if len(out) >= top_k:
                break
        return out

    return _evaluate_grouped(grouped, _predict, k)


def _composite_score(metrics: Dict[str, float]) -> float:
    return (
        0.6 * metrics.get("ndcg_at_k", 0.0)
        + 0.3 * metrics.get("map_at_k", 0.0)
        + 0.1 * metrics.get("recall_at_k", 0.0)
    )


def select_best_recommender(
    mongo_uri: str,
    candidate_configs: List[RecommenderConfig] | None = None,
) -> tuple[ALSRecommender, RecommenderConfig, Dict[str, float]]:
    """Try multiple ALS configs, evaluate, and return the best model retrained on all data."""
    base_config = RecommenderConfig()
    repository = ActivityRepository(mongo_uri, base_config)
    raw_df = repository.fetch_activities()

    if raw_df.empty:
        raise ValueError("No interaction data available to train recommender")

    train_df, val_df = _temporal_holdout_split(raw_df)

    raw_users = raw_df["userId"].astype(str).nunique()
    raw_posts = raw_df["postId"].astype(str).nunique()
    raw_interactions = len(raw_df)
    print("[data_profile]")
    print(
        f"  interactions={raw_interactions} users={raw_users} posts={raw_posts} "
        f"avg_interactions_per_user={(raw_interactions / raw_users) if raw_users else 0:.2f}"
    )

    train_posts = set(train_df["postId"].astype(str)) if not train_df.empty else set()
    val_posts = set(val_df["postId"].astype(str)) if not val_df.empty else set()
    val_overlap = len(train_posts & val_posts)
    val_overlap_ratio = (val_overlap / len(val_posts)) if val_posts else 0.0
    print("[split_profile]")
    print(
        f"  train={len(train_df)} val={len(val_df)} "
        f"val_post_overlap_with_train={val_overlap_ratio:.2%}"
    )
    if candidate_configs is None:
        candidate_configs = [
            RecommenderConfig(factors=32, regularization=0.05, iterations=30, alpha=20.0),
            RecommenderConfig(factors=64, regularization=0.05, iterations=30, alpha=40.0),
            RecommenderConfig(factors=96, regularization=0.02, iterations=30, alpha=40.0),
            RecommenderConfig(factors=128, regularization=0.01, iterations=40, alpha=60.0),
        ]

    seed_env = os.getenv("EVAL_SEEDS", "7,42,2026")
    eval_seeds = [int(s.strip()) for s in seed_env.split(",") if s.strip().isdigit()]
    if not eval_seeds:
        eval_seeds = [42]

    baseline_metrics = evaluate_popularity_baseline(train_df, val_df, k=base_config.eval_k)
    baseline_score = _composite_score(baseline_metrics)
    print("[baseline_popularity]")
    print(
        f"  users={baseline_metrics['users_evaluated']} "
        f"P@{base_config.eval_k}={baseline_metrics['precision_at_k']:.4f} "
        f"R@{base_config.eval_k}={baseline_metrics['recall_at_k']:.4f} "
        f"MAP@{base_config.eval_k}={baseline_metrics['map_at_k']:.4f} "
        f"NDCG@{base_config.eval_k}={baseline_metrics['ndcg_at_k']:.4f} "
        f"score={baseline_score:.4f}"
    )

    best_score = -1.0
    best_config = candidate_configs[0]
    best_seed = eval_seeds[0]
    best_metrics: Dict[str, float] = {}
    eval_report: Dict[str, object] = {
        "data_profile": {
            "interactions": raw_interactions,
            "users": raw_users,
            "posts": raw_posts,
        },
        "split_profile": {
            "train": len(train_df),
            "val": len(val_df),
            "val_post_overlap_with_train": val_overlap_ratio,
        },
        "baseline_popularity": {
            "metrics": baseline_metrics,
            "composite_score": baseline_score,
        },
        "candidates": [],
        "selection_policy": "max(mean_composite_score)",
    }

    for config in candidate_configs:
        metrics_runs: List[Dict[str, float]] = []
        for seed in eval_seeds:
            seeded_config = replace(config, random_state=seed)
            model = ALSRecommender(seeded_config)
            model.fit(train_df)
            metrics_runs.append(evaluate_recommender(model, val_df, k=config.eval_k))

        mean_metrics = {
            "users_evaluated": int(np.mean([m["users_evaluated"] for m in metrics_runs])),
            "precision_at_k": float(np.mean([m["precision_at_k"] for m in metrics_runs])),
            "recall_at_k": float(np.mean([m["recall_at_k"] for m in metrics_runs])),
            "map_at_k": float(np.mean([m["map_at_k"] for m in metrics_runs])),
            "ndcg_at_k": float(np.mean([m["ndcg_at_k"] for m in metrics_runs])),
        }

        run_scores = [_composite_score(m) for m in metrics_runs]
        mean_score = float(np.mean(run_scores))
        score_std = float(np.std(run_scores))

        if mean_score > best_score:
            best_score = mean_score
            best_config = replace(config, random_state=eval_seeds[int(np.argmax(run_scores))])
            best_seed = best_config.random_state
            best_metrics = mean_metrics

        eval_report["candidates"].append(
            {
                "config": asdict(config),
                "seeds": eval_seeds,
                "mean_metrics": mean_metrics,
                "run_scores": run_scores,
                "mean_composite_score": mean_score,
                "score_std": score_std,
            }
        )

        print(
            "[candidate] "
            f"factors={config.factors}, reg={config.regularization}, "
            f"iter={config.iterations}, alpha={config.alpha}, seeds={eval_seeds}"
        )
        print(
            f"  users={mean_metrics['users_evaluated']} "
            f"P@{config.eval_k}={mean_metrics['precision_at_k']:.4f} "
            f"R@{config.eval_k}={mean_metrics['recall_at_k']:.4f} "
            f"MAP@{config.eval_k}={mean_metrics['map_at_k']:.4f} "
            f"NDCG@{config.eval_k}={mean_metrics['ndcg_at_k']:.4f} "
            f"score={mean_score:.4f} std={score_std:.4f}"
        )

    final_model = ALSRecommender(best_config)
    final_model.fit(raw_df)

    eval_report["best_model"] = {
        "config": asdict(best_config),
        "best_seed": best_seed,
        "mean_metrics": best_metrics,
        "mean_composite_score": best_score,
        "beats_popularity": best_score >= baseline_score,
    }

    eval_output_path = os.getenv("MODEL_EVAL_OUTPUT", os.path.join(os.path.dirname(__file__), "model_eval.json"))
    try:
        with open(eval_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)
        print(f"[eval_report] saved={eval_output_path}")
    except OSError as ex:
        print(f"[eval_report] save_failed: {ex}")

    return final_model, best_config, best_metrics


if __name__ == "__main__":
    _load_env()
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("Missing MONGO_URI environment variable")

    recommender, best_config, best_metrics = select_best_recommender(uri)
    print("\n[best_model]")
    print(
        "  "
        f"factors={best_config.factors}, reg={best_config.regularization}, "
        f"iter={best_config.iterations}, alpha={best_config.alpha}"
    )
    print(
        f"  users={best_metrics.get('users_evaluated', 0)} "
        f"P@{best_config.eval_k}={best_metrics.get('precision_at_k', 0.0):.4f} "
        f"R@{best_config.eval_k}={best_metrics.get('recall_at_k', 0.0):.4f} "
        f"MAP@{best_config.eval_k}={best_metrics.get('map_at_k', 0.0):.4f} "
        f"NDCG@{best_config.eval_k}={best_metrics.get('ndcg_at_k', 0.0):.4f} "
        f"score={_composite_score(best_metrics):.4f}"
    )

    sample_user_id = os.getenv("SAMPLE_USER_ID", "")
    if sample_user_id:
        print("\n[sample_recommendations]")
        print(recommender.recommend_post_ids(sample_user_id, k=30))
