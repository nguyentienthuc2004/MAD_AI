import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from pymongo import MongoClient
from scipy.sparse import csr_matrix


@dataclass
class RecommenderConfig:
    database: str = "blog_db"
    collection: str = "user_activities"
    decay_days: int = 30
    factors: int = 64
    regularization: float = 0.05
    iterations: int = 30
    alpha: float = 40.0
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
        )
        self.matrix = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_post: Dict[int, str] = {}

    def fit(self, df: pd.DataFrame) -> None:
        interactions = self._build_interactions(df)
        if interactions.empty:
            raise ValueError("No interaction data available to train recommender")

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

    def recommend_post_ids(self, user_id: str, k: int = 20) -> List[str]:
        if self.matrix is None:
            raise RuntimeError("Recommender is not trained yet")

        user_key = str(user_id)
        user_idx = self.user_to_idx.get(user_key)
        if user_idx is None:
            return []

        seen_items = set(self.matrix[user_idx].indices)
        ids, _scores = self.model.recommend(
            userid=user_idx,
            user_items=self.matrix[user_idx],
            N=max(k * 3, k),
        )

        post_ids: List[str] = []
        for post_idx in ids:
            if post_idx in seen_items:
                continue

            post_id = self.idx_to_post.get(int(post_idx))
            if post_id is None:
                continue

            post_ids.append(post_id)
            if len(post_ids) >= k:
                break

        return post_ids

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


def build_recommender(mongo_uri: str) -> ALSRecommender:
    config = RecommenderConfig()
    repository = ActivityRepository(mongo_uri, config)
    model = ALSRecommender(config)
    model.fit(repository.fetch_activities())
    return model


if __name__ == "__main__":
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("Missing MONGO_URI environment variable")

    recommender = build_recommender(uri)
    sample_user_id = os.getenv("SAMPLE_USER_ID", "")
    if sample_user_id:
        print(recommender.recommend_post_ids(sample_user_id, k=10))
