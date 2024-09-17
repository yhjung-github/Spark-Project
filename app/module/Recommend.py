import os
import pickle
import json

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from module.MovieLens import MovieLens

import time

from pydantic import BaseModel, Field, field_validator

from typing import Any, Optional
from datetime import datetime


class SVDModelConfig(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    model_name: str = Field(default="")
    model_version: Optional[float] = None
    model_dir: str
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None

    @field_validator("model_name")
    @classmethod
    def set_model_name(cls, v: str, info: Any) -> str:
        if v:
            return v
        created_at = info.data.get("created_at", datetime.now())
        return f"SVD_Model_{created_at.strftime('%Y%m%d_%H%M%S')}"

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        d["created_at"] = d["created_at"].isoformat()
        return d


class Recommend:
    def __init__(self, spark_app: MovieLens, model_config: SVDModelConfig):
        self.ml = spark_app
        self.movie_titles_cache = None
        ratings_base, ratings_test = self.ml.get_all_datasets()

        # 단일 노드 처리에 최적화된 pd.dataframe으로 변환
        self.ratings_base_pd = ratings_base.toPandas()
        self.ratings_test_pd = ratings_test.toPandas()

        # 데이터 포맷 변환
        self.reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(
            self.ratings_base_pd[["user_id", "movie_id", "rating"]], self.reader
        )
        self.trainset, self.testset = train_test_split(self.data, test_size=0.25)

        self.model = None
        self.model_config = model_config
        os.makedirs(self.model_config.model_dir, exist_ok=True)

        if not self.load_model():
            self.train()
            self.test()

    def cache_movie_titles(self):
        if self.movie_titles_cache is None:
            self.movie_titles_cache = dict(
                self.ml.movies.select("movie_id", "title").collect()
            )

    def train(self):
        print(
            f"Training model: {self.model_config.model_name}, version: {self.model_config.model_version}"
        )
        self.model = SVD()
        self.model.fit(self.trainset)
        print("모델 학습 완료")

        # 모델 저장
        self.save_model()

    def save_model(self):
        os.makedirs(self.model_config.model_dir, exist_ok=True)

        # 모델 저장
        model_path = os.path.join(
            self.model_config.model_dir, f"{self.model_config.model_name}.pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # 모델 설정 저장
        config_path = os.path.join(
            self.model_config.model_dir, f"{self.model_config.model_name}_config.json"
        )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                self.model_config.dict(), f, ensure_ascii=False, indent=4, default=str
            )

        print(f"모델이 {model_path}에 저장되었습니다.")
        print(f"모델 설정이 {config_path}에 저장되었습니다.")

    def load_model(self):
        model_path = os.path.join(
            self.model_config.model_dir, f"{self.model_config.model_name}.pkl"
        )
        config_path = os.path.join(
            self.model_config.model_dir, f"{self.model_config.model_name}_config.json"
        )

        if os.path.exists(model_path) and os.path.exists(config_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                self.model_config = SVDModelConfig(**loaded_config)
            return True
        else:
            return False

    def test(self):
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return

        predictions = self.model.test(self.testset)
        self.model_config.test_rmse = accuracy.rmse(predictions)
        self.model_config.test_mae = accuracy.mae(predictions)
        print(f"Test RMSE: {self.model_config.test_rmse}")
        print(f"Test MAE: {self.model_config.test_mae}")

    def predict(self, user_id):
        start_time = time.time()
        print("예측 Processing...")
        if self.model is None:
            print("모델이 학습되지 않았습니다. train() 함수를 먼저 실행하세요.")
            return

        user_ratings = self.ml.get_user_ratings(user_id)
        if user_ratings is None:
            print("해당 사용자의 평점 데이터가 없습니다.")
            return None

        rated_movies = set(user_ratings.select("movie_id").toPandas()["movie_id"])
        all_movies = set(self.ml.movies.toPandas()["movie_id"])
        unrated_movies = all_movies - rated_movies

        # 예측 수행
        predictions = []
        for movie_id in unrated_movies:
            pred = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

        top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        result = get_movie_details_for_predictions(self, top_predictions)
        print(f"사용자:{user_id} 사용자의 예측 결과: {result[:10]}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"예측 처리 시간: {elapsed_time:.2f} 초")

        return result


def elapsed_time_check(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} 처리 시간: {elapsed_time:.2f} 초")
        return result

    return wrapper


@elapsed_time_check
def get_movie_details_for_predictions(self, top_predictions):
    if self.movie_titles_cache is None:
        self.cache_movie_titles()

    return [
        {
            "movie_id": movie_id,
            "title": self.movie_titles_cache.get(movie_id, "Unknown"),
            "predicted_rating": rating,
        }
        for movie_id, rating in top_predictions
    ]


if __name__ == "__main__":
    model_config = SVDModelConfig(model_version=1.0, model_dir="./models")
    recommend = Recommend(model_config)
    recommend.predict(user_id=1)
    print("예측 완료")
