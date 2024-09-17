import pandas as pd
import os
import urllib.request
import zipfile
from pyspark.sql import SparkSession
from pydantic import BaseModel
from typing import Dict, List
from pyspark.sql.functions import avg, col
from surprise import SVD


class MovieLensConfig(BaseModel):
    columns_dict: Dict[str, List[str]] = {
        "movies": ["movie_id", "title"] + [f"col_{i}" for i in range(2, 24)],
        "rating": ["user_id", "movie_id", "rating", "timestamp"],
        "items": [
            "movie id",
            "movie title",
            "release date",
            "video release date",
            "IMDb URL",
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
        ],
        "base": ["user_id", "movie_id", "rating", "timestamp"],
        "test": ["user_id", "movie_id", "rating", "timestamp"],
    }
    data_dict: Dict[str, str] = {
        "movies": "u.item",
        "rating": "u.data",
        "items": "u.item",
        "base": "ua.base",
        "test": "ua.test",
    }
    sep_dict: Dict[str, str] = {
        "movies": "|",
        "rating": "\t",
        "items": "|",
        "base": "\t",
        "test": "\t",
    }
    encoding_dict: Dict[str, str] = {
        "movies": "ISO-8859-1",
        "rating": "utf-8",
        "items": "ISO-8859-1",
        "base": "utf-8",
        "test": "utf-8",
    }


class MovieLens:
    def __init__(self):
        print("MovieLens initalizing...")

        self.data_name = "ml-100k"
        self.url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        self.zip_path = "ml-100k.zip"
        self.spark = self.spark_session("MovieLens")
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "data")
        self.config = MovieLensConfig()

        self.movies = self.load_data(data="movies").select("movie_id", "title")
        self.ratings = self.load_data(data="rating").select(
            "user_id", "movie_id", "rating"
        )
        self.items = self.load_data(data="items")

        self.movie_ratings = self.movies.join(self.ratings, on="movie_id")
        self.average_ratings = self.movie_ratings.groupBy("movie_id", "title").agg(
            avg("rating").alias("average_rating")
        )
        self.top_movies = {n: self.get_top_movies(n=n) for n in [10, 20, 50, 100]}

    def spark_session(self, app_name):
        return SparkSession.builder.appName(app_name).getOrCreate()

    def download_data(self):
        if os.path.exists(self.data_dir):
            return self.data_dir
        else:
            print("데이터 다운로드 중...")
            os.makedirs(self.data_dir, exist_ok=True)

            zip_path = os.path.join(self.data_dir, self.zip_path)
            urllib.request.urlretrieve(self.url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            os.remove(zip_path)
        return self.data_dir

    def load_data(self, data, header=False):
        file_path = os.path.join(
            self.data_dir, self.data_name, self.config.data_dict[data]
        )
        df = self.spark.read.csv(
            file_path,
            sep=self.config.sep_dict[data],
            encoding=self.config.encoding_dict[data],
            inferSchema=True,
            header=header,
        )
        if self.config.columns_dict[data] is not None:
            df = df.toDF(*self.config.columns_dict[data])
        return df

    def get_top_movies(self, n):
        # 사용자 평정 정보를 이용해 영화의 평점을 계산
        average_ratings = self.movie_ratings.groupBy("movie_id", "title").agg(
            avg("rating").alias("average_rating")
        )
        top_movies = average_ratings.orderBy("average_rating", ascending=False).limit(n)
        return top_movies

    def get_user_ratings(self, user_id):
        # col : 컬럼명 지정 ex) 컬럼 user_id 지정시 col("user_id") == user_id"
        user_ratings = self.ratings.filter(col("user_id") == user_id).select(
            "movie_id", "rating"
        )
        if user_ratings.count() == 0:
            return None

        # user_ratings.show(truncate=False)  # truncate(False) 전체 길이 표시, 기본 값: 20
        return user_ratings

    def recommend_movies(self, user_id):
        user_ratings = self.get_user_ratings(user_id)
        if user_ratings is None:
            return None

        # 사용자가 평점을 매긴 영화의 평점 정보를 이용해 영화의 평점을 계산
        average_ratings = self.movie_ratings.groupBy("movie_id", "title").agg(
            avg("rating").alias("average_rating")
        )

    def get_all_movies(self):
        movies_list = (
            self.movies.select("movie_id", "title").orderBy("movie_id").collect()
        )
        return [
            {"movie_id": movie.movie_id, "title": movie.title} for movie in movies_list
        ]

    def get_all_datasets(self):
        base_data = self.load_data(data="base")
        test_data = self.load_data(data="test")
        return base_data, test_data
