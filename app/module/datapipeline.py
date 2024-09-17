import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql.functions import collect_set
from module.MovieLens import MovieLens


class DataPipeline:
    def __init__(self):
        self.spark_app = MovieLens()

    def get_all_user_ids(self):
        # 모든 고유한 user_id 추출
        user_ids = self.spark_app.ratings.select(collect_set("user_id")).first()[0]
        return sorted(user_ids)  # 정렬된 리스트로 반환


if __name__ == "__main__":
    pipeline = DataPipeline()
    all_user_ids = pipeline.get_all_user_ids()
    print(all_user_ids)
