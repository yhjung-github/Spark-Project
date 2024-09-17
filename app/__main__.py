import uvicorn

from fastapi import FastAPI, Depends, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from module.MovieLens import MovieLens
from module.Recommend import Recommend, SVDModelConfig

from contextlib import asynccontextmanager

from functools import lru_cache

from module.schema import TopMoviesParams, UsersParams, RecommendParams


spark_app = None
recommend_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global spark_app, recommend_model
    spark_app = MovieLens()
    model_config = SVDModelConfig(
        model_name="SVD_Model_20240601_120000", model_version=1.0, model_dir="./models"
    )
    recommend_model = Recommend(spark_app=spark_app, model_config=model_config)
    yield
    spark_app.spark.stop()


app = FastAPI(lifespan=lifespan)


@lru_cache(maxsize=4)
def get_top_movies(n: int):
    top_movies_df = spark_app.top_movies[n]
    pandas_df = top_movies_df.toPandas()
    return pandas_df.head(n).to_dict(orient="records")


@lru_cache(maxsize=16)
def get_user_ratings(user_id: int):
    user_ratings_df = spark_app.get_user_ratings(user_id)
    pandas_df = user_ratings_df.toPandas()
    return pandas_df.to_dict(orient="records")


@lru_cache(maxsize=16)
def get_recommend(user_id: int):
    return recommend_model.predict(user_id=user_id)


@app.get("/recommend")
async def read_recommend(
    user_id: int = Query(default=1, ge=1, le=1000000),
    n: int = Query(default=10, ge=1, le=100),
):
    try:
        params = RecommendParams(user_id=user_id, n=n)
        recommend_df = get_recommend(params.user_id)
        return recommend_df[: params.n]
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/top-movies")
async def read_top_movies(n: int = Query(default=10, ge=10, le=100)):
    try:
        params = TopMoviesParams(n=n)
        return get_top_movies(params.n)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/user-ratings")
async def read_top_movies_by_user(user_id: int = Query(default=1, ge=1, le=1000000)):
    try:
        params = UsersParams(user_id=user_id)
        return get_user_ratings(params.user_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
