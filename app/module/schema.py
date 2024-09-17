from pydantic import BaseModel, Field, field_validator
from module.datapipeline import DataPipeline as dp


class TopMoviesParams(BaseModel):
    n: int = Field(default=10, ge=10, le=100)

    @property
    def valid_categories(self):
        return [10, 20, 50, 100]

    # 입력 'data' 객체 초기화, 데이터 유효성 검증.
    def __init__(self, **data):
        super().__init__(**data)
        if self.n not in self.valid_categories:
            raise ValueError(f"n must be one of {self.valid_categories}")


class UsersParams(BaseModel):
    user_id: int = Field(default=1, ge=1, le=1000000)

    @field_validator("user_id")
    @classmethod
    def valid_categories(cls, v):
        pipeline = dp()
        valid_user_ids = pipeline.get_all_user_ids()
        if v not in valid_user_ids:
            raise ValueError(f"Not Found User {v}")
        return v


class RecommendParams(BaseModel):
    user_id: int = Field(..., ge=1, le=1000000)
    n: int = Field(default=10, ge=1, le=100)
