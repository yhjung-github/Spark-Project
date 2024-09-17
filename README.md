# MovieLens Recommendation System

이 프로젝트는 MovieLens 데이터셋을 사용하여 영화 추천 시스템을 구현한 FastAPI 기반의 RESTful API 서비스입니다.

## 주요 기능

1. 영화 데이터 처리 및 분석
2. 사용자 기반 영화 추천
3. 인기 영화 목록 제공
4. 주기적인 모델 재학습

## 기술 스택

- **FastAPI**: RESTful API 구현
- **PySpark**: 대용량 데이터 처리
- **Surprise**: SVD 알고리즘을 이용한 추천 시스템 구현
- **Pydantic**: 데이터 검증 및 설정 관리

## TODO

- **APScheduler**: 주기적인 모델 재학습 스케줄링 및 학습 API 추가
- **Streamlit**: API 시각화
- **NUI**: Natural User Interface 기능 추가

## 프로젝트 구조

- `__main__.py`: FastAPI 애플리케이션 및 API 엔드포인트 정의
- `module/MovieLens.py`: PySpark를 이용한 MovieLens 데이터 처리
- `module/Recommend.py`: SVD 모델 기반 추천 시스템 구현
- `module/schema.py`: API 요청/응답 데이터 모델 정의


## 주요 API 엔드포인트

- `GET /top-movies`: 인기 영화 목록 조회
- `GET /recommend?user_id={user_id}`: 사용자 기반 영화 추천

## 설치 및 실행
0. Python 3.11

0. 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

0. 애플리케이션 실행:
   ```bash
   python -m app
   ```

## 데이터

이 프로젝트는 MovieLens 100K 데이터셋을 사용합니다. 데이터는 첫 실행 시 자동으로 다운로드되며, `SPARK-PROJECT/app/data` 디렉토리에 저장

## 모델 학습 
- SVD 모델 학습 및 저장, `SPARK-PROJECT/models` 디렉토리에 저장
- (TO-DO) `POST /train` 엔드포인트를 통해 모델을 수동 학습
- (TO-DO) 선택적으로 1분 간격의 주기적 재학습
