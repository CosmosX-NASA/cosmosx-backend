# 1. Base Image
#FROM --platform=linux/arm64 python:3.11-slim
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Poetry or pip 기반 설치 (pyproject.toml & requirements로 처리)
COPY pyproject.toml .
# (만약 poetry.lock이 있다면 같이 복사)
# COPY poetry.lock .

# 5. pip로 의존성 설치 (uv 등 포함)
RUN pip install --upgrade pip
# 만약 Poetry를 사용하지 않는다면 requirements 추출
# pip가 pyproject를 자동 인식하도록 하거나, 아래처럼 수동 설치도 가능
RUN pip install fastapi uvicorn \
    bcrypt ipykernel joblib pillow pydantic pyjwt python-decouple \
    python-dotenv python-multipart scikit-learn sqlalchemy

# 6. 애플리케이션 코드 복사
COPY . .

# 7. 컨테이너 실행 시 uvicorn 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
