# 1. Base Image (슬림 + 빌드툴 최소)
FROM python:3.11-slim

# 2. 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 3. 작업 디렉토리
WORKDIR /app

# 4. 시스템 패키지 설치 (torch, transformers, faiss 설치 대비)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. pip 업그레이드
RUN pip install --upgrade pip

# 6. 필요한 패키지만 설치
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    python-dotenv \
    sqlalchemy \
    python-multipart \
    openai \
    numpy \
    pandas \
    faiss-cpu

# 7. 소스 코드 복사
COPY . .

# 8. 컨테이너 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
