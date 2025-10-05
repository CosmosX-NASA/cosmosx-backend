# 1. Base Image
FROM python:3.11-slim

# 2. 빌드 아규먼트 정의
ARG OPENAI_API_KEY

# 3. 환경 변수로 설정
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV PYTHONUNBUFFERED=1

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 6. pip 업그레이드
RUN pip install --upgrade pip

# 7. pyproject.toml 복사 및 의존성 설치
COPY pyproject.toml .

# 8. pip로 pyproject.toml의 모든 의존성 설치
# --no-cache-dir: 캐시를 저장하지 않아 이미지 크기 감소
RUN pip install --no-cache-dir -e .

# 9. 애플리케이션 코드 복사
COPY . .

# 10. 컨테이너 실행 시 uvicorn 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]