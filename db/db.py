# db/__init__.py
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from .db_base import db_base
import subprocess
import time

# 모델을 메타데이터에 등록하기 위해 import가 필요.
from model import Research, Figure, ResearchGap, Hypothesis, HypothesisResearch  # noqa: F401


# 2. 데이터베이스 URL 설정 using absolute path
DB_PATH = os.path.join(os.getcwd(), "Database.db")
DB_URL = f'sqlite:///{DB_PATH}'

engine = create_engine(DB_URL, connect_args={
                       "check_same_thread": False}, echo=False)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# 4. 테이블 생성
try:
    db_base.metadata.create_all(engine)
    print("테이블 생성 성공")
    print(f"Database path: {DB_PATH}")
    print("현재 metadata 등록된 테이블:", db_base.metadata.tables.keys())
except Exception as e:
    print(f"테이블 생성 실패: {e}")

try:
    time.sleep(1)  # 잠시 대기
    populate_sql_path = os.path.join(os.getcwd(), "populate.sql")
    if os.path.exists(populate_sql_path):
        with open(populate_sql_path, "rb") as sql_file:
            subprocess.run([
                "sqlite3",
                DB_PATH,
            ], stdin=sql_file, check=True)
        print("기본 데이터 삽입 완료(1/2)")
    else:
        print(f"populate.sql 파일을 찾을 수 없습니다: {populate_sql_path}")

    time.sleep(1)  # 잠시 대기
    populate_sql_path = os.path.join(os.getcwd(), "insert_research_gaps.sql")
    if os.path.exists(populate_sql_path):
        with open(populate_sql_path, "rb") as sql_file:
            subprocess.run([
                "sqlite3",
                DB_PATH,
            ], stdin=sql_file, check=True)
        print("기본 데이터 삽입 완료(2/2)")
    else:
        print(f"insert_research_gaps.sql 파일을 찾을 수 없습니다: {populate_sql_path}")
except Exception as e:
    print(f"기본 데이터 삽입 실패: {e}")

# 5. 세션 생성기 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 6. 의존성으로 사용할 세션 생성 함수


def get_db_session():
    """
    Dependency
    try-finally 블록을 통해 db 연결을 종료하거나 문제가 생겼을 때 무조건 close.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# jdbc:sqlite:/Users/coli/Desktop/projects/fast-api-practice/Database.db
