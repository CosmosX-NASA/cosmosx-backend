# db/__init__.py
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from db_base import db_base

# 모델을 메타데이터에 등록하기 위해 import가 필요.
from model import UserTable, TodoTable  # noqa: F401


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
except Exception as e:
    print(f"테이블 생성 실패: {e}")

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
