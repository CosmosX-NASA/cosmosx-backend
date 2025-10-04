from __future__ import annotations

from sqlalchemy import Column, String, Integer
from db.db_base import db_base


class Hypothesis(db_base):
    __tablename__ = "hypothesis"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="가설 PK")
    status = Column(String, nullable=False, comment = "가설 생성 상태")
    statement = Column(String, nullable=True, comment = "가설")
    usage = Column(String, nullable=True, comment= "가설 활용 방안")
    evidence = Column(String, nullable=True, comment= "가설 추출 근거")
    user_id = Column(Integer, nullable=False, comment="유저 id")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, data: dict) -> Hypothesis:
        return cls(**data)
