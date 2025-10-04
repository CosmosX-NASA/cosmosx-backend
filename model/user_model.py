from __future__ import annotations

from PIL.TiffImagePlugin import DATE_TIME
from sqlalchemy import Column, String, Integer
from db.db_base import db_base


class Research(db_base):
    __tablename__ = "research"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="논문 PK")
    title = Column(String(255), nullable=False, comment="논문 제목")
    journal = Column(String(255), nullable=False, comment="저널 이름")
    doi = Column(String, nullable=False, comment="논문 url")
    author = Column(String(255), nullable=False, comment="논문 저자")
    release_date = Column(DATE_TIME, nullable=False, comment="발간 일자")
    brief_summary = Column(String, nullable=False, comment="짧은 요약")
    overall_summary = Column(String, nullable=False, comment="전반 요약")
    methods = Column(String, nullable=False, comment="연구 방법")
    results = Column(String, nullable=False, comment="연구 결과")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, data: dict) -> Research:
        return cls(**data)
