from __future__ import annotations

from sqlalchemy import Column, String, Integer
from db.db_base import db_base


class HypothesisResearch(db_base):
    __tablename__ = "hypothesis_reserachs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="가설 PK")
    url = Column(String, nullable=False, comment = "가설 관련 연구 논문 url")
    hypothesis_id = Column(Integer, nullable=False, comment="가설 id")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, data: dict) -> HypothesisResearch:
        return cls(**data)
