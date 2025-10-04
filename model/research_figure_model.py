from __future__ import annotations

from sqlalchemy import Column, String, Integer
from db.db_base import db_base


class Figure(db_base):
    __tablename__ = "figure"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="논문 PK")
    url = Column(String, unique=True, nullable=False, comment = "이미지 url")
    caption = Column(String, unique=True, nullable=False, comment = "이미지 캡션")
    research_id = Column(Integer, nullable=False, comment= "연구 논문 id")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, data: dict) -> Figure:
        return cls(**data)
