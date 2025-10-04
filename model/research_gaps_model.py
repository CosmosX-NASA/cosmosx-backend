from __future__ import annotations

from sqlalchemy import Column, String, Integer
from db.db_base import db_base


class ResearchGap(db_base):
    __tablename__ = "research_gaps"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="논문 간극 PK")
    type = Column(String(20), nullable=False, comment = "간극 type ex) CONECEPTUAL")
    content = Column(String, nullable=False, comment = "간극 내용")
    evidence = Column(Integer, nullable=False, comment= "간극 추출 근거")
    research_title = Column(String(255), nullable=False, comment= "논문 제목")
    research_id = Column(Integer, nullable=False, comment= "연구 논문 id")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, data: dict) -> ResearchGap:
        return cls(**data)
