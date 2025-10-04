from sqlalchemy.orm import Session
from model import ResearchGap
from typing import List


class ResearchGapsRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_ids(self, ids: List[int]) -> list[ResearchGap]:
        return (self.db.query(ResearchGap)
                .filter(ResearchGap.id.in_(ids))
                .all())