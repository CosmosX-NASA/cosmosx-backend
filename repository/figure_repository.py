from sqlalchemy.orm import Session
from typing import List
from model import Figure

class FigureRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_researchs(self,  research_ids: List[int]) -> list[Figure]:
        return (
            self.db.query(Figure)
                .filter(Figure.research_id.in_(research_ids))
                .all()
        )