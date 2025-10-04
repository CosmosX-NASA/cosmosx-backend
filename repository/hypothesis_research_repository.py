from sqlalchemy.orm import Session
from model import HypothesisResearch

class HypothesisResearchRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_hypothesis_id(self, hypothesis_id : int) -> list[HypothesisResearch]:
        return (self.db.query(HypothesisResearch)
                .filter(HypothesisResearch.hypothesis_id.is_(hypothesis_id))
                .all())
