from sqlalchemy.orm import Session
from model import HypothesisResearch

class HypothesisResearchRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_hypothesis_id(self, hypothesis_id : int) -> list[HypothesisResearch]:
        return (self.db.query(HypothesisResearch)
                .filter(HypothesisResearch.hypothesis_id.is_(hypothesis_id))
                .all())

    def save(self, hypothesis_id : int, url: str) -> HypothesisResearch:
        entity = HypothesisResearch.create({
            "url": url,
            "hypothesis_id": hypothesis_id
        })
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)  # id 등 자동 생성 값 반영
        return entity