from sqlalchemy.orm import Session
from model import Hypothesis

class HypothesisRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_userId(self, userid : int) -> list[Hypothesis]:
        return (
            self.db.query(Hypothesis)
                .filter(Hypothesis.user_id == userid)
                .all()
        )


    def save(self, hypo: Hypothesis) -> Hypothesis:
        self.db.add(hypo)
        self.db.flush()
        return hypo

