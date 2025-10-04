from openai import NotFoundError
from sqlalchemy.orm import Session

from dto.hypothesis_dto import HypothesisAiResponse
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

    def get_by_id(self, hypo_id: int) -> Hypothesis:
        foundHypo = self.db.query(Hypothesis).filter(Hypothesis.id == hypo_id).first()
        if foundHypo is None:
            raise NotFoundError("Hypothesis not found")
        return foundHypo

    def update(self, ai_response: HypothesisAiResponse, hypo_id : int):
        hypo = self.get_by_id(hypo_id)
        hypo.statement = ai_response.statement
        hypo.usage = ai_response.usage
        hypo.evidence = ai_response.evidence
        hypo.status = "DONE"
        self.db.commit()
        self.db.refresh(hypo)  # 최신 상태 반영
        return hypo

    def count_by_user_id_and_status(self, status: str, user_id: int) -> int:
        return (
            self.db.query(Hypothesis)
            .filter(
                Hypothesis.status == status,
                Hypothesis.user_id == user_id
            )
            .count()
        )
