from typing import Optional, List
from dto.hypothesis_dto import HypothesisResponse, HypothesisResponses, HypothesisResponse, HypothesisCreateRequest, HypothesisCreateResponse
from repository import research_gaps_repository
from repository.hypothesis_repository import HypothesisRepository
from repository.hypothesis_reserach_repository import HypothesisResearchRepository
from repository.research_gaps_repository import ResearchGapsRepository
from model import Hypothesis
import threading

_user_id_counter = 1
_user_id_lock = threading.Lock()

class HypothesisService:
    def __init__(
            self,
            hypothesis_repository: HypothesisRepository,
            hypothesis_research_repository: HypothesisResearchRepository,
            research_gaps_repository: ResearchGapsRepository
    ):
        self.hypothesis_repository = hypothesis_repository
        self.hypothesis_research_repository = hypothesis_research_repository
        self.research_gaps_repository = research_gaps_repository

    def _next_user_id(self, user_id: int) -> int:
        if user_id is None:
            global _user_id_counter
            with _user_id_lock:
                user_id = _user_id_counter
                _user_id_counter += 1
        return user_id

    def _save_pending_hypo(self, user_id: int) -> Hypothesis:
        new_hypo = Hypothesis(
            status="PENDING",
            statement=None,
            usage=None,
            evidence=None,
            user_id=user_id
        )
        return self.hypothesis_repository.save(new_hypo)


    def create_hypothesis(self, request: HypothesisCreateRequest) -> HypothesisCreateResponse:
        user_id = request.user_id
        gaps_id = request.gapIds
        reserach_with_gaps = self.research_gaps_repository.get_research_with_gaps(gaps_id)
        user_id = self._next_user_id(user_id)
        new_hypo = self._save_pending_hypo(user_id)
        return HypothesisCreateResponse(user_id = user_id)

    def get_my_hypothesis(self, user_id: int) -> HypothesisResponses:
        my_hypothesis : list[Hypothesis] = self.hypothesis_repository.get_by_userId(userid = user_id)

        responses: list[HypothesisResponse] = []

        for hypo in my_hypothesis:
            hypo_researchs = self.hypothesis_research_repository.get_by_hypothesis_id(hypothesis_id=hypo.id)
            responses.append(HypothesisResponse.from_model(hypo, hypo_researchs))
        return HypothesisResponses(hypotheses=responses)
