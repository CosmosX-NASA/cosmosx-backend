from typing import List
from dto.hypothesis_dto import HypothesisResponses, HypothesisResponse, HypothesisCreateRequest, \
    HypothesisCreateResponse, HypothesisDoneCountResponse
from repository.hypothesis_repository import HypothesisRepository
from repository.hypothesis_research_repository import HypothesisResearchRepository
from repository.research_gaps_repository import ResearchGapsRepository
from client.open_ai_client import OpenAiClient
from model import Hypothesis, ResearchWithGaps
import threading

from repository.research_repository import ResearchRagRepository

_user_id_counter = 1
_user_id_lock = threading.Lock()

class HypothesisService:
    def __init__(
            self,
            hypothesis_repository: HypothesisRepository,
            hypothesis_research_repository: HypothesisResearchRepository,
            research_gaps_repository: ResearchGapsRepository,
            openai_client: OpenAiClient,
            research_repository: ResearchRagRepository
    ):
        self.hypothesis_repository = hypothesis_repository
        self.research_repository = research_repository
        self.hypothesis_research_repository = hypothesis_research_repository
        self.research_gaps_repository = research_gaps_repository
        self.openai_client = openai_client

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

    def _save_hypothesis_research(self, hypothesis:str, hypo_id: int):
        related_research = self.research_repository.find_by_user_search(hypothesis, 3)
        for research in related_research:
            self.hypothesis_research_repository.save(hypo_id, research.doi)

    def _update_hypothesis_and_save_research(self, research_with_gaps : List[ResearchWithGaps], hypo_id: int):
        try:
            raw_hypothesis = self.openai_client.create_hypothesis(research_with_gaps)
            self.hypothesis_repository.update(ai_response=raw_hypothesis, hypo_id=hypo_id)
            self._save_hypothesis_research(raw_hypothesis.statement, hypo_id)
        except Exception as e:
            print(f"Failed to update hypothesis {hypo_id}: {e}")

    def create_hypothesis(self, request: HypothesisCreateRequest) -> HypothesisCreateResponse:
        user_id = request.userId
        gaps_id = request.gapIds
        research_with_gaps = self.research_gaps_repository.get_research_with_gaps(gaps_id)
        user_id = self._next_user_id(user_id)
        new_hypo = self._save_pending_hypo(user_id)

        #백그라운드로 실행
        thread = threading.Thread(
            target=self._update_hypothesis_and_save_research,
            args=(research_with_gaps, new_hypo.id),
            daemon=True
        )
        thread.start()

        return HypothesisCreateResponse(userId = user_id)

    def get_my_hypothesis(self, user_id: int) -> HypothesisResponses:
        my_hypothesis : list[Hypothesis] = self.hypothesis_repository.get_by_userId(userid = user_id)

        responses: list[HypothesisResponse] = []

        for hypo in my_hypothesis:
            hypo_researchs = self.hypothesis_research_repository.get_by_hypothesis_id(hypothesis_id=hypo.id)
            responses.append(HypothesisResponse.from_model(hypo, hypo_researchs))
        return HypothesisResponses(hypotheses=responses)

    def get_done_hypothesis_count(self, user_id: int) -> HypothesisDoneCountResponse:
        count = self.hypothesis_repository.count_by_user_id_and_status("DONE", user_id)
        return HypothesisDoneCountResponse(count = count)