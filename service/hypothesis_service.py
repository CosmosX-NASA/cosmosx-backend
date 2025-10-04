from dto.hypothesis_dto import HypothesisResponse, HypothesisResponses
from repository.hypothesis_repository import HypothesisRepository
from model import Hypothesis
from repository.hypothesis_reserach_repository import HypothesisResearchRepository


class HypothesisService:
    def __init__(
            self,
            hypothesis_repository: HypothesisRepository,
            hypothesis_research_repository: HypothesisResearchRepository
    ):
        self.hypothesis_repository = hypothesis_repository
        self.hypothesis_research_repository = hypothesis_research_repository

    def get_my_hypothesis(self, user_id: int) -> HypothesisResponses:
        my_hypothesis : list[Hypothesis] = self.hypothesis_repository.get_by_userId(userid = user_id)

        responses: list[HypothesisResponse] = []

        for hypo in my_hypothesis:
            hypo_researchs = self.hypothesis_research_repository.get_by_hypothesis_id(hypothesis_id=hypo.id)
            responses.append(HypothesisResponse.from_model(hypo, hypo_researchs))
        return HypothesisResponses(hypotheses=responses)
