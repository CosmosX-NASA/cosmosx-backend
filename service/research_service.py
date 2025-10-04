from dto.research_dto import ResearchResponses
from repository.figure_repository import FigureRepository
from repository.reserach_repository import ResearchRagRepository


class ResearchService:
    def __init__(self, figure_repository: FigureRepository, research_repository: ResearchRagRepository):
        self.figure_repository = figure_repository
        self.research_repository = research_repository

    def find_research_by_rag(self, search: str, page_size: int) -> ResearchResponses:
        found_researchs = self.research_repository.find_by_user_search(search=search, pageSize=page_size)
        research_ids = [r.id for r in found_researchs]
        figures = self.figure_repository.get_by_researchs(research_ids)
        return ResearchResponses.from_model(researchs=found_researchs, all_figures=figures)
