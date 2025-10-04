# service/research_gaps_service.py
from typing import List, Dict
from model import ResearchGap
from dto.research_gaps_dto import ResearchItemGapResponse, ResearchGapGroupResponse

class ResearchGapsService:
    def __init__(self, repository):
        self.repository = repository

    def get_grouped_gaps(self, research_ids: List[int]) -> List[ResearchGapGroupResponse]:
        gaps: List[ResearchGap] = self.repository.get_by_research_ids(research_ids)

        grouped: Dict[str, List[ResearchItemGapResponse]] = {}
        for gap in gaps:
            if gap.type not in grouped:
                grouped[gap.type] = []
            grouped[gap.type].append(ResearchItemGapResponse.from_model(gap))


        # DTO 변환
        result = [
            ResearchGapGroupResponse(type=gap_type, researchs=items)
            for gap_type, items in grouped.items()
        ]

        return result
