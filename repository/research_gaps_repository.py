from sqlalchemy.orm import Session
from typing import List, Dict
from model import Research, ResearchGap, ResearchWithGaps


class ResearchGapsRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_ids(self, ids: List[int]) -> list[ResearchGap]:
        return (self.db.query(ResearchGap)
                .filter(ResearchGap.id.in_(ids))
                .all())

    def get_research_with_gaps(self, gaps_ids: List[int]) -> List[ResearchWithGaps]:
        results = (
            self.db.query(Research, ResearchGap)
            .join(Research, Research.id == ResearchGap.research_id)
            .filter(ResearchGap.id.in_(gaps_ids))
            .all()
        )

        # Research별로 gaps 묶기
        grouped: Dict[int, ResearchWithGaps] = {}

        for research, gap in results:
            if research.id not in grouped:
                grouped[research.id] = ResearchWithGaps(research=research, gaps=[])
            grouped[research.id].gaps.append(gap)

        return list(grouped.values())