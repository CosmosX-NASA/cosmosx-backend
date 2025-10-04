from pydantic import BaseModel
from typing import List
from model import ResearchGap

class ResearchItemGapResponse(BaseModel):
    id: int
    title: str
    content: str
    evidence: str

    @classmethod
    def from_model(cls, gap: ResearchGap):
        return cls(
            id=gap.research_id,
            title=gap.research_title,
            content=gap.content,
            evidence=gap.evidence
        )

class ResearchGapGroupResponse(BaseModel):
    type: str
    researchs: List[ResearchItemGapResponse]

class ResearchGapsGroupedResponse(BaseModel):
    gaps: List[ResearchGapGroupResponse]
