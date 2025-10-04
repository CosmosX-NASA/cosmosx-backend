from typing import List, Optional
from pydantic import BaseModel
from model import Hypothesis, HypothesisResearch

#Request
class HypothesisCreateRequest(BaseModel):
    userId: Optional[int] = None
    gapIds: List[int]

#Response

class HypothesisAiResponse(BaseModel):
    statement: str
    usage: str
    evidence: str


class HypothesisResponse(BaseModel):
    id: int
    status: str
    statement: Optional[str] = None
    usage: Optional[str] = None
    evidence: Optional[str] = None
    research_urls: Optional[List[str]] = None

    @classmethod
    def from_model(cls, hypo: Hypothesis, hypo_researchs : list[HypothesisResearch]):
        return cls(
            id=hypo.id,
            status=hypo.status,
            statement=hypo.statement,
            usage=hypo.usage,
            evidence=hypo.evidence,
            research_urls=(
                [research.url for research in hypo_researchs]
                if hypo_researchs
                else None
            )
        )

class HypothesisResponses(BaseModel):
    hypotheses: List[HypothesisResponse]

class HypothesisCreateResponse(BaseModel):
    userId: int