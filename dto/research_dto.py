from typing import List, Optional
from pydantic import BaseModel

class FigureResponse(BaseModel):
    url: str
    caption: str


class ResearchResponse(BaseModel):
    id: int
    title: str
    journal: str
    brief_summary: str
    overall_summary: str
    figures: Optional[List[FigureResponse]] = None
    methods: str
    results: str


class ResearchResponses(BaseModel):
    researchs: List[ResearchResponse]
