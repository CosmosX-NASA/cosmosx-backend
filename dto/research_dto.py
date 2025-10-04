from typing import List, Optional, Dict
from pydantic import BaseModel
from model import Figure, Research

class FigureResponse(BaseModel):
    url: str
    caption: str

    @classmethod
    def from_model(cls, figure: Figure):
        return cls(
            url = figure.url,
            caption = figure.caption
        )

class ResearchResponse(BaseModel):
    id: int
    title: str
    journal: str
    brief_summary: str
    overall_summary: str
    figures: Optional[List[FigureResponse]] = None
    methods: str
    results: str

    @classmethod
    def from_model(cls, research: Research, figures: List[Figure]):
        return cls(
            id=research.id,
            title=research.title,
            journal=research.journal,
            brief_summary=research.brief_summary,
            overall_summary=research.overall_summary,
            methods=research.methods,
            results=research.results,
            figures = (
                [FigureResponse.from_model(fig) for fig in figures]
                if figures
                else None
            )
        )


class ResearchResponses(BaseModel):
    researchs: List[ResearchResponse]

    @classmethod
    def from_model(cls, researchs: List[Research], all_figures: List[Figure]):
        figures_map: Dict[int, List[Figure]] = {}
        for fig in all_figures:
            figures_map.setdefault(fig.research_id, []).append(fig)

        # 각 Research에 맞는 Figures 연결
        response_list = [
            ResearchResponse.from_model(r, figures_map.get(r.id, []))
            for r in researchs
        ]
        return cls(researchs=response_list)
