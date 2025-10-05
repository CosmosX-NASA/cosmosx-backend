from pydantic import BaseModel
from typing import List

from model import ResearchGap, Research

class ResearchWithGaps(BaseModel):
    research : Research
    gaps: List[ResearchGap]

    model_config = {
        "arbitrary_types_allowed": True
    }

    def get_prompt_summary(self) -> str:
        gaps_json = [
            gap.model_dump() if hasattr(gap, "model_dump") else {
                "id": gap.id,
                "type": gap.type,
                "content": gap.content,
                "evidence": gap.evidence,
                "research_title": gap.research_title,
                "research_id": gap.research_id
            }
            for gap in self.gaps
        ]
        return """
            reserach summary: {summary},
            research methods: {methods},
            research result: {result},
            reserach gaps: {gaps}
        """.format(
            summary=self.research.overall_summary,
            methods=self.research.methods,
            result= self.research.results,
            gaps=gaps_json
        )
