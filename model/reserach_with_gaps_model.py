from pydantic import BaseModel
from typing import List

from model import ResearchGap, Research

class ResearchWithGaps(BaseModel):
    research : Research
    gaps: List[ResearchGap]
