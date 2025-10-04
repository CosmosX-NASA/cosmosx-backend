from typing import List
from pathlib import Path
from model import ResearchWithGaps


class HypothesisPromptResolver:

    def _load_prompt(filename: str) -> str:
        path = Path(filename)
        return path.read_text(encoding="utf-8")

    def resolve(self, research_with_gaps : List[ResearchWithGaps]) -> str:
        prompt = self._load_prompt("../create_hypothesis_prompt.txt")
        print(prompt) # 로깅
        return prompt

