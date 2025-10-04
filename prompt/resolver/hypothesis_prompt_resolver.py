from typing import List
from pathlib import Path
from model import ResearchWithGaps


class HypothesisPromptResolver:

    def _load_prompt(self, filename: str) -> str:
        project_root = Path(__file__).resolve().parents[2]  # resolver -> prompt -> FastAPIProject 루트
        full_path = project_root / filename
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        return full_path.read_text(encoding="utf-8")

    def resolve(self, research_with_gaps : List[ResearchWithGaps]) -> str:
        prompt = self._load_prompt("prompt/create_hypothesis_prompt.txt")
        print(prompt) # 로깅
        return prompt

