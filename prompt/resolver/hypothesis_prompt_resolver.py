from typing import List
from pathlib import Path

from prompt_toolkit.key_binding.bindings.named_commands import reverse_search_history

from model import ResearchWithGaps


class HypothesisPromptResolver:

    def _load_prompt(self, filename: str) -> str:
        project_root = Path(__file__).resolve().parents[2]  # resolver -> prompt -> FastAPIProject 루트
        full_path = project_root / filename
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        return full_path.read_text(encoding="utf-8")

    def resolve_hypothesis_create_prompt(self, research_with_gaps : List[ResearchWithGaps]) -> str:
        prompt_template = self._load_prompt("prompt/create_hypothesis_prompt.txt")
        research_with_gaps = ' '.join([research_with_gap.get_prompt_summary() for research_with_gap in research_with_gaps])
        prompt = prompt_template.format(research_and_gaps=research_with_gaps)
        print(prompt) # 로깅
        return prompt

    def resolve_specify_question_prompt(self, search:str) -> str:
        prompt_template = str(self._load_prompt("prompt/specify_question_prompt.txt"))
        prompt = prompt_template.format(search=search)
        print(prompt)
        return prompt

