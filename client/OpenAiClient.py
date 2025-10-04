import os
from dotenv import load_dotenv
from openai import OpenAI
from model import ResearchWithGaps
from typing import List
from prompt.resolver.hypothesis_prompt_resolver import HypothesisPromptResolver

load_dotenv()


class OpenAiClient:
    def __init__(self, prompt_resolver: HypothesisPromptResolver, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.prompt_resolver = prompt_resolver
        self.client = OpenAI(api_key=self.api_key)

    def create_hypothesis(self, research_with_gaps : List[ResearchWithGaps]) -> str:
        prompt= self.prompt_resolver.resolve(research_with_gaps)
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
