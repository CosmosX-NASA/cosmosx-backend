import os
from dotenv import load_dotenv
from openai import OpenAI
from model import ResearchWithGaps
from typing import List
from prompt.resolver.hypothesis_prompt_resolver import HypothesisPromptResolver
from dto.hypothesis_dto import HypothesisAiResponse
from util.json_decoder import JsonDecoder

load_dotenv()

class OpenAiClient:
    def __init__(self, prompt_resolver: HypothesisPromptResolver, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.prompt_resolver = prompt_resolver
        self.client = OpenAI(api_key=self.api_key)

    def create_hypothesis(self, research_with_gaps : List[ResearchWithGaps]) -> HypothesisAiResponse:
        prompt= self.prompt_resolver.resolve_hypothesis_create_prompt(research_with_gaps)
        response = self.client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_response = response.choices[0].message.content

        print("응답 : " + raw_response)

        decoded = JsonDecoder.decode(raw_response, HypothesisAiResponse)
        if decoded is None:
            raise ValueError(f"OpenAI response could not be parsed as HypothesisAiResponse: {raw_response}")
        return decoded

    def specify_question(self, search:str) -> str:
        user_prompt = self.prompt_resolver.resolve_specify_question_prompt(search)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw_response = response.choices[0].message.content
        print("응답 : " + raw_response)
        return raw_response

