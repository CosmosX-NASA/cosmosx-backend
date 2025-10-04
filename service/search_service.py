from client.open_ai_client import OpenAiClient
from dto.search_dto import SpecifySearchResponse


class SearchService:
    def __init__(self, openai_client: OpenAiClient):
        self.client = openai_client

    def specify(self, search: str) -> SpecifySearchResponse:
        response = self.client.specify_question(search=search)
        return SpecifySearchResponse(specify_question=response)
