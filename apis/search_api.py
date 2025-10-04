from fastapi import APIRouter, status, Query
from dto.search_dto import SpecifySearchResponse
from service.search_service import SearchService
from client.open_ai_client import OpenAiClient

router = APIRouter(
    prefix="/api/searchs",
    tags=[""],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}}
)

@router.get(
    "",
    summary="검색 키워드 구체화",
    response_model= SpecifySearchResponse
)
def search_specify(search: str = Query(..., description="유저가 입력한 검색어")) :
    openai_client = OpenAiClient()
    search_service = SearchService(openai_client)
    return search_service.specify(search=search)