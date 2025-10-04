from fastapi import APIRouter, status, Query, Depends, HTTPException
from sqlalchemy.orm import Session

from repository.research_gaps_repository import ResearchGapsRepository
from service.research_gaps_service import ResearchGapsService
from dto.research_gaps_dto import ResearchGapsGroupedResponse
from dto.research_dto import ResearchResponses
from db.db import get_db_session

router = APIRouter(
    prefix="/api/researchs",
    tags=["reserach"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}}

)

@router.get(
    "",
    summary="관심사별 Reserch RAG 탐색 후 반환",
    response_model=ResearchResponses
)
def get_research(
        search: str = Query(..., description="유저가 입력한 검색어"),
        pageSize: int = Query(..., description="요청한 아이템 수"),
        db: Session = Depends(get_db_session),
):
    return ResearchResponses(researchs=[])


@router.get(
    "/gaps",
    summary="연구논문에 해당하는 Reserch Gap 반환",
    response_model=ResearchGapsGroupedResponse
)
def get_grouped_research_gaps(
        researchsIds: str = Query(..., description="연구 논문 id 리스트"),
        db: Session = Depends(get_db_session),
):
    research_ids = [int(rid.strip()) for rid in researchsIds.split(",") if rid.strip().isdigit()]
    repository = ResearchGapsRepository(db)
    research_gaps_service = ResearchGapsService(repository)
    grouped_gaps = research_gaps_service.get_grouped_gaps(research_ids=research_ids)
    return ResearchGapsGroupedResponse(gaps=grouped_gaps)
