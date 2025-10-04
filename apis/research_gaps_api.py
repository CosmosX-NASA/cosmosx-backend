from fastapi import APIRouter, status, Query, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session

from repository.research_gaps_repository import ResearchGapsRepository
from service.research_gaps_service import ResearchGapsService
from dto.research_gaps_dto import ResearchGapsGroupedResponse
import db.db as db_module

router = APIRouter(
    prefix="/api/reserachs/gaps",
    tags=["reserach-gaps"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}}

)

# DB 세션 의존성
def get_db():
    db = db_module.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get(
    "",
    summary = "연구논문에 해당하는 Reserch Gap 반환",
    response_model = ResearchGapsGroupedResponse
)
def get_grouped_research_gaps(
        reserachIds : List[int] = Query(..., description="연구 논문 id 리스트"),
        db: Session = Depends(get_db)
):
    repository = ResearchGapsRepository(db)
    research_gaps_service = ResearchGapsService(repository)
    grouped_gaps = research_gaps_service.get_gaps(reserachIds)
    return ResearchGapsGroupedResponse(gaps=grouped_gaps)
