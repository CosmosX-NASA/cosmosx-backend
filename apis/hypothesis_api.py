from fastapi import APIRouter, status, Query, Depends, HTTPException
from sqlalchemy.orm import Session

from db.db import get_db_session
from dto.hypothesis_dto import HypothesisResponses
from repository.hypothesis_repository import HypothesisRepository
from repository.hypothesis_reserach_repository import HypothesisResearchRepository
from service.hypothesis_service import HypothesisService

router = APIRouter(
    prefix="/api/hypothesis",
    tags=["Hypothesis"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}}

)

@router.get(
    "/me",
    summary="나의 가설 목록 반환",
    response_model=HypothesisResponses,
)
def get_my_hypothesis(
        userId: int = Query(..., description="유저 id"),
        db: Session = Depends(get_db_session),
):
    hypothesis_repository = HypothesisRepository(db)
    hypothesis_research_repository = HypothesisResearchRepository(db)
    hypothesis_service = HypothesisService(hypothesis_repository, hypothesis_research_repository)
    return hypothesis_service.get_my_hypothesis(userId)
