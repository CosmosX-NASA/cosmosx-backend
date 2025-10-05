from fastapi import APIRouter, status, Query, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from db.db import get_db_session
from dto.hypothesis_dto import HypothesisResponses, HypothesisCreateResponse, HypothesisCreateRequest, \
    HypothesisDoneCountResponse
from repository.hypothesis_repository import HypothesisRepository
from repository.hypothesis_research_repository import HypothesisResearchRepository
from repository.research_gaps_repository import ResearchGapsRepository
from prompt.resolver.hypothesis_prompt_resolver import HypothesisPromptResolver
from client.open_ai_client import OpenAiClient
from repository.research_repository import ResearchRagRepository
from service.hypothesis_service import HypothesisService

router = APIRouter(
    prefix="/api/hypothesis",
    tags=["Hypothesis"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}}

)


@router.post(
    path="",
    summary = "Research Gap을 조합해 가설 만들기",
    response_model = HypothesisCreateResponse
)
def create_hypothesis(request: HypothesisCreateRequest = Body(...)):
    return HypothesisCreateResponse()

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
    research_gaps_repository = ResearchGapsRepository(db)
    research_repository = ResearchRagRepository(db)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver)

    hypothesis_service = HypothesisService(
        hypothesis_repository,
        hypothesis_research_repository,
        research_gaps_repository,
        openai_client,
        research_repository,
    )
    return hypothesis_service.get_my_hypothesis(userId)


@router.get(
    "/done",
    summary="생성 완료된 가설 목록 반환",
    response_model=HypothesisDoneCountResponse,
)
def get_done_hypothesis_count(
        userId: int = Query(..., description="유저 id"),
        db: Session = Depends(get_db_session),
):
    hypothesis_repository = HypothesisRepository(db)
    hypothesis_research_repository = HypothesisResearchRepository(db)
    research_gaps_repository = ResearchGapsRepository(db)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver)
    research_repository = ResearchRagRepository(db)

    hypothesis_service = HypothesisService(
        hypothesis_repository,
        hypothesis_research_repository,
        research_gaps_repository,
        openai_client,
        research_repository
    )
    return hypothesis_service.get_done_hypothesis_count(userId)
