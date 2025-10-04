import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool


from db.db_base import db_base
from db.db import get_db_session
from model import ResearchGap
from service.research_gaps_service import ResearchGapsService
from repository.research_gaps_repository import ResearchGapsRepository
from dto.research_gaps_dto import ResearchItemGapResponse
from main import app


@pytest.fixture(scope="function")
def sample_gaps(db_session):
    gaps = [
        ResearchGap(
            type="CONCEPTUAL",
            content="디지털 신뢰 요소 정의 불명확",
            evidence="가설 추출 근거1",
            research_title="The Role of Digital Trust in Consumer Behavior",
            research_id=1
        ),
        ResearchGap(
            type="CONCEPTUAL",
            content="조직학습 구성요소 통일 안됨",
            evidence="가설 추출 근거2",
            research_title="Revisiting Organizational Learning Frameworks",
            research_id=2
        ),
        ResearchGap(
            type="METHODLOGICAL",
            content="횡단적 조사에 의존하여 인과검증 불가",
            evidence="가설 추출 근거3",
            research_title="User Acceptance of Mobile Health Apps",
            research_id=3
        )
    ]
    db_session.add_all(gaps)
    db_session.commit()
    for gap in gaps:
        db_session.refresh(gap)
    return gaps


def test_grouped_gaps_service(db_session, sample_gaps):
    repository = ResearchGapsRepository(db_session)
    service = ResearchGapsService(repository)

    research_ids = [1, 2, 3]
    grouped_gaps = service.get_grouped_gaps(research_ids)

    types = [group.type for group in grouped_gaps]
    assert set(types) == {"CONCEPTUAL", "METHODLOGICAL"}

    conceptual_group = next(g for g in grouped_gaps if g.type == "CONCEPTUAL")
    assert len(conceptual_group.researchs) == 2
    assert all(isinstance(r, ResearchItemGapResponse) for r in conceptual_group.researchs)

    method_group = next(g for g in grouped_gaps if g.type == "METHODLOGICAL")
    assert len(method_group.researchs) == 1


def test_api_research_gaps(db_session, sample_gaps):

    # dependency override 설정
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        # 요청
        response = client.get("/api/researchs/gaps?researchsIds=1,2,3")
        assert response.status_code == 200
        data = response.json()

        # gaps 키 존재 확인
        assert "gaps" in data

        # type별 그룹 개수
        types = {g["type"] for g in data["gaps"]}
        assert types == {"CONCEPTUAL", "METHODLOGICAL"}

        # 연구 개수 확인
        for g in data["gaps"]:
            if g["type"] == "CONCEPTUAL":
                assert len(g["researchs"]) == 2
            if g["type"] == "METHODLOGICAL":
                assert len(g["researchs"]) == 1
    finally:
        app.dependency_overrides.clear()