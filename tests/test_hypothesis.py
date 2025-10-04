# tests/test_research_gaps.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from repository.hypothesis_repository import HypothesisRepository
from repository.hypothesis_reserach_repository import HypothesisResearchRepository
from repository.research_gaps_repository import ResearchGapsRepository
from prompt.resolver.hypothesis_prompt_resolver import HypothesisPromptResolver
from client.open_ai_client import OpenAiClient
from service.hypothesis_service import HypothesisService
from dto.hypothesis_dto import HypothesisResponses
from model import Hypothesis, HypothesisResearch

from db.db import get_db_session
from main import app

@pytest.fixture
def sample_hypotheses(db_session):
    data = [
        Hypothesis(id=1, status="DONE",
                   statement="가설1", usage="A", evidence="E1", user_id=1),
        Hypothesis(id=2, status="DONE",
                   statement="가설2", usage="B", evidence="E2", user_id=1),
        Hypothesis(id=3, status="PENDING",
                   statement=None, usage=None, evidence=None, user_id=1),
        Hypothesis(id=4, status="DONE",
                   statement="다른 유저", usage="X", evidence="Y", user_id=2),
    ]
    db_session.add_all(data)
    db_session.commit()
    return data

@pytest.fixture
def sample_hypothesis_researches(db_session):
    data = [
        HypothesisResearch(id=1, url="https://example.com/research1", hypothesis_id=1),
        HypothesisResearch(id=2, url="https://example.com/research2", hypothesis_id=1),
        HypothesisResearch(id=3, url="https://example.com/research3", hypothesis_id=1),
        HypothesisResearch(id=4, url="https://example.com/research4", hypothesis_id=2),
        HypothesisResearch(id=5, url="https://example.com/research5", hypothesis_id=2),
    ]
    db_session.add_all(data)
    db_session.commit()
    return data

def test_get_my_hypothesis_service(db_session, sample_hypotheses, sample_hypothesis_researches):
    # Given
    hypothesis_repo = HypothesisRepository(db_session)
    hypothesis_research_repo = HypothesisResearchRepository(db_session)
    research_gaps_repo = ResearchGapsRepository(db_session)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver=prompt_resolver)
    service = HypothesisService(hypothesis_repo, hypothesis_research_repo, research_gaps_repo, openai_client)

    # When
    result = service.get_my_hypothesis(user_id=1)

    # Then
    assert isinstance(result, HypothesisResponses)
    assert len(result.hypotheses) == 3

    h1 = next(h for h in result.hypotheses if h.id == 1)
    assert h1.status == "DONE"
    assert h1.statement == "가설1"
    assert len(h1.research_urls) == 3
    assert sorted(h1.research_urls) == sorted([
        "https://example.com/research1",
        "https://example.com/research2",
        "https://example.com/research3"
    ])

    h2 = next(h for h in result.hypotheses if h.id == 2)
    assert h2.status == "DONE"
    assert h2.statement == "가설2"
    assert len(h2.research_urls) == 2
    assert sorted(h2.research_urls) == sorted([
        "https://example.com/research4",
        "https://example.com/research5",
    ])

    h3 = next(h for h in result.hypotheses if h.id == 3)
    assert h3.status == "PENDING"
    assert h3.statement is None
    assert h3.research_urls is None

    # ✅ user_id=2 (ID=4 Hypothesis)는 포함되지 않아야 함
    ids_returned = [h.id for h in result.hypotheses]
    assert 4 not in ids_returned


def test_get_my_hypothesis(
    db_session,
    sample_hypotheses,
    sample_hypothesis_researches,
):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/me?userId=1")
        assert response.status_code == 200

        data = response.json()
        assert "hypotheses" in data
        assert len(data["hypotheses"]) == 3  # user_id = 1 인 가설 3개

        hypo1 = next(h for h in data["hypotheses"] if h["id"] == 1)
        assert hypo1["status"] == "DONE"
        assert hypo1["statement"] == "가설1"
        assert len(hypo1["research_urls"]) == 3

        hypo3 = next(h for h in data["hypotheses"] if h["id"] == 3)
        assert hypo3["status"] == "PENDING"
        assert hypo3["statement"] is None
        assert hypo3["research_urls"] is None
    finally:
        app.dependency_overrides.clear()