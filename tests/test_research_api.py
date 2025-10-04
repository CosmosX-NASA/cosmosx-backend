import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from main import app
from db.db import get_db_session
from model import Research, Figure


@pytest.fixture
def override_db_session(db_session):
    """
    FastAPI의 get_db_session 의존성을 in-memory DB로 대체
    """
    def _override():
        return db_session
    app.dependency_overrides[get_db_session] = _override
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def setup_figures(db_session):
    """
    실제 DB에 Figure 데이터를 넣는 fixture
    """
    figures = [
        Figure.create({"url": "https://example.com/img1.png", "caption": "fig1", "research_id": 1}),
        Figure.create({"url": "https://example.com/img2.png", "caption": "fig2", "research_id": 1}),
        Figure.create({"url": "https://example.com/img3.png", "caption": "fig3", "research_id": 2}),
    ]
    for f in figures:
        db_session.add(f)
    db_session.commit()


def test_get_research(override_db_session, setup_figures, monkeypatch):
    """
    /api/researchs 엔드포인트 단위 테스트
    - research_repository를 mock
    - figure_repository는 실제 DB 사용
    """
    from repository.research_repository import ResearchRagRepository

    mock_research_repo = MagicMock(spec=ResearchRagRepository)

    mock_researchs = [
        Research(
            id=1,
            pmc_id="PMC111",
            title="Test Research 1",
            journal="Journal A",
            doi="http://test1.com",
            author="Author 1",
            release_date="2020-01-01",
            brief_summary="Brief 1",
            overall_summary="Overall 1",
            methods="Methods 1",
            results="Results 1"
        ),
        Research(
            id=2,
            pmc_id="PMC222",
            title="Test Research 2",
            journal="Journal B",
            doi="http://test2.com",
            author="Author 2",
            release_date="2020-02-02",
            brief_summary="Brief 2",
            overall_summary="Overall 2",
            methods="Methods 2",
            results="Results 2"
        ),
    ]

    mock_research_repo.find_by_user_search.return_value = mock_researchs

    monkeypatch.setattr(
        "apis.research_api.ResearchRagRepository",
        lambda db: mock_research_repo
    )

    client = TestClient(app)

    response = client.get("/api/researchs", params={"search": "medical", "pageSize": 10})

    assert response.status_code == 200
    data = response.json()

    assert "researchs" in data
    assert len(data["researchs"]) == 2

    r1 = data["researchs"][0]
    assert r1["id"] == 1
    assert len(r1["figures"]) == 2  # research_id=1 → fig1, fig2

    r2 = data["researchs"][1]
    assert r2["id"] == 2
    assert len(r2["figures"]) == 1  # research_id=2 → fig3

    mock_research_repo.find_by_user_search.assert_called_once_with(
        search="medical",
        pageSize=10
    )
