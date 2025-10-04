import pytest
from unittest.mock import MagicMock

from model import Research, Figure
from repository.figure_repository import FigureRepository
from service.research_service import ResearchService


@pytest.fixture
def figure_repo_with_data(db_session):
    """
    실제 DB에 Figure 데이터를 넣고 동작하는 FigureRepository fixture
    """
    figures = [
        Figure.create({"url": "https://example.com/img1.png", "caption": "fig1", "research_id": 1}),
        Figure.create({"url": "https://example.com/img2.png", "caption": "fig2", "research_id": 1}),
        Figure.create({"url": "https://example.com/img3.png", "caption": "fig3", "research_id": 2}),
    ]
    for f in figures:
        db_session.add(f)
    db_session.commit()

    return FigureRepository(db_session)


def test_find_research_by_rag(figure_repo_with_data):
    """
    ResearchService.find_research_by_rag 단위 테스트
    - research_repository는 mock
    - figure_repository는 실제 fixture 사용
    """
    # ✅ 1. Mock 연구 데이터
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

    # ✅ 2. research_repository Mocking
    mock_research_repo = MagicMock()
    mock_research_repo.find_by_user_search.return_value = mock_researchs

    # ✅ 3. Service 주입
    service = ResearchService(
        figure_repository=figure_repo_with_data,
        research_repository=mock_research_repo
    )

    # ✅ 4. 호출
    response = service.find_research_by_rag(search="medical", page_size=10)

    # ✅ 5. 검증
    assert len(response.researchs) == 2

    r1 = response.researchs[0]
    assert r1.id == 1
    assert len(r1.figures) == 2  # research_id=1 → fig1, fig2

    r2 = response.researchs[1]
    assert r2.id == 2
    assert len(r2.figures) == 1  # research_id=2 → fig3

    mock_research_repo.find_by_user_search.assert_called_once_with(
        search="medical",
        pageSize=10
    )
