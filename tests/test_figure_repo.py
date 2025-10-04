from model import Figure
from repository.figure_repository import FigureRepository

def test_get_by_researchs(db_session):
    figures_data = [
        {"url": "https://example.com/img1.png", "caption": "img1", "research_id": 1},
        {"url": "https://example.com/img2.png", "caption": "img2", "research_id": 1},
        {"url": "https://example.com/img3.png", "caption": "img3", "research_id": 2},
        {"url": "https://example.com/img4.png", "caption": "img4", "research_id": 3},
    ]

    for data in figures_data:
        figure = Figure.create(data)
        db_session.add(figure)
    db_session.commit()

    repo = FigureRepository(db_session)

    result = repo.get_by_researchs([1, 3])

    assert len(result) == 3  # research_id 1 → 2개, research_id 3 → 1개
    research_ids = {fig.research_id for fig in result}
    assert research_ids == {1, 3}

    urls = {fig.url for fig in result}
    assert urls == {
        "https://example.com/img1.png",
        "https://example.com/img2.png",
        "https://example.com/img4.png",
    }


def test_get_by_researchs_empty(db_session):
    repo = FigureRepository(db_session)
    result = repo.get_by_researchs([99, 100])

    assert result == []
