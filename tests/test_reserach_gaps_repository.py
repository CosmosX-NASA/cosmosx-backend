from model import Research, ResearchGap, ResearchWithGaps
from repository.research_gaps_repository import ResearchGapsRepository

def test_get_research_with_gaps(db_session):
    # 테스트용 Research 및 ResearchGap 생성
    research1 = Research(
        id=1, pmc_id="PMC001", title="Research 1", journal="Journal 1", doi="doi1",
        author="Author 1", release_date="2025-10-04",
        brief_summary="Brief 1", overall_summary="Overall 1",
        methods="Methods 1", results="Results 1"
    )
    research2 = Research(
        id=2, pmc_id="PMC002", title="Research 2", journal="Journal 2", doi="doi2",
        author="Author 2", release_date="2025-10-05",
        brief_summary="Brief 2", overall_summary="Overall 2",
        methods="Methods 2", results="Results 2"
    )
    gap1 = ResearchGap(id=1, type="CONCEPTUAL", content="Gap 1", evidence="Evidence 1",
                       research_title="Research 1", research_id=1)
    gap2 = ResearchGap(id=2, type="METHOD", content="Gap 2", evidence="Evidence 2",
                       research_title="Research 1", research_id=1)
    gap3 = ResearchGap(id=3, type="CONCEPTUAL", content="Gap 3", evidence="Evidence 3",
                       research_title="Research 2", research_id=2)

    db_session.add_all([research1, research2, gap1, gap2, gap3])
    db_session.commit()

    repo = ResearchGapsRepository(db=db_session)
    result = repo.get_research_with_gaps([1,2,3])

    # Research별로 gaps가 묶였는지 확인
    assert len(result) == 2

    # research1
    r1_gaps = next(r.gaps for r in result if r.research.id == 1)
    assert len(r1_gaps) == 2
    assert {g.id for g in r1_gaps} == {1,2}

    # research2
    r2_gaps = next(r.gaps for r in result if r.research.id == 2)
    assert len(r2_gaps) == 1
    assert r2_gaps[0].id == 3
