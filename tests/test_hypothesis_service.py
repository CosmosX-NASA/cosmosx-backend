# tests/test_hypothesis_service.py
import pytest
from model import Hypothesis
from repository.hypothesis_repository import HypothesisRepository
from repository.hypothesis_research_repository import HypothesisResearchRepository
from repository.research_gaps_repository import ResearchGapsRepository
from prompt.resolver.hypothesis_prompt_resolver import HypothesisPromptResolver
from client.open_ai_client import OpenAiClient
from repository.research_repository import ResearchRagRepository
from service.hypothesis_service import HypothesisService
from dto.hypothesis_dto import HypothesisDoneCountResponse


@pytest.fixture
def sample_hypotheses(db_session):
    """테스트용 가설 데이터"""
    data = [
        Hypothesis(id=1, status="DONE",
                   statement="디지털 신뢰는 소비자 행동에 긍정적 영향을 미친다",
                   usage="온라인 쇼핑몰 신뢰도 향상 전략",
                   evidence="기존 연구 분석 결과",
                   user_id=1),
        Hypothesis(id=2, status="DONE",
                   statement="조직학습은 기업 성과를 향상시킨다",
                   usage="조직 학습 문화 구축",
                   evidence="메타분석 결과",
                   user_id=1),
        Hypothesis(id=3, status="PENDING",
                   statement=None,
                   usage=None,
                   evidence=None,
                   user_id=1),
        Hypothesis(id=4, status="DONE",
                   statement="AI 기술 수용은 조직 혁신을 촉진한다",
                   usage="AI 도입 전략",
                   evidence="혁신확산이론 기반",
                   user_id=2),
        Hypothesis(id=5, status="DONE",
                   statement="원격근무는 업무 생산성에 영향을 미친다",
                   usage="하이브리드 근무 정책",
                   evidence="코로나19 이후 연구 동향",
                   user_id=2),
    ]
    db_session.add_all(data)
    db_session.commit()
    for hypo in data:
        db_session.refresh(hypo)
    return data


def test_get_done_hypothesis_count_with_results(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - user_id=1, 2개 완료"""
    # Given
    hypothesis_repo = HypothesisRepository(db_session)
    hypothesis_research_repo = HypothesisResearchRepository(db_session)
    research_gaps_repo = ResearchGapsRepository(db_session)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver=prompt_resolver)
    research_repository = ResearchRagRepository(db_session)
    service = HypothesisService(
        hypothesis_repo,
        hypothesis_research_repo,
        research_gaps_repo,
        openai_client,
        research_repository
    )

    # When
    result = service.get_done_hypothesis_count(user_id=1)

    # Then
    assert isinstance(result, HypothesisDoneCountResponse)
    assert result.count == 2


def test_get_done_hypothesis_count_different_user(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - user_id=2, 2개 완료"""
    # Given
    hypothesis_repo = HypothesisRepository(db_session)
    hypothesis_research_repo = HypothesisResearchRepository(db_session)
    research_gaps_repo = ResearchGapsRepository(db_session)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver=prompt_resolver)
    research_repository = ResearchRagRepository(db_session)

    service = HypothesisService(
        hypothesis_repo,
        hypothesis_research_repo,
        research_gaps_repo,
        openai_client,
        research_repository
    )

    # When
    result = service.get_done_hypothesis_count(user_id=2)

    # Then
    assert isinstance(result, HypothesisDoneCountResponse)
    assert result.count == 2


def test_get_done_hypothesis_count_no_results(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - 존재하지 않는 user_id=999"""
    # Given
    hypothesis_repo = HypothesisRepository(db_session)
    hypothesis_research_repo = HypothesisResearchRepository(db_session)
    research_gaps_repo = ResearchGapsRepository(db_session)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver=prompt_resolver)
    research_repository = ResearchRagRepository(db_session)

    service = HypothesisService(
        hypothesis_repo,
        hypothesis_research_repo,
        research_gaps_repo,
        openai_client,
        research_repository
    )

    # When
    result = service.get_done_hypothesis_count(user_id=999)

    # Then
    assert isinstance(result, HypothesisDoneCountResponse)
    assert result.count == 0


def test_get_done_hypothesis_count_empty_db(db_session):
    """완료된 가설 개수 조회 - 데이터가 없는 경우"""
    # Given
    hypothesis_repo = HypothesisRepository(db_session)
    hypothesis_research_repo = HypothesisResearchRepository(db_session)
    research_gaps_repo = ResearchGapsRepository(db_session)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver=prompt_resolver)
    research_repository = ResearchRagRepository(db_session)
    service = HypothesisService(
        hypothesis_repo,
        hypothesis_research_repo,
        research_gaps_repo,
        openai_client,
        research_repository
    )

    # When
    result = service.get_done_hypothesis_count(user_id=1)

    # Then
    assert isinstance(result, HypothesisDoneCountResponse)
    assert result.count == 0


def test_get_done_hypothesis_count_only_pending(db_session):
    """완료된 가설 개수 조회 - PENDING만 있는 경우"""
    # Given
    pending_only = [
        Hypothesis(id=1, status="PENDING", statement=None, usage=None, evidence=None, user_id=1),
        Hypothesis(id=2, status="PENDING", statement=None, usage=None, evidence=None, user_id=1),
    ]
    db_session.add_all(pending_only)
    db_session.commit()

    hypothesis_repo = HypothesisRepository(db_session)
    hypothesis_research_repo = HypothesisResearchRepository(db_session)
    research_gaps_repo = ResearchGapsRepository(db_session)
    prompt_resolver = HypothesisPromptResolver()
    openai_client = OpenAiClient(prompt_resolver=prompt_resolver)
    research_repository = ResearchRagRepository(db_session)

    service = HypothesisService(
        hypothesis_repo,
        hypothesis_research_repo,
        research_gaps_repo,
        openai_client,
        research_repository
    )

    # When
    result = service.get_done_hypothesis_count(user_id=1)

    # Then
    assert isinstance(result, HypothesisDoneCountResponse)
    assert result.count == 0