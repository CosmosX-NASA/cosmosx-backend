# tests/test_hypothesis_repository.py
import pytest
from model import Hypothesis
from repository.hypothesis_repository import HypothesisRepository


@pytest.fixture(scope="function")
def sample_hypotheses(db_session):
    """테스트용 가설 데이터 생성"""
    hypotheses = [
        Hypothesis(
            status="DONE",
            statement="디지털 신뢰는 소비자 행동에 긍정적 영향을 미친다",
            usage="온라인 쇼핑몰 신뢰도 향상 전략 수립",
            evidence="기존 연구에서 신뢰와 구매의도 간 정적 상관관계 확인",
            user_id=1
        ),
        Hypothesis(
            status="DONE",
            statement="조직학습은 기업 성과를 향상시킨다",
            usage="조직 학습 문화 구축 방안 제시",
            evidence="메타분석 결과 학습조직과 성과 간 유의미한 관계",
            user_id=1
        ),
        Hypothesis(
            status="PENDING",
            statement="모바일 앱 사용성은 지속사용의도에 영향을 준다",
            usage="모바일 헬스케어 앱 개선 방향 도출",
            evidence="TAM 모델 기반 선행연구 검토 필요",
            user_id=1
        ),
        Hypothesis(
            status="DONE",
            statement="AI 기술 수용은 조직 혁신을 촉진한다",
            usage="AI 도입 전략 수립",
            evidence="혁신확산이론 기반 분석",
            user_id=2
        ),
        Hypothesis(
            status="PENDING",
            statement="원격근무는 업무 생산성에 영향을 미친다",
            usage="하이브리드 근무 정책 수립",
            evidence="코로나19 이후 연구 동향 분석 중",
            user_id=2
        ),
    ]
    db_session.add_all(hypotheses)
    db_session.commit()
    for hypo in hypotheses:
        db_session.refresh(hypo)
    return hypotheses


def test_count_by_user_id_and_status_done(db_session, sample_hypotheses):
    """user_id=1, status=DONE인 가설 개수 확인"""
    repository = HypothesisRepository(db_session)

    count = repository.count_by_user_id_and_status(status="DONE", user_id=1)

    assert count == 2


def test_count_by_user_id_and_status_pending(db_session, sample_hypotheses):
    """user_id=1, status=PENDING인 가설 개수 확인"""
    repository = HypothesisRepository(db_session)

    count = repository.count_by_user_id_and_status(status="PENDING", user_id=1)

    assert count == 1


def test_count_by_user_id_and_status_different_user(db_session, sample_hypotheses):
    """user_id=2, status=DONE인 가설 개수 확인"""
    repository = HypothesisRepository(db_session)

    count = repository.count_by_user_id_and_status(status="DONE", user_id=2)

    assert count == 1


def test_count_by_user_id_and_status_no_results(db_session, sample_hypotheses):
    """존재하지 않는 조합 (user_id=1, status=COMPLETED) 확인"""
    repository = HypothesisRepository(db_session)

    count = repository.count_by_user_id_and_status(status="COMPLETED", user_id=1)

    assert count == 0


def test_count_by_user_id_and_status_nonexistent_user(db_session, sample_hypotheses):
    """존재하지 않는 사용자 (user_id=999)"""
    repository = HypothesisRepository(db_session)

    count = repository.count_by_user_id_and_status(status="DONE", user_id=999)

    assert count == 0


def test_count_by_user_id_and_status_empty_db(db_session):
    """데이터가 없는 경우"""
    repository = HypothesisRepository(db_session)

    count = repository.count_by_user_id_and_status(status="DONE", user_id=1)

    assert count == 0


def test_count_by_user_id_and_status_all_users(db_session, sample_hypotheses):
    """여러 사용자의 DONE 상태 가설 개수 확인"""
    repository = HypothesisRepository(db_session)

    user1_done = repository.count_by_user_id_and_status(status="DONE", user_id=1)
    user2_done = repository.count_by_user_id_and_status(status="DONE", user_id=2)

    assert user1_done == 2
    assert user2_done == 1
    assert user1_done + user2_done == 3  # 전체 DONE 상태 가설