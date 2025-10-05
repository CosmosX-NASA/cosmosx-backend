# tests/test_hypothesis_api.py
import pytest
from fastapi.testclient import TestClient
from model import Hypothesis
from db.db import get_db_session
from main import app


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


def test_get_done_hypothesis_count_success(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 성공 - user_id=1, 2개"""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done?userId=1")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert data["count"] == 2
    finally:
        app.dependency_overrides.clear()


def test_get_done_hypothesis_count_different_user(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - user_id=2, 2개"""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done?userId=2")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert data["count"] == 2
    finally:
        app.dependency_overrides.clear()


def test_get_done_hypothesis_count_no_hypotheses(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - 존재하지 않는 user_id=999"""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done?userId=999")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert data["count"] == 0
    finally:
        app.dependency_overrides.clear()


def test_get_done_hypothesis_count_empty_db(db_session):
    """완료된 가설 개수 조회 - 빈 데이터베이스"""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done?userId=1")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert data["count"] == 0
    finally:
        app.dependency_overrides.clear()


def test_get_done_hypothesis_count_only_pending(db_session):
    """완료된 가설 개수 조회 - PENDING 상태만 있는 경우"""
    pending_only = [
        Hypothesis(id=1, status="PENDING", statement=None, usage=None, evidence=None, user_id=1),
        Hypothesis(id=2, status="PENDING", statement=None, usage=None, evidence=None, user_id=1),
    ]
    db_session.add_all(pending_only)
    db_session.commit()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done?userId=1")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert data["count"] == 0
    finally:
        app.dependency_overrides.clear()


def test_get_done_hypothesis_count_missing_user_id(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - userId 쿼리 파라미터 누락"""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done")
        assert response.status_code == 422  # Validation Error
    finally:
        app.dependency_overrides.clear()


def test_get_done_hypothesis_count_invalid_user_id(db_session, sample_hypotheses):
    """완료된 가설 개수 조회 - 잘못된 userId 형식"""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    try:
        client = TestClient(app)

        response = client.get("/api/hypothesis/done?userId=invalid")
        assert response.status_code == 422  # Validation Error
    finally:
        app.dependency_overrides.clear()