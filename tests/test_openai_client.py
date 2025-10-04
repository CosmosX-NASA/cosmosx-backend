import pytest
from unittest.mock import MagicMock

from openai import OpenAI
from prompt.resolver.hypothesis_prompt_resolver import HypothesisPromptResolver
from client.open_ai_client import OpenAiClient
from dto.hypothesis_dto import HypothesisAiResponse
from model import ResearchWithGaps, ResearchGap, Research
from dotenv import load_dotenv

# @pytest.fixture
# def mock_prompt_resolver():
#     resolver = MagicMock(spec=HypothesisPromptResolver)
#     resolver.resolve.return_value = "PROMPT CONTENT"
#     return resolver
#
#
# @pytest.fixture
# def mock_openai_client(monkeypatch):
#     # OpenAI.ChatCompletion mock
#     mock_client = MagicMock(spec=OpenAI)
#     mock_response = MagicMock()
#     # OpenAI 반환 객체 구조 맞추기
#     mock_response.choices = [MagicMock()]
#     mock_response.choices[
#         0].message.content = '{"statement":"Test stmt","usage":"Test usage","evidence":"Test evidence"}'
#     mock_client.chat.completions.create.return_value = mock_response
#
#     from client import open_ai_client
#     monkeypatch.setattr(open_ai_client, "OpenAI", lambda *args, **kwargs: mock_client)
#     return mock_client

def test_create_hypothesis_success(mock_prompt_resolver, mock_openai_client):
    client = OpenAiClient(prompt_resolver=mock_prompt_resolver, api_key="DUMMY_KEY")

    from model import ResearchWithGaps, ResearchGap, Research

    # 더미 Research 객체
    dummy_research = Research(
        id=1,
        pmc_id="PMC12345",
        title="Test Research",
        journal="Test Journal",
        doi="https://doi.org/10.1234/test",
        author="Author Name",
        release_date="2025-10-04",
        brief_summary="Brief summary",
        overall_summary="Overall summary",
        methods="Methods description",
        results="Results description"
    )

    # 더미 ResearchGap 객체
    dummy_gap1 = ResearchGap(
        id=1,
        type="CONCEPTUAL",
        content="Dummy gap 1 content",
        evidence="Evidence 1",
        research_title="Test Research",
        research_id=1
    )

    dummy_gap2 = ResearchGap(
        id=2,
        type="METHODLOGICAL",
        content="Dummy gap 2 content",
        evidence="Evidence 2",
        research_title="Test Research",
        research_id=1
    )

    # ResearchWithGaps 리스트
    research_gaps_list = [
        ResearchWithGaps(
            research=dummy_research,
            gaps=[dummy_gap1, dummy_gap2]
        )
    ]

    result = client.create_hypothesis(research_gaps_list)

    # 반환 타입 확인
    assert isinstance(result, HypothesisAiResponse)
    assert result.statement == "Test stmt"
    assert result.usage == "Test usage"
    assert result.evidence == "Test evidence"

    # prompt resolver가 호출되었는지 확인
    mock_prompt_resolver.resolve.assert_called_once_with(research_gaps_list)

    # OpenAI API가 호출되었는지 확인
    mock_openai_client.chat.completions.create.assert_called_once()


# 실제 호출 테스트

load_dotenv()

@pytest.mark.skip(reason="실제 OpenAI 호출 테스트이므로 현재는 건너뜀")
def test_create_hypothesis_real():
    # 프롬프트 리졸버 (실제 파일 로딩)
    resolver = HypothesisPromptResolver()

    client = OpenAiClient(prompt_resolver=resolver)

    # 더미 Research, ResearchGap 객체 생성
    dummy_research = Research(
        id=1, pmc_id="PMC12345", title="Test Research", journal="Test Journal",
        doi="https://doi.org/10.1234/test", author="Author Name",
        release_date="2025-10-04", brief_summary="Brief summary",
        overall_summary="Overall summary", methods="Methods description", results="Results description"
    )

    dummy_gap1 = ResearchGap(
        id=1, type="CONCEPTUAL", content="Dummy gap 1 content",
        evidence="Evidence 1", research_title="Test Research", research_id=1
    )

    research_gaps_list = [ResearchWithGaps(research=dummy_research, gaps=[dummy_gap1])]

    result: HypothesisAiResponse = client.create_hypothesis(research_gaps_list)

    assert isinstance(result, HypothesisAiResponse)
    print("AI Response:", result)


@pytest.mark.skip(reason="실제 OpenAI 호출 테스트이므로 현재는 건너뜀")
def test_specify_question_real_call():
    resolver = HypothesisPromptResolver()

    client = OpenAiClient(prompt_resolver=resolver)
    keyword = "biology"
    response = client.specify_question(keyword)

    print("OpenAI response:", response)
    # 최소한 응답이 문자열이고 비어있지 않은지 확인
    assert isinstance(response, str)
    assert len(response.strip()) > 0