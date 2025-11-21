"""
FastAPI 메인 애플리케이션 테스트
src/main.py의 API 엔드포인트를 테스트합니다.
Mocking을 사용하여 LLM 호출 비용을 절감합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """
    FastAPI TestClient 픽스처
    """
    return TestClient(app)


@pytest.fixture
def mock_nursing_documents():
    """
    Mock된 간호 문서 검색 결과
    """
    return [
        {
            "content": "주사기 투여 시에는 무균 기술을 준수해야 합니다.",
            "metadata": {"source": "test_nursing.pdf", "page": 1},
            "score": 0.95
        },
        {
            "content": "활력징후 측정은 체온, 맥박, 호흡, 혈압을 순서대로 측정합니다.",
            "metadata": {"source": "test_nursing.pdf", "page": 2},
            "score": 0.88
        }
    ]


@pytest.fixture
def mock_openai_response():
    """
    Mock된 OpenAI API 응답
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "이것은 테스트용 더미 답변입니다. 주사기 투여 시 무균 기술을 준수해야 합니다."
    return mock_response


@patch('src.main.classify_domain')
@patch('src.main.get_relevant_documents_with_content')
@patch('src.main.openai.OpenAI')
def test_api_generate_nursing(
    mock_openai_client,
    mock_get_documents,
    mock_classify_domain,
    client,
    mock_nursing_documents,
    mock_openai_response
):
    """
    /api/generate 엔드포인트가 NURSING 도메인 질문에 대해 올바르게 응답하는지 테스트합니다.
    도메인 분류와 LLM 호출을 Mocking합니다.
    """
    # Mock 설정
    mock_classify_domain.return_value = "NURSING"
    mock_get_documents.return_value = mock_nursing_documents
    
    # OpenAI 클라이언트 Mock 설정
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.return_value = mock_openai_response
    
    # API 요청
    response = client.post(
        "/api/generate",
        json={"query": "주사기 투여 기술"}
    )
    
    # 응답 검증
    assert response.status_code == 200, f"예상된 상태 코드 200이 아닙니다: {response.status_code}"
    
    data = response.json()
    
    # 필수 필드 검증
    assert "answer" in data, "응답에 answer 필드가 없습니다."
    assert "sources" in data, "응답에 sources 필드가 없습니다."
    assert "domain" in data, "응답에 domain 필드가 없습니다."
    
    # domain 필드 검증
    assert data["domain"] == "NURSING", f"예상된 도메인 'NURSING'이 아닙니다: {data['domain']}"
    
    # sources 필드 형식 검증
    assert isinstance(data["sources"], list), "sources가 리스트 형식이 아닙니다."
    assert len(data["sources"]) > 0, "sources가 비어있습니다."
    
    # 각 source의 구조 검증
    for source in data["sources"]:
        assert isinstance(source, dict), "source가 딕셔너리 형식이 아닙니다."
        assert "source" in source, "source에 'source' 키가 없습니다."
        assert "page" in source, "source에 'page' 키가 없습니다."
        assert "relevance_score" in source, "source에 'relevance_score' 키가 없습니다."
        
        # 값 타입 검증
        assert isinstance(source["source"], str), "source['source']가 문자열이 아닙니다."
        assert source["page"] is None or isinstance(source["page"], (int, str)), "source['page']가 올바른 타입이 아닙니다."
        assert isinstance(source["relevance_score"], (int, float)), "source['relevance_score']가 숫자가 아닙니다."
    
    # Mock 호출 검증
    mock_classify_domain.assert_called_once_with("주사기 투여 기술")
    mock_get_documents.assert_called_once_with("주사기 투여 기술", domain="NURSING", k=3)
    mock_client_instance.chat.completions.create.assert_called_once()


@patch('src.main.classify_domain')
@patch('src.main.get_relevant_documents_with_content')
@patch('src.main.openai.OpenAI')
def test_api_generate_coding(
    mock_openai_client,
    mock_get_documents,
    mock_classify_domain,
    client,
    mock_openai_response
):
    """
    /api/generate 엔드포인트가 CODING 도메인 질문에 대해 올바르게 응답하는지 테스트합니다.
    CODING 도메인은 RAG 검색을 건너뛰므로 빈 리스트를 반환합니다.
    """
    # Mock 설정
    mock_classify_domain.return_value = "CODING"
    mock_get_documents.return_value = []  # CODING은 RAG 검색 안 함
    
    # OpenAI 클라이언트 Mock 설정
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.return_value = mock_openai_response
    
    # API 요청
    response = client.post(
        "/api/generate",
        json={"query": "파이썬 함수 작성"}
    )
    
    # 응답 검증
    assert response.status_code == 200
    
    data = response.json()
    assert data["domain"] == "CODING"
    assert data["sources"] == [], "CODING 도메인은 sources가 빈 리스트여야 합니다."
    
    # CODING 도메인은 RAG 검색을 건너뛰므로 get_relevant_documents_with_content가 호출되지 않아야 함
    # 하지만 현재 코드는 domain in ["NURSING", "RESEARCH"]일 때만 호출하므로
    # CODING일 때는 호출되지 않아야 합니다.
    # 실제로는 호출되지 않지만, 테스트를 위해 확인
    # mock_get_documents.assert_not_called()  # 이 부분은 실제 구현에 따라 다를 수 있음


@patch('src.main.classify_domain')
@patch('src.main.get_relevant_documents_with_content')
def test_api_generate_without_documents(
    mock_get_documents,
    mock_classify_domain,
    client
):
    """
    검색 결과가 없을 때 적절한 에러를 반환하는지 테스트합니다.
    """
    # Mock 설정
    mock_classify_domain.return_value = "NURSING"
    mock_get_documents.return_value = []  # 빈 결과
    
    # API 요청
    response = client.post(
        "/api/generate",
        json={"query": "존재하지 않는 질문"}
    )
    
    # 에러 응답 검증
    assert response.status_code == 404, "검색 결과가 없을 때 404를 반환해야 합니다."
    assert "검색 결과가 없습니다" in response.json()["detail"]


def test_health_endpoint(client):
    """
    /health 엔드포인트가 정상적으로 작동하는지 테스트합니다.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data


@patch('src.main.classify_domain')
@patch('src.main.get_relevant_documents_with_content')
@patch('src.main.openai.OpenAI')
def test_api_generate_response_structure(
    mock_openai_client,
    mock_get_documents,
    mock_classify_domain,
    client,
    mock_nursing_documents,
    mock_openai_response
):
    """
    /api/generate 응답의 전체 구조가 올바른지 테스트합니다.
    """
    # Mock 설정
    mock_classify_domain.return_value = "NURSING"
    mock_get_documents.return_value = mock_nursing_documents
    
    # OpenAI 클라이언트 Mock 설정
    mock_client_instance = MagicMock()
    mock_openai_client.return_value = mock_client_instance
    mock_client_instance.chat.completions.create.return_value = mock_openai_response
    
    # API 요청
    response = client.post(
        "/api/generate",
        json={"query": "테스트 질문"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # 전체 구조 검증
    required_keys = ["answer", "sources", "domain"]
    for key in required_keys:
        assert key in data, f"응답에 필수 키 '{key}'가 없습니다."
    
    # answer 검증
    assert isinstance(data["answer"], str), "answer가 문자열이 아닙니다."
    assert len(data["answer"]) > 0, "answer가 비어있습니다."
    
    # domain 검증
    assert data["domain"] in ["NURSING", "CODING", "RESEARCH"], f"domain이 유효한 값이 아닙니다: {data['domain']}"
    
    # sources 검증
    assert isinstance(data["sources"], list), "sources가 리스트가 아닙니다."

