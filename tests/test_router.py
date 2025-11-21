"""
도메인 분류 라우터 테스트
src/router.py의 classify_domain() 함수를 테스트합니다.
"""

import pytest
import os
from src.router import classify_domain, DomainType


# 테스트용 질문 및 예상 도메인 매핑
TEST_QUERIES = [
    # CODING 도메인 테스트 케이스
    ("파이썬 함수 작성", "CODING"),
    ("Python 리스트 정렬 방법", "CODING"),
    ("자바스크립트 비동기 처리", "CODING"),
    ("FastAPI 라우터 설정", "CODING"),
    ("데이터베이스 쿼리 최적화", "CODING"),
    
    # NURSING 도메인 테스트 케이스
    ("혈압 측정 절차", "NURSING"),
    ("활력징후 측정 방법", "NURSING"),
    ("주사기 투여 기술", "NURSING"),
    ("환자 케어 계획 수립", "NURSING"),
    ("의료진과의 소통 방법", "NURSING"),
    
    # RESEARCH 도메인 테스트 케이스
    ("AI 연구 동향", "RESEARCH"),
    ("간호 연구 방법론", "RESEARCH"),
    ("데이터 분석 기법", "RESEARCH"),
    ("논문 작성 가이드", "RESEARCH"),
    ("실험 설계 원칙", "RESEARCH"),
]


@pytest.fixture(scope="module")
def check_openai_key():
    """
    OpenAI API 키가 설정되어 있는지 확인하는 픽스처
    테스트 실행 전에 환경 변수를 확인합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 테스트를 실행하려면 환경 변수를 설정해주세요.")
    return api_key


@pytest.mark.parametrize("query,expected_domain", TEST_QUERIES)
def test_classify_domain(query: str, expected_domain: DomainType, check_openai_key):
    """
    classify_domain() 함수가 질문을 올바른 도메인으로 분류하는지 테스트합니다.
    
    Args:
        query: 테스트할 질문
        expected_domain: 예상되는 도메인
        check_openai_key: OpenAI API 키 확인 픽스처
    """
    # 도메인 분류 실행
    result = classify_domain(query)
    
    # 결과 검증
    assert result == expected_domain, (
        f"질문 '{query}'가 예상 도메인 '{expected_domain}' 대신 '{result}'로 분류되었습니다."
    )


def test_classify_domain_valid_domains(check_openai_key):
    """
    classify_domain() 함수가 유효한 도메인만 반환하는지 테스트합니다.
    """
    test_query = "테스트 질문"
    result = classify_domain(test_query)
    
    valid_domains = ["NURSING", "CODING", "RESEARCH"]
    assert result in valid_domains, (
        f"분류 결과 '{result}'가 유효한 도메인 목록 {valid_domains}에 포함되지 않습니다."
    )


def test_classify_domain_without_api_key(monkeypatch):
    """
    OpenAI API 키가 없을 때 ValueError가 발생하는지 테스트합니다.
    """
    # 환경 변수에서 OPENAI_API_KEY 제거
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    # 모듈 레벨의 OPENAI_API_KEY도 None으로 설정하기 위해 모듈 재로드 필요
    # 하지만 실제로는 함수 내부에서 os.getenv를 호출하므로 monkeypatch로 충분
    import importlib
    import src.router
    importlib.reload(src.router)
    
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        src.router.classify_domain("테스트 질문")


def test_classify_domain_coding_queries(check_openai_key):
    """
    코딩 관련 질문들이 CODING 도메인으로 분류되는지 집중 테스트합니다.
    """
    coding_queries = [
        "파이썬 함수 작성",
        "Python 리스트 정렬 방법",
        "자바스크립트 비동기 처리",
        "FastAPI 라우터 설정",
    ]
    
    for query in coding_queries:
        result = classify_domain(query)
        assert result == "CODING", (
            f"코딩 관련 질문 '{query}'가 '{result}'로 분류되었습니다. 예상: CODING"
        )


def test_classify_domain_nursing_queries(check_openai_key):
    """
    간호 관련 질문들이 NURSING 도메인으로 분류되는지 집중 테스트합니다.
    """
    nursing_queries = [
        "혈압 측정 절차",
        "활력징후 측정 방법",
        "주사기 투여 기술",
        "환자 케어 계획 수립",
    ]
    
    for query in nursing_queries:
        result = classify_domain(query)
        assert result == "NURSING", (
            f"간호 관련 질문 '{query}'가 '{result}'로 분류되었습니다. 예상: NURSING"
        )


def test_classify_domain_research_queries(check_openai_key):
    """
    연구 관련 질문들이 RESEARCH 도메인으로 분류되는지 집중 테스트합니다.
    """
    research_queries = [
        "AI 연구 동향",
        "간호 연구 방법론",
        "데이터 분석 기법",
        "논문 작성 가이드",
    ]
    
    for query in research_queries:
        result = classify_domain(query)
        assert result == "RESEARCH", (
            f"연구 관련 질문 '{query}'가 '{result}'로 분류되었습니다. 예상: RESEARCH"
        )

