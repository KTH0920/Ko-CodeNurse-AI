"""
프롬프트 템플릿 테스트
src/prompts.py의 프롬프트 함수들을 테스트합니다.
"""

import pytest
from src.prompts import (
    get_nursing_system_prompt,
    get_coding_system_prompt,
    get_research_system_prompt,
    format_user_prompt_with_context,
    format_user_prompt_without_context
)


def test_nursing_prompt_contains_safety_warning():
    """
    간호 전문가 프롬프트에 안전 고지 키워드가 포함되어 있는지 테스트합니다.
    """
    prompt = get_nursing_system_prompt()
    
    # 핵심 키워드 검증
    assert "안전 고지" in prompt or "주의" in prompt, "간호 프롬프트에 안전 고지가 포함되어 있지 않습니다."
    assert "의료진" in prompt or "의료" in prompt, "간호 프롬프트에 의료 관련 키워드가 포함되어 있지 않습니다."
    assert "출처" in prompt, "간호 프롬프트에 출처 요청이 포함되어 있지 않습니다."


def test_coding_prompt_contains_code_block():
    """
    코딩 전문가 프롬프트에 코드 블록 키워드가 포함되어 있는지 테스트합니다.
    """
    prompt = get_coding_system_prompt()
    
    # 핵심 키워드 검증
    assert "코드 블록" in prompt or "```" in prompt, "코딩 프롬프트에 코드 블록 키워드가 포함되어 있지 않습니다."
    assert "코드" in prompt, "코딩 프롬프트에 코드 관련 키워드가 포함되어 있지 않습니다."
    assert "예제" in prompt or "예시" in prompt, "코딩 프롬프트에 예제 요청이 포함되어 있지 않습니다."


def test_research_prompt_contains_objective():
    """
    연구 전문가 프롬프트에 객관적 요약 키워드가 포함되어 있는지 테스트합니다.
    """
    prompt = get_research_system_prompt()
    
    # 핵심 키워드 검증
    assert "객관적" in prompt, "연구 프롬프트에 객관적 키워드가 포함되어 있지 않습니다."
    assert "출처" in prompt, "연구 프롬프트에 출처 요청이 포함되어 있지 않습니다."
    assert "연구" in prompt or "분석" in prompt, "연구 프롬프트에 연구/분석 관련 키워드가 포함되어 있지 않습니다."


def test_nursing_prompt_structure():
    """
    간호 전문가 프롬프트의 구조가 올바른지 테스트합니다.
    """
    prompt = get_nursing_system_prompt()
    
    # 기본 구조 검증
    assert len(prompt) > 0, "프롬프트가 비어있습니다."
    assert "간호" in prompt or "간호 전문가" in prompt, "프롬프트에 간호 관련 내용이 없습니다."
    assert "한국어" in prompt, "프롬프트에 한국어 작성 지침이 없습니다."


def test_coding_prompt_structure():
    """
    코딩 전문가 프롬프트의 구조가 올바른지 테스트합니다.
    """
    prompt = get_coding_system_prompt()
    
    # 기본 구조 검증
    assert len(prompt) > 0, "프롬프트가 비어있습니다."
    assert "소프트웨어" in prompt or "프로그래밍" in prompt, "프롬프트에 프로그래밍 관련 내용이 없습니다."
    assert "한국어" in prompt, "프롬프트에 한국어 작성 지침이 없습니다."


def test_research_prompt_structure():
    """
    연구 전문가 프롬프트의 구조가 올바른지 테스트합니다.
    """
    prompt = get_research_system_prompt()
    
    # 기본 구조 검증
    assert len(prompt) > 0, "프롬프트가 비어있습니다."
    assert "연구" in prompt or "학술" in prompt, "프롬프트에 연구 관련 내용이 없습니다."
    assert "한국어" in prompt, "프롬프트에 한국어 작성 지침이 없습니다."


def test_format_user_prompt_with_context():
    """
    컨텍스트를 포함한 사용자 프롬프트 형식이 올바른지 테스트합니다.
    """
    query = "테스트 질문"
    context_documents = [
        {
            "content": "테스트 내용",
            "metadata": {"source": "test.pdf", "page": 1}
        }
    ]
    
    prompt = format_user_prompt_with_context(query, context_documents)
    
    # 구조 검증
    assert query in prompt, "질문이 프롬프트에 포함되어 있지 않습니다."
    assert "참고 문서" in prompt, "참고 문서 섹션이 없습니다."
    assert "출처" in prompt, "출처 정보가 포함되어 있지 않습니다."


def test_format_user_prompt_without_context():
    """
    컨텍스트 없이 사용자 프롬프트 형식이 올바른지 테스트합니다.
    """
    query = "테스트 질문"
    
    prompt = format_user_prompt_without_context(query)
    
    # 구조 검증
    assert query in prompt, "질문이 프롬프트에 포함되어 있지 않습니다."
    assert "질문" in prompt, "질문 섹션이 없습니다."
    assert "답변" in prompt, "답변 섹션이 없습니다."


def test_format_user_prompt_with_empty_context():
    """
    빈 컨텍스트로 사용자 프롬프트를 생성하는지 테스트합니다.
    """
    query = "테스트 질문"
    context_documents = []
    
    prompt = format_user_prompt_with_context(query, context_documents)
    
    # 빈 컨텍스트일 때는 간단한 형식 반환
    assert query in prompt, "질문이 프롬프트에 포함되어 있지 않습니다."

