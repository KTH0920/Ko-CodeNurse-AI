"""
도메인 분류 라우터 모듈
사용자 질문을 NURSING, CODING, RESEARCH 중 하나로 분류합니다.
"""

from typing import Literal
import re
from src.local_llm_connector import generate_response

# 도메인 타입 정의
DomainType = Literal["NURSING", "CODING", "RESEARCH"]


def classify_domain(query: str) -> DomainType:
    """
    사용자 질문을 도메인으로 분류합니다.
    로컬 LLM을 사용하여 NURSING, CODING, RESEARCH 중 하나를 반환합니다.
    
    Args:
        query: 사용자 질문
    
    Returns:
        분류된 도메인 (NURSING, CODING, RESEARCH)
    """
    # 도메인 분류 프롬프트 (KoAlpaca 형식)
    classification_prompt = """당신은 사용자의 질문을 다음 세 가지 도메인 중 하나로 분류하는 전문가입니다:

1. NURSING: 간호, 의료, 환자 케어, 건강 관리, 간호 실무, 의학적 절차 등과 관련된 질문
2. CODING: 프로그래밍, 코딩, 소프트웨어 개발, 알고리즘, 기술 스택, 코드 작성 등과 관련된 질문
3. RESEARCH: 연구, 학술, 논문, 데이터 분석, 실험, 조사 등과 관련된 질문

사용자 질문을 분석하여 가장 적합한 도메인을 선택하세요.
반드시 다음 중 하나만 정확히 반환하세요: NURSING, CODING, RESEARCH

### 질문:
{query}

### 답변:
""".format(query=query)

    try:
        # 로컬 LLM 호출 (temperature=0으로 일관성 확보)
        response = generate_response(classification_prompt, max_length=50, temperature=0)
        
        # 응답에서 도메인 추출 (대문자 변환 및 정규화)
        response_upper = response.strip().upper()
        
        # 유효한 도메인 키워드 찾기
        valid_domains = ["NURSING", "CODING", "RESEARCH"]
        domain = None
        
        for valid_domain in valid_domains:
            if valid_domain in response_upper:
                domain = valid_domain
                break
        
        # 도메인을 찾지 못한 경우 기본값으로 NURSING 반환
        if domain is None:
            print(f"⚠️ 경고: 예상치 못한 도메인 응답 '{response}'. 기본값 NURSING으로 설정합니다.")
            return "NURSING"
        
        return domain
    
    except Exception as e:
        # 에러 발생 시 기본값으로 NURSING 반환
        print(f"⚠️ 경고: 도메인 분류 중 오류 발생: {e}. 기본값 NURSING으로 설정합니다.")
        return "NURSING"


def get_domain_description(domain: DomainType) -> str:
    """
    도메인에 대한 설명을 반환합니다.
    
    Args:
        domain: 도메인 타입
    
    Returns:
        도메인 설명 문자열
    """
    descriptions = {
        "NURSING": "간호 및 의료 관련 질문",
        "CODING": "프로그래밍 및 코딩 관련 질문",
        "RESEARCH": "연구 및 학술 관련 질문"
    }
    return descriptions.get(domain, "알 수 없는 도메인")

