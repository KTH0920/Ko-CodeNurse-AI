"""
Ko-CodeNurse AI FastAPI 백엔드
RAG 시스템과 LLM을 통합한 간호 지식 검색 및 답변 생성 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import openai
from src.rag_connector import get_relevant_documents_with_content
from src.router import classify_domain, DomainType
from src.prompts import (
    get_nursing_system_prompt,
    get_coding_system_prompt,
    get_research_system_prompt,
    format_user_prompt_with_context,
    format_user_prompt_without_context
)

# 환경 변수 로드
load_dotenv()

# FastAPI 인스턴스 생성
app = FastAPI(
    title="Ko-CodeNurse AI API",
    description="한국어 기반 간호 지식 검색 및 답변 생성 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ 경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")


# 요청/응답 모델 정의
class GenerateRequest(BaseModel):
    """답변 생성 요청 모델"""
    query: str


class GenerateResponse(BaseModel):
    """답변 생성 응답 모델"""
    answer: str
    sources: List[dict]  # 출처 정보
    domain: str  # 분류된 도메인 (NURSING, CODING, RESEARCH)


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    message: str


def generate_answer_by_domain(
    query: str,
    domain: DomainType,
    context_documents: List[dict]
) -> str:
    """
    도메인에 따라 적절한 전문가 페르소나로 답변을 생성합니다.
    
    Args:
        query: 사용자 질문
        domain: 분류된 도메인 (NURSING, CODING, RESEARCH)
        context_documents: RAG 검색 결과 문서 리스트 (CODING일 경우 빈 리스트)
    
    Returns:
        생성된 답변
    """
    if not OPENAI_API_KEY:
        # API 키가 없을 경우 기본 답변 반환
        return f"질문: {query}\n\n답변을 생성하려면 OPENAI_API_KEY 환경 변수를 설정해주세요."
    
    # 도메인에 따라 시스템 프롬프트 선택
    if domain == "NURSING":
        system_prompt = get_nursing_system_prompt()
    elif domain == "CODING":
        system_prompt = get_coding_system_prompt()
    elif domain == "RESEARCH":
        system_prompt = get_research_system_prompt()
    else:
        # 기본값으로 간호 전문가 사용
        system_prompt = get_nursing_system_prompt()
    
    # 컨텍스트 문서가 있으면 포함, 없으면 질문만 전달
    if context_documents:
        user_prompt = format_user_prompt_with_context(query, context_documents)
    else:
        user_prompt = format_user_prompt_without_context(query)
    
    try:
        # OpenAI API 호출
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        return answer
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 답변 생성 중 오류가 발생했습니다: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API 서버 상태를 확인하는 헬스 체크 엔드포인트
    """
    return HealthResponse(
        status="healthy",
        message="Ko-CodeNurse AI API is running"
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    """
    도메인 분류 후 RAG 검색 결과를 기반으로 답변을 생성합니다.
    
    Args:
        request: 답변 생성 요청 (질문)
    
    Returns:
        생성된 답변, 출처 정보, 분류된 도메인
    """
    try:
        # 1. 도메인 분류 수행
        domain = classify_domain(request.query)
        
        # 2. RAG 검색 분기: NURSING과 RESEARCH일 때만 RAG 검색 수행
        context_documents = []
        if domain in ["NURSING", "RESEARCH"]:
            # NURSING과 RESEARCH 도메인일 때만 도메인별 RAG 검색 수행
            context_documents = get_relevant_documents_with_content(request.query, domain=domain, k=3)
            
            if not context_documents:
                raise HTTPException(
                    status_code=404,
                    detail="검색 결과가 없습니다. 벡터 저장소가 비어있거나 질문과 관련된 문서가 없습니다."
                )
        # CODING 도메인일 때는 RAG 검색을 건너뛰고 context_documents는 빈 리스트로 유지
        
        # 3. 도메인에 따라 적절한 전문가 페르소나로 답변 생성
        answer = generate_answer_by_domain(request.query, domain, context_documents)
        
        # 4. 출처 정보를 API 형식으로 변환
        sources = []
        if context_documents:
            sources = [
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "page": doc["metadata"].get("page", None),
                    "relevance_score": doc.get("score", None)
                }
                for doc in context_documents
            ]
        # CODING 도메인일 때는 sources가 빈 리스트
        
        return GenerateResponse(
            answer=answer,
            sources=sources,
            domain=domain
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"벡터 저장소를 찾을 수 없습니다: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # 환경 변수에서 호스트와 포트 읽기
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)

