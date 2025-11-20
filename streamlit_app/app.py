"""
Ko-CodeNurse AI Streamlit 웹 인터페이스
FastAPI 백엔드와 통합된 간호 지식 검색 및 답변 생성 애플리케이션
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, List


def call_fastapi_generate(query: str, api_url: str = "http://localhost:8000/api/generate") -> Optional[Dict]:
    """
    FastAPI 백엔드의 /api/generate 엔드포인트를 호출합니다.
    
    Args:
        query: 사용자 질문
        api_url: FastAPI 엔드포인트 URL
    
    Returns:
        API 응답 딕셔너리 (answer, sources, domain) 또는 None
    """
    try:
        response = requests.post(
            api_url,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=60  # 60초 타임아웃
        )
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        st.info("💡 FastAPI 서버를 시작하려면: `uvicorn src.main:app --reload`")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ 요청 시간이 초과되었습니다. 다시 시도해주세요.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API 오류 발생: {e}")
        if response.status_code == 404:
            st.info("💡 벡터 저장소가 생성되지 않았을 수 있습니다. rag_integration.py를 실행해주세요.")
        return None
    except Exception as e:
        st.error(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        return None


def display_safety_warning():
    """
    NURSING 도메인일 때 안전 고지를 표시합니다.
    """
    st.warning("""
    ⚠️ **안전 고지**
    
    이 정보는 참고용이며, 실제 환자 케어 시에는 반드시 의료진과 상의하시기 바랍니다.
    의료 및 간호 관련 정보는 정확성이 매우 중요하므로, 제공된 정보를 검증하고 전문가의 조언을 구하시기 바랍니다.
    """)


def display_sources(sources: List[Dict]):
    """
    출처 정보를 사용자에게 보기 쉽게 표시합니다.
    
    Args:
        sources: 출처 정보 리스트
    """
    if not sources:
        return
    
    st.markdown("### 📚 출처 정보")
    st.markdown("---")
    
    for i, source in enumerate(sources, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{i}. {source.get('source', 'Unknown')}**")
                if source.get('page') is not None:
                    st.caption(f"페이지: {source.get('page')}")
            
            with col2:
                if source.get('relevance_score') is not None:
                    score = source.get('relevance_score', 0)
                    st.metric("유사도", f"{score:.3f}")
            
            st.markdown("---")


def main():
    """Streamlit 앱 메인 함수"""
    # 페이지 설정
    st.set_page_config(
        page_title="Ko-CodeNurse AI",
        page_icon="🏥",
        layout="wide"
    )
    
    # 제목 및 설명
    st.title("🏥 Ko-CodeNurse AI")
    st.markdown("### 한국어 기반 간호 지식 검색 시스템")
    st.markdown("---")
    
    # 사이드바에 정보 표시
    with st.sidebar:
        st.header("ℹ️ 시스템 정보")
        st.markdown("""
        **Ko-CodeNurse AI**는 RAG(Retrieval-Augmented Generation) 기술을 활용하여
        간호, 코딩, 연구 관련 질문에 답변을 제공하는 시스템입니다.
        
        ### 사용 방법
        1. 아래 입력창에 질문을 입력하세요
        2. "답변 생성" 버튼을 클릭하세요
        3. 생성된 답변과 출처 정보를 확인하세요
        
        ### 도메인
        - **NURSING**: 간호 및 의료 관련 질문
        - **CODING**: 프로그래밍 및 코딩 관련 질문
        - **RESEARCH**: 연구 및 학술 관련 질문
        """)
        
        st.markdown("---")
        st.markdown("### 🔗 API 상태")
        # API 서버 상태 확인
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API 서버 연결됨")
            else:
                st.warning("⚠️ API 서버 응답 이상")
        except:
            st.error("❌ API 서버 연결 실패")
            st.info("💡 FastAPI 서버를 시작하려면:\n`uvicorn src.main:app --reload`")
    
    # 사용자 입력 영역
    st.markdown("### 💬 질문 입력")
    query = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 활력징후 측정 절차, Python 리스트 정렬 방법, 간호 연구 방법론",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 10])
    with col1:
        generate_button = st.button("✨ 답변 생성", type="primary", use_container_width=True)
    
    # 답변 생성 실행
    if generate_button:
        if not query.strip():
            st.warning("⚠️ 질문을 입력해주세요.")
        else:
            with st.spinner("🤖 답변을 생성하는 중..."):
                api_response = call_fastapi_generate(query)
                
                if api_response:
                    # 도메인별 UI 분기
                    domain = api_response.get("domain", "UNKNOWN")
                    answer = api_response.get("answer", "")
                    sources = api_response.get("sources", [])
                    
                    # 도메인 표시
                    domain_labels = {
                        "NURSING": "🏥 간호",
                        "CODING": "💻 코딩",
                        "RESEARCH": "📊 연구"
                    }
                    domain_label = domain_labels.get(domain, domain)
                    st.markdown(f"### 📌 분류된 도메인: {domain_label}")
                    
                    # NURSING 도메인일 경우 안전 고지 표시
                    if domain == "NURSING":
                        display_safety_warning()
                    
                    # 답변 표시
                    st.markdown("### 💡 답변")
                    st.markdown("---")
                    st.markdown(answer)
                    st.markdown("---")
                    
                    # 출처 정보 표시
                    if sources:
                        display_sources(sources)
                    else:
                        st.info("📝 이 답변은 문서 검색 없이 생성되었습니다.")
    
    # 초기 안내 메시지
    elif not query:
        st.info("👆 위 입력창에 질문을 입력하고 '답변 생성' 버튼을 클릭하세요.")


if __name__ == "__main__":
    main()

