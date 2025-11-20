"""
Ko-CodeNurse AI Streamlit 웹 인터페이스
RAG 시스템을 활용한 간호 지식 검색 애플리케이션
"""

import streamlit as st
import sys
import os

# 상위 디렉토리를 경로에 추가하여 모듈 import 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_integration import load_embedding_model, create_vector_store
from rag_setup import load_and_split_pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def load_embeddings():
    """임베딩 모델을 로드하고 캐싱합니다."""
    return load_embedding_model()


@st.cache_resource
def load_or_create_vector_store(_embeddings: HuggingFaceEmbeddings, persist_directory: str = "vector_store"):
    """
    기존 벡터 저장소를 로드하거나, 없으면 새로 생성합니다.
    
    Args:
        _embeddings: 임베딩 모델 (Streamlit 캐싱을 위한 언더스코어 접두사)
        persist_directory: 벡터 저장소 디렉토리
    
    Returns:
        Chroma 벡터 저장소 객체
    """
    # 기존 벡터 저장소가 있는지 확인
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            # 기존 저장소 로드
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=_embeddings
            )
            st.success(f"✅ 기존 벡터 저장소를 로드했습니다.")
            return vector_store
        except Exception as e:
            st.warning(f"⚠️ 기존 저장소 로드 실패: {e}. 새로 생성합니다.")
    
    # 저장소가 없으면 새로 생성
    pdf_path = "data/sample_nursing.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        st.stop()
    
    with st.spinner("📚 문서를 로드하고 벡터 저장소를 생성하는 중..."):
        documents = load_and_split_pdf(pdf_path)
        vector_store = create_vector_store(documents, _embeddings, persist_directory)
        st.success(f"✅ {len(documents)}개의 문서로 벡터 저장소를 생성했습니다.")
    
    return vector_store


def search_documents(vector_store: Chroma, query: str, k: int = 3):
    """
    벡터 저장소에서 검색을 수행합니다.
    
    Args:
        vector_store: Chroma 벡터 저장소 객체
        query: 검색 질문
        k: 반환할 문서 개수
    
    Returns:
        검색 결과 리스트 (doc, score) 튜플
    """
    return vector_store.similarity_search_with_score(query, k=k)


def display_search_results(results, query: str):
    """
    검색 결과를 Streamlit에 표시합니다.
    
    Args:
        results: 검색 결과 리스트
        query: 검색 질문
    """
    if not results:
        st.warning("❌ 검색 결과가 없습니다.")
        return
    
    st.markdown(f"### 🔍 검색 결과 ({len(results)}개)")
    st.markdown(f"**질문:** {query}")
    st.markdown("---")
    
    for i, (doc, score) in enumerate(results, 1):
        # 결과 카드 스타일
        with st.container():
            st.markdown(f"#### 📄 결과 {i} (유사도: {score:.4f})")
            
            # 메타데이터 출력
            metadata = doc.metadata
            if metadata:
                st.markdown("**📍 출처 정보:**")
                metadata_text = " | ".join([f"**{key}**: {value}" for key, value in metadata.items()])
                st.markdown(f"   {metadata_text}")
            
            # 문서 내용 출력
            st.markdown("**📝 내용:**")
            content = doc.page_content
            # 내용을 코드 블록 스타일로 표시 (가독성 향상)
            st.markdown(f"```\n{content}\n```")
            
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
        간호 관련 문서에서 정보를 검색하는 시스템입니다.
        
        ### 사용 방법
        1. 아래 입력창에 질문을 입력하세요
        2. "검색" 버튼을 클릭하세요
        3. 검색된 문서와 출처 정보를 확인하세요
        """)
    
    # 초기화: 임베딩 모델 및 벡터 저장소 로드
    try:
        with st.spinner("🤖 임베딩 모델을 로드하는 중..."):
            embeddings = load_embeddings()
        
        vector_store = load_or_create_vector_store(embeddings)
        
    except Exception as e:
        st.error(f"❌ 시스템 초기화 중 오류가 발생했습니다: {e}")
        st.stop()
    
    # 사용자 입력 영역
    st.markdown("### 💬 질문 입력")
    query = st.text_input(
        "검색할 내용을 입력하세요:",
        placeholder="예: 활력징후 측정 절차",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 10])
    with col1:
        search_button = st.button("🔍 검색", type="primary", use_container_width=True)
    
    # 검색 실행
    if search_button:
        if not query.strip():
            st.warning("⚠️ 질문을 입력해주세요.")
        else:
            with st.spinner("🔍 검색 중..."):
                try:
                    results = search_documents(vector_store, query, k=3)
                    display_search_results(results, query)
                except Exception as e:
                    st.error(f"❌ 검색 중 오류가 발생했습니다: {e}")
    
    # 초기 안내 메시지
    elif not query:
        st.info("👆 위 입력창에 질문을 입력하고 검색 버튼을 클릭하세요.")


if __name__ == "__main__":
    main()

