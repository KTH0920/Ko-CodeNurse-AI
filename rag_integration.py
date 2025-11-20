"""
RAG 시스템 통합 및 테스트 스크립트
KoSimCSE 임베딩 모델과 ChromaDB를 사용하여 도메인별 벡터 저장소를 구축합니다.

이 스크립트는 도메인별로 분리된 벡터 저장소를 생성합니다:
- NURSING 도메인: vector_store/nursing/
- RESEARCH 도메인: vector_store/research/
"""

import sys
import os
from typing import List
from langchain.schema import Document

# 상위 디렉토리를 경로에 추가하여 src 모듈 import 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_setup import load_and_split_pdf
from src.rag_connector import load_embedding_model
from langchain_community.vectorstores import Chroma


def load_and_split_nursing_docs(
    pdf_path: str = "data/sample_nursing.pdf",
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    간호 관련 문서를 로드하고 분할합니다.
    
    Args:
        pdf_path: 간호 문서 PDF 파일 경로 (기본값: data/sample_nursing.pdf)
        chunk_size: 청크 크기 (기본값: 800)
        chunk_overlap: 청크 간 오버랩 크기 (기본값: 80)
    
    Returns:
        Document 객체 리스트
    """
    print(f"📚 간호 문서 로드 중: {pdf_path}")
    documents = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
    print(f"✅ 간호 문서 {len(documents)}개 청크 생성 완료")
    return documents


def load_and_split_research_docs(
    pdf_path: str = "data/sample_research.pdf",
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    연구 관련 문서를 로드하고 분할합니다.
    
    Args:
        pdf_path: 연구 문서 PDF 파일 경로 (기본값: data/sample_research.pdf)
        chunk_size: 청크 크기 (기본값: 800)
        chunk_overlap: 청크 간 오버랩 크기 (기본값: 80)
    
    Returns:
        Document 객체 리스트
    """
    print(f"📚 연구 문서 로드 중: {pdf_path}")
    documents = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
    print(f"✅ 연구 문서 {len(documents)}개 청크 생성 완료")
    return documents


def create_vector_store(
    documents: List[Document],
    embeddings,
    persist_directory: str = "vector_store"
) -> Chroma:
    """
    ChromaDB 벡터 저장소를 생성하고 문서를 저장합니다.
    (src.rag_connector의 함수를 재사용하기 위해 동일한 인터페이스 제공)
    
    Args:
        documents: 저장할 Document 객체 리스트
        embeddings: 임베딩 모델
        persist_directory: 벡터 저장소 저장 디렉토리 (기본값: vector_store)
    
    Returns:
        Chroma 벡터 저장소 객체
    """
    print(f"\n🔄 벡터 저장소 생성 중... (총 {len(documents)}개 문서)")
    
    # vector_store 디렉토리 생성
    os.makedirs(persist_directory, exist_ok=True)
    
    # ChromaDB에 문서 저장
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # 저장소 영구 저장
    vector_store.persist()
    
    print(f"✅ 벡터 저장소 생성 완료: {persist_directory}/")
    return vector_store


def search_and_display(
    vector_store: Chroma,
    query: str,
    k: int = 3
):
    """
    벡터 저장소에서 검색을 수행하고 결과를 출력합니다.
    
    Args:
        vector_store: Chroma 벡터 저장소 객체
        query: 검색 질문
        k: 반환할 문서 개수 (기본값: 3)
    """
    print(f"\n🔍 검색 질문: {query}")
    print("=" * 70)
    
    # 유사도 검색 수행
    results = vector_store.similarity_search_with_score(query, k=k)
    
    if not results:
        print("❌ 검색 결과가 없습니다.")
        return
    
    print(f"\n📋 검색 결과 ({len(results)}개):\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{'─' * 70}")
        print(f"📄 결과 {i} (유사도 점수: {score:.4f})")
        print(f"{'─' * 70}")
        
        # 메타데이터 출력
        metadata = doc.metadata
        print(f"📍 출처 정보:")
        for key, value in metadata.items():
            print(f"   - {key}: {value}")
        
        # 문서 내용 출력
        print(f"\n📝 내용:")
        content = doc.page_content
        # 내용이 너무 길면 일부만 출력
        if len(content) > 500:
            print(f"   {content[:500]}...")
        else:
            print(f"   {content}")
        print()


if __name__ == "__main__":
    """
    도메인별 벡터 저장소 생성 메인 실행 블록
    NURSING과 RESEARCH 도메인별로 별도의 저장소를 생성합니다.
    """
    try:
        print("=" * 70)
        print("🚀 도메인별 RAG 벡터 저장소 생성 시작")
        print("=" * 70)
        
        # 1. 임베딩 모델 로드 (전역 공유)
        print("\n" + "=" * 70)
        print("🤖 1단계: KoSimCSE 임베딩 모델 로드")
        print("=" * 70)
        embeddings = load_embedding_model()
        print()
        
        # 2. NURSING 도메인 문서 로드 및 저장소 생성
        print("=" * 70)
        print("📚 2단계: NURSING 도메인 문서 처리")
        print("=" * 70)
        nursing_documents = load_and_split_nursing_docs(
            pdf_path="data/sample_nursing.pdf",
            chunk_size=800,
            chunk_overlap=80
        )
        
        if nursing_documents:
            nursing_store_path = "vector_store/nursing"
            print(f"\n💾 NURSING 도메인 벡터 저장소 생성 중...")
            create_vector_store(
                documents=nursing_documents,
                embeddings=embeddings,
                persist_directory=nursing_store_path
            )
            print(f"✅ NURSING 도메인 저장소 생성 완료: {nursing_store_path}/\n")
        else:
            print("⚠️ 경고: NURSING 문서가 없습니다. 저장소 생성을 건너뜁니다.\n")
        
        # 3. RESEARCH 도메인 문서 로드 및 저장소 생성
        print("=" * 70)
        print("📚 3단계: RESEARCH 도메인 문서 처리")
        print("=" * 70)
        research_documents = load_and_split_research_docs(
            pdf_path="data/sample_research.pdf",
            chunk_size=800,
            chunk_overlap=80
        )
        
        if research_documents:
            research_store_path = "vector_store/research"
            print(f"\n💾 RESEARCH 도메인 벡터 저장소 생성 중...")
            create_vector_store(
                documents=research_documents,
                embeddings=embeddings,
                persist_directory=research_store_path
            )
            print(f"✅ RESEARCH 도메인 저장소 생성 완료: {research_store_path}/\n")
        else:
            print("⚠️ 경고: RESEARCH 문서가 없습니다. 저장소 생성을 건너뜁니다.\n")
        
        # 4. 완료 메시지
        print("=" * 70)
        print("✅ 도메인별 벡터 저장소 생성 완료!")
        print("=" * 70)
        print("\n생성된 저장소:")
        if nursing_documents:
            print(f"  - NURSING: vector_store/nursing/ ({len(nursing_documents)}개 문서)")
        if research_documents:
            print(f"  - RESEARCH: vector_store/research/ ({len(research_documents)}개 문서)")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("💡 PDF 파일이 존재하는지 확인해주세요:")
        print("   - data/sample_nursing.pdf")
        print("   - data/sample_research.pdf")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

