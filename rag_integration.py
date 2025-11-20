"""
RAG 시스템 통합 및 테스트 스크립트
KoSimCSE 임베딩 모델과 ChromaDB를 사용하여 벡터 저장소를 구축하고 검색을 테스트합니다.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import os
from rag_setup import load_and_split_pdf


def load_embedding_model(model_name: str = "snunlp/KR-SCoS-NLI-KLUE-STS"):
    """
    KoSimCSE 임베딩 모델을 로드합니다.
    
    Args:
        model_name: HuggingFace 모델 이름 (기본값: snunlp/KR-SCoS-NLI-KLUE-STS)
    
    Returns:
        HuggingFaceEmbeddings 객체
    """
    print(f"🔄 임베딩 모델 로딩 중: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료")
    return embeddings


def create_vector_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = "vector_store"
) -> Chroma:
    """
    ChromaDB 벡터 저장소를 생성하고 문서를 저장합니다.
    
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


def test_rag_system(
    pdf_path: str = "data/sample_nursing.pdf",
    query: str = "활력징후 측정 절차"
):
    """
    RAG 시스템 전체 파이프라인을 테스트합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        query: 검색 테스트 질문
    """
    try:
        # 1. PDF 문서 로드 및 분할
        print("=" * 70)
        print("📚 1단계: PDF 문서 로드 및 분할")
        print("=" * 70)
        documents = load_and_split_pdf(pdf_path)
        print(f"✅ {len(documents)}개의 문서 청크 생성 완료\n")
        
        # 2. 임베딩 모델 로드
        print("=" * 70)
        print("🤖 2단계: KoSimCSE 임베딩 모델 로드")
        print("=" * 70)
        embeddings = load_embedding_model()
        print()
        
        # 3. 벡터 저장소 생성
        print("=" * 70)
        print("💾 3단계: ChromaDB 벡터 저장소 생성")
        print("=" * 70)
        vector_store = create_vector_store(documents, embeddings)
        print()
        
        # 4. 검색 테스트
        print("=" * 70)
        print("🔎 4단계: 검색 테스트")
        print("=" * 70)
        search_and_display(vector_store, query)
        
        print("\n" + "=" * 70)
        print("✅ RAG 시스템 테스트 완료!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print(f"💡 {pdf_path} 파일이 존재하는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # RAG 시스템 테스트 실행
    test_rag_system(
        pdf_path="data/sample_nursing.pdf",
        query="활력징후 측정 절차"
    )

