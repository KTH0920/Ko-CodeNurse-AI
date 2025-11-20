"""
RAG 커넥터 모듈
KoSimCSE 임베딩 모델과 ChromaDB를 사용한 실제 RAG 검색 기능 제공
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Optional
import os


# 하드코딩된 설정값
EMBEDDING_MODEL_NAME = "snunlp/KR-SCoS-NLI-KLUE-STS"
VECTOR_STORE_PATH = "vector_store"


# 전역 변수로 모델과 저장소 캐싱
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[Chroma] = None


def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    KoSimCSE 임베딩 모델을 로드합니다.
    모델은 전역 변수에 캐싱되어 재사용됩니다.
    
    Returns:
        HuggingFaceEmbeddings 객체
    """
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    return _embedding_model


def load_vector_store() -> Chroma:
    """
    ChromaDB 벡터 저장소를 로드합니다.
    저장소는 전역 변수에 캐싱되어 재사용됩니다.
    
    Returns:
        Chroma 벡터 저장소 객체
    
    Raises:
        FileNotFoundError: 벡터 저장소가 존재하지 않는 경우
    """
    global _vector_store
    
    if _vector_store is None:
        # 벡터 저장소 경로 확인
        if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"벡터 저장소를 찾을 수 없습니다: {VECTOR_STORE_PATH}\n"
                f"먼저 rag_integration.py를 실행하여 벡터 저장소를 생성해주세요."
            )
        
        # 임베딩 모델 로드
        embeddings = load_embedding_model()
        
        # 기존 벡터 저장소 로드
        _vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
    
    return _vector_store


def get_relevant_documents(query: str, k: int = 3) -> List[tuple]:
    """
    벡터 저장소에서 질문과 관련된 문서를 검색합니다.
    
    Args:
        query: 검색 질문
        k: 반환할 문서 개수 (기본값: 3)
    
    Returns:
        (Document, score) 튜플의 리스트
    """
    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def convert_to_api_format(search_results: List[tuple]) -> List[Dict]:
    """
    RAG 검색 결과를 FastAPI 응답 모델 형식으로 변환합니다.
    
    Args:
        search_results: (Document, score) 튜플의 리스트
    
    Returns:
        FastAPI GenerateResponse의 sources 형식에 맞는 딕셔너리 리스트
        각 딕셔너리는 source, page, relevance_score를 포함합니다.
    """
    sources = []
    
    for doc, score in search_results:
        metadata = doc.metadata
        
        source_info = {
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", None),
            "relevance_score": float(score)  # numpy 타입일 수 있으므로 float로 변환
        }
        
        sources.append(source_info)
    
    return sources


def get_relevant_documents_with_content(query: str, k: int = 3) -> List[Dict]:
    """
    벡터 저장소에서 질문과 관련된 문서를 검색하고, 
    내용과 메타데이터를 포함한 딕셔너리 리스트로 반환합니다.
    
    Args:
        query: 검색 질문
        k: 반환할 문서 개수 (기본값: 3)
    
    Returns:
        문서 내용과 메타데이터를 포함한 딕셔너리 리스트
        각 딕셔너리는 content, metadata, score를 포함합니다.
    """
    results = get_relevant_documents(query, k)
    
    documents = []
    for doc, score in results:
        documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })
    
    return documents

