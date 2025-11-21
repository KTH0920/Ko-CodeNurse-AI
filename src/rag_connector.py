"""
RAG 커넥터 모듈
KoSimCSE 임베딩 모델과 ChromaDB를 사용한 실제 RAG 검색 기능 제공
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os


# 하드코딩된 설정값
EMBEDDING_MODEL_NAME = "BM-K/KoSimCSE-roberta-multitask"  # 한국어 임베딩 모델
VECTOR_STORE_BASE_PATH = "vector_store"


# 전역 변수로 모델과 저장소 캐싱
_embedding_model: Optional[HuggingFaceEmbeddings] = None
# 도메인별 벡터 저장소 캐싱 (도메인명을 키로 사용)
_vector_stores: Dict[str, Chroma] = {}


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


def load_vector_store(domain: str) -> Chroma:
    """
    도메인별 ChromaDB 벡터 저장소를 로드합니다.
    저장소는 전역 딕셔너리에 도메인별로 캐싱되어 재사용됩니다.
    
    Args:
        domain: 도메인 이름 (예: "NURSING", "RESEARCH")
    
    Returns:
        Chroma 벡터 저장소 객체
    
    Raises:
        FileNotFoundError: 벡터 저장소가 존재하지 않는 경우
    """
    global _vector_stores
    
    # 도메인명을 소문자로 변환하여 경로 생성
    domain_lower = domain.lower()
    vector_store_path = os.path.join(VECTOR_STORE_BASE_PATH, domain_lower)
    
    # 이미 로드된 저장소가 있으면 반환
    if domain_lower in _vector_stores:
        return _vector_stores[domain_lower]
    
    # 벡터 저장소 경로 확인
    if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
        raise FileNotFoundError(
            f"벡터 저장소를 찾을 수 없습니다: {vector_store_path}\n"
            f"먼저 rag_integration.py를 수정하여 {domain} 도메인용 벡터 저장소를 생성해주세요.\n"
            f"참고: 도메인별 저장소는 'vector_store/{domain_lower}/' 경로에 생성되어야 합니다."
        )
    
    # 임베딩 모델 로드 (전역에서 공유)
    embeddings = load_embedding_model()
    
    # 도메인별 벡터 저장소 로드
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings
    )
    
    # 캐싱
    _vector_stores[domain_lower] = vector_store
    
    return vector_store


def get_relevant_documents(query: str, domain: str, k: int = 3) -> List[tuple]:
    """
    도메인별 벡터 저장소에서 질문과 관련된 문서를 검색합니다.
    
    Args:
        query: 검색 질문
        domain: 도메인 이름 (예: "NURSING", "RESEARCH")
        k: 반환할 문서 개수 (기본값: 3)
    
    Returns:
        (Document, score) 튜플의 리스트
    """
    vector_store = load_vector_store(domain)
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


def get_relevant_documents_with_content(query: str, domain: str, k: int = 3) -> List[Dict]:
    """
    도메인별 벡터 저장소에서 질문과 관련된 문서를 검색하고, 
    내용과 메타데이터를 포함한 딕셔너리 리스트로 반환합니다.
    
    Args:
        query: 검색 질문
        domain: 도메인 이름 (예: "NURSING", "RESEARCH")
        k: 반환할 문서 개수 (기본값: 3)
    
    Returns:
        문서 내용과 메타데이터를 포함한 딕셔너리 리스트
        각 딕셔너리는 content, metadata, score를 포함합니다.
    """
    results = get_relevant_documents(query, domain, k)
    
    documents = []
    for doc, score in results:
        documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })
    
    return documents

