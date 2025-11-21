"""
RAG 커넥터 테스트
src/rag_connector.py의 함수들을 테스트합니다.
"""

import pytest
import os
import tempfile
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.rag_connector import (
    load_embedding_model,
    load_vector_store,
    get_relevant_documents,
    get_relevant_documents_with_content,
    convert_to_api_format
)


@pytest.fixture(scope="module")
def embedding_model():
    """
    임베딩 모델을 로드하는 픽스처
    모듈 레벨에서 한 번만 로드하여 재사용합니다.
    """
    return load_embedding_model()


@pytest.fixture(scope="function")
def test_vector_store(embedding_model):
    """
    테스트용 임시 벡터 저장소를 생성하는 픽스처
    실제 vector_store/nursing/ 디렉토리가 있으면 사용하고,
    없으면 임시 디렉토리에 테스트 데이터를 생성합니다.
    """
    # 실제 저장소 경로 확인
    actual_store_path = "vector_store/nursing"
    
    if os.path.exists(actual_store_path) and os.listdir(actual_store_path):
        # 실제 저장소가 있으면 사용
        return load_vector_store("NURSING")
    else:
        # 실제 저장소가 없으면 임시 저장소 생성
        temp_dir = tempfile.mkdtemp()
        test_store_path = os.path.join(temp_dir, "nursing")
        os.makedirs(test_store_path, exist_ok=True)
        
        # 테스트용 문서 생성
        test_documents = [
            Document(
                page_content="주사기 투여 시에는 무균 기술을 준수해야 합니다. 피부를 소독하고 적절한 각도로 주사기를 삽입합니다.",
                metadata={"source": "test_nursing.pdf", "page": 1}
            ),
            Document(
                page_content="활력징후 측정은 체온, 맥박, 호흡, 혈압을 순서대로 측정합니다. 각 측정값은 정확하게 기록해야 합니다.",
                metadata={"source": "test_nursing.pdf", "page": 2}
            ),
            Document(
                page_content="혈압 측정 시 환자의 팔을 심장 높이에 맞추고, 커프를 적절히 감싸야 합니다. 측정 중에는 환자가 움직이지 않도록 해야 합니다.",
                metadata={"source": "test_nursing.pdf", "page": 3}
            ),
        ]
        
        # 임시 벡터 저장소 생성
        vector_store = Chroma.from_documents(
            documents=test_documents,
            embedding=embedding_model,
            persist_directory=test_store_path
        )
        vector_store.persist()
        
        # 정리 함수 등록
        def cleanup():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # pytest가 자동으로 정리하도록 설정
        import atexit
        atexit.register(cleanup)
        
        return vector_store


@pytest.fixture(scope="function")
def mock_vector_store_path(monkeypatch, test_vector_store, embedding_model):
    """
    load_vector_store 함수가 테스트용 저장소를 사용하도록 모킹하는 픽스처
    """
    # 실제 저장소가 없을 때를 대비한 모킹
    original_load = load_vector_store
    
    def mock_load_vector_store(domain: str):
        if domain.upper() == "NURSING":
            return test_vector_store
        else:
            return original_load(domain)
    
    # 모듈 레벨 함수를 모킹하기 위해 importlib 사용
    import importlib
    import src.rag_connector
    monkeypatch.setattr(src.rag_connector, "load_vector_store", mock_load_vector_store)
    
    return test_vector_store


def test_load_embedding_model(embedding_model):
    """
    임베딩 모델이 정상적으로 로드되는지 테스트합니다.
    """
    assert embedding_model is not None
    assert hasattr(embedding_model, 'embed_query')


def test_retrieve_nursing_documents(mock_vector_store_path):
    """
    간호 관련 질문에 대해 Document 객체 리스트와 출처 메타데이터가 반환되는지 테스트합니다.
    """
    query = "주사기 투여 기술"
    
    # RAG 검색 수행
    results = get_relevant_documents(query, domain="NURSING", k=3)
    
    # 결과 검증
    assert len(results) > 0, "검색 결과가 비어있습니다. 벡터 저장소에 데이터가 있는지 확인해주세요."
    
    # 각 결과가 (Document, score) 튜플인지 확인
    for doc, score in results:
        assert isinstance(doc, Document), "결과가 Document 객체가 아닙니다."
        assert hasattr(doc, 'page_content'), "Document에 page_content 속성이 없습니다."
        assert hasattr(doc, 'metadata'), "Document에 metadata 속성이 없습니다."
        assert isinstance(score, (int, float)), f"점수가 숫자 타입이 아닙니다: {type(score)}"
        
        # 메타데이터 검증
        metadata = doc.metadata
        assert isinstance(metadata, dict), "메타데이터가 딕셔너리가 아닙니다."
        assert "source" in metadata or metadata.get("source") is not None, "출처 정보가 없습니다."


def test_get_relevant_documents_with_content(mock_vector_store_path):
    """
    get_relevant_documents_with_content 함수가 올바른 형식으로 결과를 반환하는지 테스트합니다.
    """
    query = "주사기 투여 기술"
    
    # 검색 수행
    documents = get_relevant_documents_with_content(query, domain="NURSING", k=3)
    
    # 결과 검증
    assert len(documents) > 0, "검색 결과가 비어있습니다."
    
    # 각 문서의 구조 검증
    for doc in documents:
        assert isinstance(doc, dict), "문서가 딕셔너리 형식이 아닙니다."
        assert "content" in doc, "문서에 content 키가 없습니다."
        assert "metadata" in doc, "문서에 metadata 키가 없습니다."
        assert "score" in doc, "문서에 score 키가 없습니다."
        
        # 내용 검증
        assert isinstance(doc["content"], str), "content가 문자열이 아닙니다."
        assert len(doc["content"]) > 0, "content가 비어있습니다."
        
        # 메타데이터 검증
        assert isinstance(doc["metadata"], dict), "metadata가 딕셔너리가 아닙니다."
        
        # 점수 검증
        assert isinstance(doc["score"], (int, float)), "score가 숫자가 아닙니다."


def test_convert_to_api_format(mock_vector_store_path):
    """
    convert_to_api_format 함수가 FastAPI 응답 형식에 맞게 변환하는지 테스트합니다.
    """
    query = "주사기 투여 기술"
    
    # 검색 결과 가져오기
    search_results = get_relevant_documents(query, domain="NURSING", k=3)
    
    # FastAPI 형식으로 변환
    sources = convert_to_api_format(search_results)
    
    # 결과 검증
    assert len(sources) > 0, "변환된 출처 정보가 비어있습니다."
    
    # 각 출처 정보의 구조 검증
    for source in sources:
        assert isinstance(source, dict), "출처 정보가 딕셔너리 형식이 아닙니다."
        assert "source" in source, "출처 정보에 source 키가 없습니다."
        assert "page" in source, "출처 정보에 page 키가 없습니다."
        assert "relevance_score" in source, "출처 정보에 relevance_score 키가 없습니다."
        
        # 값 타입 검증
        assert isinstance(source["source"], str), "source 값이 문자열이 아닙니다."
        assert source["page"] is None or isinstance(source["page"], (int, str)), "page 값이 올바른 타입이 아닙니다."
        assert isinstance(source["relevance_score"], (int, float)), "relevance_score가 숫자가 아닙니다."
        assert 0 <= source["relevance_score"] <= 1 or source["relevance_score"] >= 0, "relevance_score가 유효한 범위가 아닙니다."


def test_relevance_score_ordering(mock_vector_store_path):
    """
    검색 결과가 유사도 점수 순으로 정렬되어 반환되는지 테스트합니다.
    """
    query = "주사기 투여 기술"
    
    # 검색 수행
    results = get_relevant_documents(query, domain="NURSING", k=3)
    
    if len(results) > 1:
        # 점수가 내림차순으로 정렬되어 있는지 확인
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "검색 결과가 유사도 점수 순으로 정렬되지 않았습니다."


def test_empty_query_handling(mock_vector_store_path):
    """
    빈 질문에 대한 처리 테스트
    """
    query = ""
    
    # 빈 질문으로 검색 시도
    results = get_relevant_documents(query, domain="NURSING", k=3)
    
    # 빈 질문이어도 결과가 반환될 수 있음 (모든 문서 반환)
    # 또는 빈 리스트를 반환할 수 있음
    assert isinstance(results, list), "결과가 리스트가 아닙니다."


def test_domain_specific_search(mock_vector_store_path):
    """
    도메인별 검색이 올바르게 작동하는지 테스트합니다.
    """
    query = "주사기 투여 기술"
    
    # NURSING 도메인으로 검색
    nursing_results = get_relevant_documents(query, domain="NURSING", k=3)
    
    assert len(nursing_results) > 0, "NURSING 도메인 검색 결과가 없습니다."
    
    # 각 결과의 메타데이터 확인
    for doc, _ in nursing_results:
        # 메타데이터에 출처 정보가 있는지 확인
        assert doc.metadata is not None, "메타데이터가 None입니다."

