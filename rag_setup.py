"""
RAG 시스템 초기 설정 스크립트
PDF 문서를 로드하고 청크로 분할하는 기능을 제공합니다.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os


def load_and_split_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    PDF 파일을 로드하고 지정된 크기로 분할합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        chunk_size: 청크 크기 (기본값: 800)
        chunk_overlap: 청크 간 오버랩 크기 (기본값: 80)
    
    Returns:
        Document 객체 리스트
    """
    # 파일 존재 여부 확인
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    
    # PDF 로더를 사용하여 문서 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 문서를 청크로 분할
    split_documents = text_splitter.split_documents(documents)
    
    return split_documents


if __name__ == "__main__":
    # 샘플 PDF 파일 경로
    pdf_path = "data/sample_nursing.pdf"
    
    try:
        # PDF 로드 및 분할
        documents = load_and_split_pdf(
            pdf_path=pdf_path,
            chunk_size=800,
            chunk_overlap=80
        )
        
        print(f"✅ 총 {len(documents)}개의 문서 청크가 생성되었습니다.")
        print(f"\n첫 번째 청크 미리보기:")
        print("-" * 50)
        if documents:
            print(f"페이지: {documents[0].metadata.get('page', 'N/A')}")
            print(f"내용 (처음 200자): {documents[0].page_content[:200]}...")
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print(f"💡 {pdf_path} 파일이 존재하는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

