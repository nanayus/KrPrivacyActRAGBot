"""문서 로더 모듈

4.1 데이터 소스 및 수집을 구현합니다.
PDF 파일을 로드하고 메타데이터를 첨부합니다.
"""
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import GUIDES_DIR, LAW_DIR

logger = logging.getLogger(__name__)


def infer_document_type(file_path: Path) -> str:
    """파일 경로에서 문서 타입을 추론합니다.
    
    Args:
        file_path: 문서 파일 경로
        
    Returns:
        문서 타입 (law, guideline, faq 등)
    """
    path_str = str(file_path).lower()
    
    if "law" in path_str or "법률" in path_str:
        return "law"
    elif "decree" in path_str or "시행령" in path_str:
        return "decree"
    elif "rule" in path_str or "시행규칙" in path_str:
        return "rule"
    elif "faq" in path_str:
        return "faq"
    elif "case" in path_str or "심결례" in path_str:
        return "case"
    else:
        return "guideline"


def load_pdf(file_path: Path, document_type: Optional[str] = None) -> List[Document]:
    """PDF 파일을 로드하고 메타데이터를 첨부합니다.
    
    Args:
        file_path: PDF 파일 경로
        document_type: 문서 타입 (None이면 자동 추론)
        
    Returns:
        Document 객체 리스트
    """
    if document_type is None:
        document_type = infer_document_type(file_path)
    
    logger.info(f"Loading PDF: {file_path}")
    
    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        # 각 문서에 메타데이터 추가
        for doc in documents:
            doc.metadata.update({
                "document_type": document_type,
                "source_file": str(file_path),
                "source": file_path.name,
                "title": file_path.stem,  # 파일명에서 확장자 제거
            })
        
        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def load_all_documents(data_dir: Optional[Path] = None) -> List[Document]:
    """지정된 디렉토리의 모든 PDF 파일을 로드합니다.
    
    Args:
        data_dir: 데이터 디렉토리 (None이면 config의 기본 경로 사용)
        
    Returns:
        모든 문서의 Document 리스트
    """
    if data_dir is None:
        # LAW_DIR과 GUIDES_DIR 모두에서 로드
        data_dirs = [LAW_DIR, GUIDES_DIR]
    else:
        data_dirs = [data_dir]
    
    all_documents = []
    
    for base_dir in data_dirs:
        if not base_dir.exists():
            logger.warning(f"Directory does not exist: {base_dir}")
            continue
        
        # PDF 파일 찾기
        pdf_files = list(base_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {base_dir}")
        
        for pdf_file in pdf_files:
            documents = load_pdf(pdf_file)
            all_documents.extend(documents)
    
    logger.info(f"Total loaded documents: {len(all_documents)}")
    return all_documents


