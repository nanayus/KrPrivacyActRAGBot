"""인덱스 빌드 스크립트

4.3 임베딩 및 인덱싱을 구현합니다.
벡터 스토어와 BM25 인덱스를 구축합니다.

사용법:
    python -m src.ingestion.build_index
"""
import json
import logging
import pickle
import time
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import (
    BM25_INDEX_PATH,
    CHROMA_DIR,
    CHUNKS_JSON_PATH,
    CHUNKS_TEXT_DIR,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    INDEX_DIR,
)
from src.ingestion.chunking import chunk_documents
from src.ingestion.loader import load_all_documents
from src.ingestion.preprocessing import preprocess_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indexes():
    """벡터 스토어와 BM25 인덱스를 구축합니다."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 디렉토리 생성
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("1. 문서 로딩 중...")
    documents = load_all_documents()
    
    if not documents:
        logger.warning("로드된 문서가 없습니다. data/raw/ 디렉토리에 PDF 파일을 추가하세요.")
        return
    
    logger.info("2. 전처리 중...")
    processed_docs = preprocess_documents(documents)
    
    logger.info("3. 청킹 중...")
    chunks = chunk_documents(processed_docs)
    
    logger.info(f"총 {len(chunks)}개의 청크 생성됨")
    
    # 3-1. 청킹 결과를 별도 파일로 저장 (디버깅용)
    logger.info("3-1. 청킹 결과 저장 중...")
    CHUNKS_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON 형식으로 저장
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunk_info = {
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "content_length": len(chunk.page_content),
        }
        chunks_data.append(chunk_info)
        
        # 텍스트 파일로도 저장 (개별 파일)
        chunk_file = CHUNKS_TEXT_DIR / f"chunk_{i:05d}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(f"=== 청크 ID: {i} ===\n")
            f.write(f"출처: {chunk.metadata.get('source', 'Unknown')}\n")
            f.write(f"문서 타입: {chunk.metadata.get('document_type', 'Unknown')}\n")
            f.write(f"청크 인덱스: {chunk.metadata.get('chunk_index', 'N/A')}\n")
            f.write(f"전체 청크 수: {chunk.metadata.get('total_chunks', 'N/A')}\n")
            f.write(f"내용 길이: {len(chunk.page_content)}자\n")
            f.write("=" * 50 + "\n\n")
            f.write(chunk.page_content)
    
    # JSON 파일로 저장
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": len(chunks),
            "chunks": chunks_data
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"청킹 결과 저장 완료:")
    logger.info(f"  - JSON: {CHUNKS_JSON_PATH}")
    logger.info(f"  - 텍스트 파일: {CHUNKS_TEXT_DIR} ({len(chunks)}개 파일)")
    
    # 4.3 임베딩 및 인덱싱
    logger.info("4. 벡터 스토어 구축 중...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    
    # 기존 Chroma 디렉토리가 있으면 삭제 (재구축)
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    # 재시도 로직 추가 (Google API 일시적 오류 대응)
    max_retries = 3
    retry_delay = 5  # 초
    
    for attempt in range(max_retries):
        try:
            logger.info(f"벡터 스토어 구축 시도 {attempt + 1}/{max_retries}...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            logger.info(f"벡터 스토어가 {CHROMA_DIR}에 저장되었습니다.")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"벡터 스토어 구축 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                logger.info(f"{retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 지수 백오프
            else:
                logger.error(f"벡터 스토어 구축 최종 실패: {e}")
                raise
    
    # BM25 인덱스 구축
    logger.info("5. BM25 인덱스 구축 중...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8  # 기본 k 값 설정
    
    # BM25 인덱스 저장 (문서 리스트와 k 값 저장)
    bm25_data = {
        "documents": chunks,
        "k": 8,
    }
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    
    logger.info(f"BM25 인덱스가 {BM25_INDEX_PATH}에 저장되었습니다.")
    
    logger.info("인덱스 구축 완료!")


if __name__ == "__main__":
    build_indexes()

