"""기본 설정 모듈

이 모듈은 RAG 시스템의 전역 설정을 관리합니다.
환경 변수를 통해 API 키 및 모델 설정을 로드합니다.
"""
import os
from pathlib import Path
from typing import Optional

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
LAW_DIR = RAW_DATA_DIR / "law"
GUIDES_DIR = RAW_DATA_DIR / "guides"

# 인덱스 저장 경로
INDEX_DIR = DATA_ROOT / "index"
CHROMA_DIR = DATA_ROOT / "chroma"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
CHUNKS_JSON_PATH = INDEX_DIR / "chunks.json"  # 디버깅용 청킹 결과 저장
CHUNKS_TEXT_DIR = INDEX_DIR / "chunks_text"  # 텍스트 형식으로도 저장

# Google Gemini 설정
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding 모델 (기본값 사용 시 None)
LLM_MODEL = "gemini-2.5-pro"  # 또는 "gemini-1.5-pro" 필요시

# RAG 설정
CHUNK_SIZE = 1000  # 청크 크기 (문자 수)
CHUNK_OVERLAP = 200  # 청크 간 겹침
TOP_K_RETRIEVAL = 8  # 검색 시 반환할 문서 수

# Hybrid Retrieval 가중치
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# 문서 타입 매핑
DOCUMENT_TYPES = {
    "law": "법률",
    "decree": "시행령",
    "rule": "시행규칙",
    "guideline": "가이드라인",
    "faq": "FAQ",
    "case": "심결례",
}

# API 설정
API_HOST = "0.0.0.0"
API_PORT = 8000

