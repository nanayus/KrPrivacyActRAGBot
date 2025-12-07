"""하이브리드 검색 모듈

5.1 Retrieval 최적화를 구현합니다.
Dense + BM25 하이브리드 검색을 제공합니다.
LangChain EnsembleRetriever 기반 + 후처리 방식으로 구현됩니다.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# EnsembleRetriever는 langchain.retrievers 또는 langchain_community.retrievers에 있을 수 있음
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        raise ImportError(
            "EnsembleRetriever를 찾을 수 없습니다. "
            "langchain 또는 langchain-community 패키지가 설치되어 있는지 확인하세요."
        )

from src.config import (
    BM25_INDEX_PATH,
    CHROMA_DIR,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    SPARSE_WEIGHT,
    TOP_K_RETRIEVAL,
)

logger = logging.getLogger(__name__)


def _make_doc_key(doc: Document) -> str:
    """Dense/BM25 모두에서 동일한 청크를 가리키는 안정적인 키 생성"""
    meta = doc.metadata or {}
    
    # 여러 필드를 시도하여 키 생성
    # 1순위: source_file + 조 + chunk_index
    source = meta.get("source_file", "")
    jo = meta.get("조", "")
    chunk_idx = meta.get("chunk_index", "")
    
    # source_file이 없으면 source 시도
    if not source:
        source = meta.get("source", "")
    
    # chunk_index가 없으면 다른 필드 시도
    if chunk_idx == "":
        chunk_idx = meta.get("chunk_id", "")
    
    # 최종 키 생성
    key = f"{source}::조={jo}::chunk={chunk_idx}"
    
    # 키가 비어있거나 너무 짧으면 page_content 해시 사용 (fallback)
    if not key or key == "::조=::chunk=" or len(key) < 10:
        import hashlib
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:16]
        key = f"content_hash::{content_hash}"
        logger.warning(f"[키 생성] 메타데이터 부족, content_hash 사용: {key[:50]}...")
        print(f"[키 생성] 메타데이터 부족, content_hash 사용")
    
    return key

# 동의어 확장 딕셔너리 (도메인 핵심 동의어만 유지)
# 일반적인 단어("정의", "개인정보" 등)는 제거하여 과도한 확장 방지
SYNONYM_DICT = {
    "보유기간": ["보존기간", "보관기간", "삭제 기준"],
    "제3자 제공": ["개인정보 제공", "이용 제공"],
    "위탁": ["위탁처리", "개인정보처리 위탁"],
    "국외이전": ["해외이전", "국외 전송"],
    # "정의", "개인정보의 정의", "개인정보" 제거 - 너무 광범위하게 매칭됨
}


def expand_query_synonyms(query: str) -> str:
    """쿼리에 동의어를 확장합니다.
    
    5.1 Retrieval 최적화의 쿼리 리라이트 기능입니다.
    
    Args:
        query: 원본 쿼리
        
    Returns:
        확장된 쿼리
    """
    logger.info(f"[쿼리 확장] 원본 쿼리: '{query}'")
    print(f"[쿼리 확장] 원본 쿼리: '{query}'")
    
    expanded = query
    applied_changes = []
    
    # 긴 패턴부터 매칭 (더 구체적인 패턴 우선)
    sorted_keys = sorted(SYNONYM_DICT.keys(), key=len, reverse=True)
    logger.info(f"[쿼리 확장] 동의어 딕셔너리 키 개수: {len(SYNONYM_DICT)}, 검색 순서: {sorted_keys}")
    print(f"[쿼리 확장] 동의어 딕셔너리 키 개수: {len(SYNONYM_DICT)}, 검색 순서: {sorted_keys}")
    
    for key in sorted_keys:
        if key in query:
            # 동의어를 추가 (BM25에서 OR 검색 효과)
            synonyms = SYNONYM_DICT[key]
            synonym_str = " ".join(synonyms)
            before = expanded
            expanded = expanded.replace(key, f"{key} {synonym_str}")
            applied_changes.append({
                "matched_key": key,
                "synonyms": synonyms,
                "before": before,
                "after": expanded
            })
            logger.info(f"[쿼리 확장] 매칭 발견: '{key}' -> 동의어 추가: {synonyms}")
            logger.info(f"[쿼리 확장] 변경 전: '{before}'")
            logger.info(f"[쿼리 확장] 변경 후: '{expanded}'")
            print(f"[쿼리 확장] 매칭 발견: '{key}' -> 동의어 추가: {synonyms}")
            print(f"[쿼리 확장] 변경 전: '{before}'")
            print(f"[쿼리 확장] 변경 후: '{expanded}'")
    
    if not applied_changes:
        logger.info(f"[쿼리 확장] 매칭된 키워드 없음, 원본 쿼리 유지")
        print(f"[쿼리 확장] 매칭된 키워드 없음, 원본 쿼리 유지")
    else:
        logger.info(f"[쿼리 확장] 총 {len(applied_changes)}개 키워드 확장 완료")
        print(f"[쿼리 확장] 총 {len(applied_changes)}개 키워드 확장 완료")
    
    logger.info(f"[쿼리 확장] 최종 확장된 쿼리: '{expanded}'")
    print(f"[쿼리 확장] 최종 확장된 쿼리: '{expanded}'")
    
    return expanded


def normalize_jo_title_category(jo_title: str) -> str:
    """조항제목을 정규화된 카테고리로 매핑합니다.
    
    Args:
        jo_title: 조항제목
        
    Returns:
        카테고리 ("definition", "purpose", "general" 등)
    """
    if not jo_title:
        return "general"
    
    # 정의 조문 인식: "용어", "뜻" 등이 포함된 경우
    if "용어" in jo_title or "뜻" in jo_title or "정의" in jo_title:
        return "definition"
    
    if "목적" in jo_title:
        return "purpose"
    
    return "general"


def is_definition_query(query: str) -> bool:
    """쿼리가 정의 관련 질문인지 확인합니다.
    
    Args:
        query: 사용자 쿼리
        
    Returns:
        정의 관련 질문 여부
    """
    definition_keywords = ["정의", "개인정보 정의", "개인정보란", "개인정보란 무엇", "뜻", "의미", "개념"]
    query_lower = query.lower()
    return any(keyword in query for keyword in definition_keywords)


def find_matching_jo_title_keywords(query: str) -> List[str]:
    """쿼리에서 "조항제목"과 매칭될 수 있는 동의어 딕셔너리 키를 찾습니다.
    
    Args:
        query: 사용자 쿼리
        
    Returns:
        "조항제목"에 포함될 수 있는 키워드 리스트
    """
    logger.info(f"[조항제목 매칭] 쿼리 분석 시작: '{query}'")
    print(f"[조항제목 매칭] 쿼리 분석 시작: '{query}'")
    
    matching_keys = []
    
    # 긴 패턴부터 매칭 (더 구체적인 패턴 우선)
    sorted_keys = sorted(SYNONYM_DICT.keys(), key=len, reverse=True)
    logger.info(f"[조항제목 매칭] 검색할 키워드 리스트: {sorted_keys}")
    print(f"[조항제목 매칭] 검색할 키워드 리스트: {sorted_keys}")
    
    for key in sorted_keys:
        if key in query:
            matching_keys.append(key)
            logger.info(f"[조항제목 매칭] 키워드 발견: '{key}' (쿼리에 포함됨)")
            print(f"[조항제목 매칭] 키워드 발견: '{key}' (쿼리에 포함됨)")
    
    if matching_keys:
        logger.info(f"[조항제목 매칭] 총 {len(matching_keys)}개 키워드 매칭: {matching_keys}")
        print(f"[조항제목 매칭] 총 {len(matching_keys)}개 키워드 매칭: {matching_keys}")
    else:
        logger.info(f"[조항제목 매칭] 매칭된 키워드 없음")
        print(f"[조항제목 매칭] 매칭된 키워드 없음")
    
    return matching_keys


def postprocess_ensemble_results(
    documents: List[Document],
    query: str,
    vectorstore: Optional[Chroma],
    dense_retriever: BaseRetriever,
    bm25_retriever: BM25Retriever,
    dense_weight: float,
    sparse_weight: float,
) -> List[Document]:
    """EnsembleRetriever 결과를 후처리합니다.
    
    점수 계산 및 메타데이터 저장을 수행합니다.
    
    Args:
        documents: EnsembleRetriever에서 반환된 문서 리스트
        query: 검색 쿼리
        vectorstore: Chroma vectorstore (점수 계산용)
        dense_retriever: Dense retriever
        bm25_retriever: BM25 retriever
        dense_weight: Dense 가중치
        sparse_weight: Sparse 가중치
        
    Returns:
        점수가 메타데이터에 저장된 문서 리스트
    """
    logger.info(f"[후처리] {len(documents)}개 문서 후처리 시작")
    print(f"[후처리] {len(documents)}개 문서 후처리 시작")
    
    # 문서를 키로 매핑
    doc_dict = {}
    for doc in documents:
        key = _make_doc_key(doc)
        doc_dict[key] = doc
    
    # Dense 점수 계산
    if vectorstore is not None:
        try:
            dense_results = vectorstore.similarity_search_with_score(query, k=len(documents) * 2)
            logger.info(f"[후처리] Dense 검색: {len(dense_results)}개 결과")
            print(f"[후처리] Dense 검색: {len(dense_results)}개 결과")
            
            # Min-Max 정규화를 위한 거리 점수 수집
            distances = [abs(score) for _, score in dense_results]
            if distances:
                min_dist = min(distances)
                max_dist = max(distances)
                range_dist = max_dist - min_dist if max_dist > min_dist else 1.0
            else:
                min_dist = 0.0
                range_dist = 1.0
            
            for doc, score in dense_results:
                key = _make_doc_key(doc)
                if key in doc_dict:
                    # Min-Max 정규화를 사용한 거리 → 유사도 변환
                    normalized_dist = (abs(score) - min_dist) / range_dist if range_dist > 0 else 0.0
                    similarity = 1.0 - normalized_dist
                    doc_dict[key].metadata["dense_score"] = float(similarity)
        except Exception as e:
            logger.error(f"[후처리] Dense 점수 계산 오류: {e}", exc_info=True)
            print(f"[후처리] Dense 점수 계산 오류: {e}")
    
    # BM25 점수 계산 (순위 기반)
    try:
        bm25_docs = bm25_retriever.get_relevant_documents(query)
        max_rank = len(bm25_docs)
        
        for rank, doc in enumerate(bm25_docs, 1):
            key = _make_doc_key(doc)
            if key in doc_dict:
                # 순위 기반 정규화 점수
                normalized_score = (max_rank - rank + 1) / max_rank if max_rank > 0 else 0.0
                doc_dict[key].metadata["sparse_score"] = float(normalized_score)
    except Exception as e:
        logger.error(f"[후처리] BM25 점수 계산 오류: {e}", exc_info=True)
        print(f"[후처리] BM25 점수 계산 오류: {e}")
    
    # 하이브리드 점수 계산 및 메타데이터 저장
    for key, doc in doc_dict.items():
        dense_score = doc.metadata.get("dense_score", 0.0)
        sparse_score = doc.metadata.get("sparse_score", 0.0)
        final_score = dense_score * dense_weight + sparse_score * sparse_weight
        doc.metadata["final_score"] = float(final_score)
    
    logger.info(f"[후처리] 후처리 완료")
    print(f"[후처리] 후처리 완료")
    
    return documents


def reorder_by_jo_title_priority(
    documents: List[Document],
    query: str,
    matching_keywords: List[str],
) -> List[Document]:
    """조항제목 매칭 우선순위에 따라 문서를 재정렬합니다.
    
    Args:
        documents: 문서 리스트
        query: 검색 쿼리
        matching_keywords: 매칭된 키워드 리스트
        
    Returns:
        우선순위가 적용된 문서 리스트
    """
    is_def_query = is_definition_query(query)
    
    if not (matching_keywords or is_def_query) or not documents:
        return documents
    
    logger.info(f"[우선순위 재정렬] 조항제목 매칭 우선순위 적용 (정의 쿼리: {is_def_query})")
    print(f"[우선순위 재정렬] 조항제목 매칭 우선순위 적용 (정의 쿼리: {is_def_query})")
    
    prioritized_docs = []
    remaining_docs = []
    
    for doc in documents:
        jo_title = doc.metadata.get("조항제목", "")
        jo_num = doc.metadata.get("조", None)
        is_matched = False
        
        # 1. 제2조 특별 처리 (정의 조문)
        if is_def_query and jo_num == 2:
            prioritized_docs.append(doc)
            is_matched = True
            logger.debug(f"[우선순위] 제2조 매칭 - 조: {jo_num}, 조항제목: '{jo_title}'")
            print(f"[우선순위] 제2조 매칭 - 조: {jo_num}, 조항제목: '{jo_title}'")
        
        # 2. 조항제목 카테고리 매칭
        if not is_matched and is_def_query:
            category = normalize_jo_title_category(jo_title)
            if category == "definition":
                prioritized_docs.append(doc)
                is_matched = True
                logger.debug(f"[우선순위] 정의 카테고리 매칭 - 조: {jo_num}, 조항제목: '{jo_title}'")
                print(f"[우선순위] 정의 카테고리 매칭 - 조: {jo_num}, 조항제목: '{jo_title}'")
        
        # 3. 기존 키워드 매칭
        if not is_matched and matching_keywords:
            for keyword in matching_keywords:
                if keyword in jo_title:
                    prioritized_docs.append(doc)
                    is_matched = True
                    logger.debug(f"[우선순위] 키워드 매칭 - 조: {jo_num}, 조항제목: '{jo_title}', 키워드: '{keyword}'")
                    print(f"[우선순위] 키워드 매칭 - 조: {jo_num}, 조항제목: '{jo_title}', 키워드: '{keyword}'")
                    break
        
        if not is_matched:
            remaining_docs.append(doc)
    
    # 우선순위 문서를 앞에 배치
    result = prioritized_docs + remaining_docs
    
    if prioritized_docs:
        logger.info(f"[우선순위 재정렬] 완료 - {len(prioritized_docs)}개 우선, {len(remaining_docs)}개 일반")
        print(f"[우선순위 재정렬] 완료 - {len(prioritized_docs)}개 우선, {len(remaining_docs)}개 일반")
    
    return result


class PostProcessedRetriever(BaseRetriever):
    """후처리 로직을 포함한 Retriever 래퍼
    
    retrieve_documents() 함수를 사용하여 쿼리 확장, 점수 계산, 우선순위 재정렬을 수행합니다.
    """
    
    def __init__(
        self,
        ensemble_retriever: EnsembleRetriever,
        vectorstore: Chroma,
        dense_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        top_k: int = TOP_K_RETRIEVAL,
    ):
        # BaseRetriever 초기화 먼저
        super().__init__()
        # __dict__를 직접 조작하여 Pydantic 검증 우회
        self.__dict__['ensemble_retriever'] = ensemble_retriever
        self.__dict__['vectorstore'] = vectorstore
        self.__dict__['dense_retriever'] = dense_retriever
        self.__dict__['bm25_retriever'] = bm25_retriever
        self.__dict__['top_k'] = top_k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """후처리 로직을 포함한 문서 검색"""
        # __dict__에서 직접 속성 가져오기
        return retrieve_documents(
            query=query,
            top_k=self.__dict__['top_k'],
            ensemble_retriever=self.__dict__['ensemble_retriever'],
            vectorstore=self.__dict__['vectorstore'],
            dense_retriever=self.__dict__['dense_retriever'],
            bm25_retriever=self.__dict__['bm25_retriever'],
        )
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """비동기 버전 (동기 버전 사용)"""
        return self._get_relevant_documents(query)


def load_hybrid_retriever(
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    top_k: int = TOP_K_RETRIEVAL,
    use_postprocessing: bool = True,
) -> Tuple[BaseRetriever, Chroma, BaseRetriever, BM25Retriever]:
    """하이브리드 검색기를 로드합니다.
    
    4.3 임베딩 및 인덱싱에서 구축한 인덱스를 사용합니다.
    LangChain EnsembleRetriever를 사용합니다.
    
    Args:
        dense_weight: Dense retriever 가중치
        sparse_weight: Sparse (BM25) retriever 가중치
        top_k: 반환할 문서 수
        
    Returns:
        (EnsembleRetriever, Chroma vectorstore, Dense retriever, BM25 retriever) 튜플
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # Dense retriever (Chroma)
    logger.info("Dense retriever 로딩 중...")
    print("[진행] Dense retriever (Chroma) 로딩 중...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    
    try:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"벡터 스토어 디렉토리가 없습니다: {CHROMA_DIR}. "
                f"먼저 python -m src.ingestion.build_index를 실행하세요."
            )
        
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        
        logger.info("벡터 스토어 로드 완료")
        print("[진행] 벡터 스토어 로드 완료")
        
        dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"벡터 스토어 로드 중 오류: {e}", exc_info=True)
        print(f"[오류] 벡터 스토어 로드 실패: {e}")
        raise ValueError(f"벡터 스토어 로드 실패: {str(e)}. 인덱스를 먼저 구축하세요.")
    
    # Sparse retriever (BM25)
    logger.info("BM25 retriever 로딩 중...")
    print("[진행] BM25 retriever 로딩 중...")
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 인덱스가 없습니다. 먼저 python -m src.ingestion.build_index를 실행하세요."
        )
    
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    
    bm25_retriever = BM25Retriever.from_documents(bm25_data["documents"])
    bm25_retriever.k = top_k
    print("[진행] BM25 retriever 로드 완료")
    
    # LangChain EnsembleRetriever 사용
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )
    
    logger.info("하이브리드 검색기 로드 완료")
    print("[진행] 하이브리드 검색기 구성 완료")
    
    # 후처리 로직을 포함한 retriever 반환
    if use_postprocessing:
        retriever = PostProcessedRetriever(
            ensemble_retriever=ensemble_retriever,
            vectorstore=vectorstore,
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            top_k=top_k,
        )
        logger.info("후처리 로직이 포함된 retriever 생성 완료")
        return retriever, vectorstore, dense_retriever, bm25_retriever
    else:
        return ensemble_retriever, vectorstore, dense_retriever, bm25_retriever


def retrieve_documents(
    query: str,
    retriever: Optional[BaseRetriever] = None,
    expand_synonyms: bool = True,
    metadata_filter: Optional[Dict] = None,
    top_k: int = TOP_K_RETRIEVAL,
    ensemble_retriever: Optional[EnsembleRetriever] = None,
    vectorstore: Optional[Chroma] = None,
    dense_retriever: Optional[BaseRetriever] = None,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> List[Document]:
    """쿼리에 대해 문서를 검색합니다.
    
    5.1 Retrieval 최적화를 구현합니다.
    LangChain EnsembleRetriever 기반 + 후처리 방식으로 구현됩니다.
    "조항제목"에 동의어 딕셔너리 키가 포함된 청크를 우선적으로 검색합니다.
    
    Args:
        query: 사용자 쿼리
        retriever: 검색기 (하위 호환성을 위해 유지, 사용되지 않음)
        expand_synonyms: 동의어 확장 여부
        metadata_filter: 메타데이터 필터 (미구현, 향후 확장용)
        top_k: 반환할 문서 수
        ensemble_retriever: EnsembleRetriever (None이면 새로 로드)
        vectorstore: Chroma vectorstore (None이면 새로 로드)
        dense_retriever: Dense retriever (None이면 새로 로드)
        bm25_retriever: BM25 retriever (None이면 새로 로드)
        
    Returns:
        검색된 Document 리스트 ("조항제목" 매칭 청크가 우선순위)
    """
    # 검색기 로드
    if ensemble_retriever is None or vectorstore is None or dense_retriever is None or bm25_retriever is None:
        ensemble_retriever, vectorstore, dense_retriever, bm25_retriever = load_hybrid_retriever(top_k=top_k)
    
    logger.info(f"[검색 시작] 원본 쿼리: '{query}'")
    print(f"[검색 시작] 원본 쿼리: '{query}'")
    
    # 쿼리 확장
    if expand_synonyms:
        expanded_query = expand_query_synonyms(query)
        logger.info(f"[쿼리 확장] 확장된 쿼리: '{expanded_query}'")
        print(f"[쿼리 확장] 확장된 쿼리: '{expanded_query}'")
    else:
        expanded_query = query
    
    # "조항제목" 매칭 키워드 찾기
    matching_keywords_original = find_matching_jo_title_keywords(query)
    matching_keywords_expanded = find_matching_jo_title_keywords(expanded_query)
    matching_keywords = list(set(matching_keywords_original + matching_keywords_expanded))
    
    # EnsembleRetriever로 검색 실행
    logger.info(f"[검색 실행] 확장된 쿼리로 검색 시작: '{expanded_query}'")
    print(f"[검색 실행] 확장된 쿼리로 검색 시작: '{expanded_query}'")
    documents = ensemble_retriever.get_relevant_documents(expanded_query)
    logger.info(f"[검색 실행] 검색 완료, {len(documents)}개 문서 반환")
    print(f"[검색 실행] 검색 완료, {len(documents)}개 문서 반환")
    
    # 후처리: 점수 계산 및 메타데이터 저장
    documents = postprocess_ensemble_results(
        documents=documents,
        query=expanded_query,
        vectorstore=vectorstore,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_weight=DENSE_WEIGHT,
        sparse_weight=SPARSE_WEIGHT,
    )
    
    # 상위 k개 선택 (점수 기준)
    if len(documents) > top_k:
        # final_score 기준으로 정렬
        documents = sorted(
            documents,
            key=lambda doc: doc.metadata.get("final_score", 0.0),
            reverse=True
        )[:top_k]
    
    # 후처리: 조항제목 매칭 우선순위 재정렬
    documents = reorder_by_jo_title_priority(
        documents=documents,
        query=query,
        matching_keywords=matching_keywords,
    )
    
    # 메타데이터 필터링 (간단한 구현)
    if metadata_filter:
        filtered = []
        for doc in documents:
            match = True
            for key, value in metadata_filter.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
        documents = filtered
    
    logger.info(f"검색 결과: {len(documents)}개 문서")
    return documents

