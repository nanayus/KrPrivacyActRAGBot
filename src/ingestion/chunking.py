"""청킹 모듈

4.2 전처리 및 청킹 전략의 청킹 부분을 구현합니다.
구조 인식 청킹을 수행합니다.
"""
import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.preprocessing import split_into_logical_sections

logger = logging.getLogger(__name__)


def chunk_sections(sections: List[tuple], base_metadata: dict) -> List[Document]:
    """섹션 리스트를 청크로 분할합니다.
    
    개선된 청킹 전략:
    1. 조문 단위로 먼저 분할 (이미 sections에서 조문 단위로 분할됨)
    2. 각 조문을 하나의 청크로 유지 (의미 보존)
    3. 법률 문서(law)가 아닌 경우에만 1500자 초과시 추가 분할
    4. 법률 문서(law)는 길이와 무관하게 분할하지 않음
    
    Args:
        sections: (섹션 텍스트, 섹션 메타데이터) 튜플 리스트
        base_metadata: 원본 문서의 메타데이터
        
    Returns:
        Document 청크 리스트
    """
    # 문서 타입 확인
    document_type = base_metadata.get("document_type", "")
    is_law = document_type == "law"
    
    # 큰 청크를 위한 splitter (1500자 초과시에만 사용, law는 제외)
    large_chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # 조문이 1500자 초과시에만 분할
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],  # 한국어/일본어/영어 구분자
    )
    
    all_chunks = []
    
    for section_text, section_metadata in sections:
        # 기본 메타데이터와 섹션 메타데이터 병합
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update(section_metadata)
        
        # 법률 문서(law)는 길이와 무관하게 분할하지 않음
        if is_law:
            # 조문 전체를 하나의 청크로 유지 (의미 보존)
            chunk = Document(
                page_content=section_text,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk)
        else:
            # 조문 단위로 분할된 섹션은 하나의 청크로 유지
            # 단, 1500자 초과시에만 추가 분할
            if len(section_text) <= 1500:
                # 조문 전체를 하나의 청크로 유지 (의미 보존)
                chunk = Document(
                    page_content=section_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk)
            else:
                # 1500자 초과시에만 추가 분할
                sub_chunks = large_chunk_splitter.create_documents(
                    [section_text],
                    metadatas=[chunk_metadata]
                )
                all_chunks.extend(sub_chunks)
                logger.info(f"[청킹] 큰 조문 분할: {len(section_text)}자 -> {len(sub_chunks)}개 청크")
    
    return all_chunks


def chunk_documents(documents: List[Document]) -> List[Document]:
    """문서 리스트를 구조 인식 청킹합니다.
    
    4.2 전처리 및 청킹 전략에 따라:
    1. 논리적 섹션으로 분할 (제N조(제N조의 목적) 등)
    2. 큰 섹션은 RecursiveCharacterTextSplitter로 추가 분할
    3. 메타데이터가 없는 청크는 이전 청크의 "조", "조항제목"을 상속
    
    Args:
        documents: 전처리된 Document 리스트
        
    Returns:
        청크된 Document 리스트
    """
    all_chunks = []
    
    for doc in documents:
        # 논리적 섹션으로 분할
        sections = split_into_logical_sections(doc.page_content)
        
        # 각 섹션을 청크로 변환
        chunks = chunk_sections(sections, doc.metadata)
        
        # 청크에 섹션 정보 추가 (선택적)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        all_chunks.extend(chunks)
    
    # 메타데이터 전파: "조", "조항제목", "장", "장제목", "절", "절제목"이 없는 청크는 이전 청크의 메타데이터 상속
    last_jo = None
    last_jo_title = None
    last_jeol = None
    last_jeol_title = None
    last_jang = None
    last_jang_title = None
    
    for chunk in all_chunks:
        # 현재 청크에 "조" 또는 "조항제목"이 있는지 확인
        has_jo = "조" in chunk.metadata and chunk.metadata["조"] is not None
        has_jo_title = "조항제목" in chunk.metadata and chunk.metadata["조항제목"] is not None
        
        if has_jo:
            # 현재 청크에 조 정보가 있으면 업데이트
            last_jo = chunk.metadata["조"]
            if has_jo_title:
                last_jo_title = chunk.metadata["조항제목"]
        elif last_jo is not None:
            # 현재 청크에 조 정보가 없고 이전 청크에 있으면 상속
            chunk.metadata["조"] = last_jo
            if last_jo_title is not None:
                chunk.metadata["조항제목"] = last_jo_title
        
        # "절", "절제목"도 동일하게 처리
        has_jeol = "절" in chunk.metadata and chunk.metadata["절"] is not None
        has_jeol_title = "절제목" in chunk.metadata and chunk.metadata["절제목"] is not None
        
        if has_jeol:
            last_jeol = chunk.metadata["절"]
            if has_jeol_title:
                last_jeol_title = chunk.metadata["절제목"]
        elif last_jeol is not None:
            chunk.metadata["절"] = last_jeol
            if last_jeol_title is not None:
                chunk.metadata["절제목"] = last_jeol_title
        
        # "장", "장제목"도 동일하게 처리
        has_jang = "장" in chunk.metadata and chunk.metadata["장"] is not None
        has_jang_title = "장제목" in chunk.metadata and chunk.metadata["장제목"] is not None
        
        if has_jang:
            last_jang = chunk.metadata["장"]
            if has_jang_title:
                last_jang_title = chunk.metadata["장제목"]
        elif last_jang is not None:
            chunk.metadata["장"] = last_jang
            if last_jang_title is not None:
                chunk.metadata["장제목"] = last_jang_title
    
    # 파일 제목과 일치하는 청크 제거
    filtered_chunks = filter_title_chunks(all_chunks)
    
    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks (filtered: {len(all_chunks) - len(filtered_chunks)} title chunks removed)")
    return filtered_chunks


def filter_title_chunks(chunks: List[Document]) -> List[Document]:
    """파일 제목과 일치하는 청크를 제거합니다.
    
    청크 내용이 주로 파일 제목만 포함하거나, 파일 제목과 거의 일치하는 경우 제거합니다.
    
    Args:
        chunks: 청크 리스트
        
    Returns:
        필터링된 청크 리스트
    """
    # 주요 문서 제목 목록
    title_keywords = [
        "개인정보 보호법",
        "개인정보 보호법 시행령",
        "개인정보의 안전성 확보조치 기준",
        "개인정보 질의응답 모음집",
    ]
    
    filtered = []
    
    for chunk in chunks:
        # 청크 내용
        content = chunk.page_content.strip()
        
        # 문서 제목 추출
        document_title = chunk.metadata.get("title", "")
        if not document_title:
            document_title = chunk.metadata.get("source", "")
        
        # 청크가 너무 짧으면 (50자 미만) 제목일 가능성이 높음
        if len(content) < 50:
            # 주요 제목 키워드와 일치하는지 확인
            is_title = False
            for keyword in title_keywords:
                if keyword in content and len(content) <= len(keyword) + 20:  # 제목 + 약간의 여유
                    is_title = True
                    break
            
            # 문서 제목과 일치하는지 확인
            if not is_title and document_title:
                # 제목의 주요 부분 추출 (특수문자 제거)
                title_clean = re.sub(r'[^\w\s]', '', document_title)
                content_clean = re.sub(r'[^\w\s]', '', content)
                if title_clean in content_clean or content_clean in title_clean:
                    is_title = True
            
            if is_title:
                logger.debug(f"Removing title chunk: {content[:50]}")
                continue
        
        # 청크 내용이 주로 제목만 포함하는지 확인
        # 제목이 청크 내용의 80% 이상을 차지하면 제거
        if document_title:
            title_clean = re.sub(r'[^\w\s]', '', document_title)
            content_clean = re.sub(r'[^\w\s]', '', content)
            if len(title_clean) > 0 and len(content_clean) > 0:
                # 제목이 내용의 대부분을 차지하는지 확인
                if title_clean in content_clean:
                    title_ratio = len(title_clean) / len(content_clean)
                    if title_ratio > 0.8:
                        logger.debug(f"Removing chunk with high title ratio: {content[:50]}")
                        continue
        
        filtered.append(chunk)
    
    return filtered


