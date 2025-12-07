"""전처리 모듈

4.2 전처리 및 청킹 전략의 전처리 부분을 구현합니다.
텍스트를 정리하고 법률 구조를 인식합니다.
"""
import logging
import re
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_text(text: str, document_title: str = "") -> str:
    """텍스트를 정리합니다.
    
    - 페이지 헤더/푸터 제거 (법제처, 국가법령정보센터 등)
    - 파일 제목과 일치하는 헤더/푸터 제거
    - 연속된 공백/줄바꿈 정리
    - 특수 문자 정리
    
    Args:
        text: 원본 텍스트
        document_title: 문서 제목 (헤더/푸터 제거용)
        
    Returns:
        정리된 텍스트
    """
    # 페이지 헤더/푸터 패턴 제거
    # "법제처 N 국가법령정보센터" 패턴 (N은 숫자, 줄바꿈 포함)
    text = re.sub(r'법제처\s+\d+\s+국가법령정보센터\s*\n?', '', text, flags=re.MULTILINE)
    
    # "개인정보 보호법" 단독 라인 제거 (헤더/푸터로 사용되는 경우)
    # 단, 문장 중간에 있는 것은 유지 (줄의 시작과 끝에만 있는 경우)
    text = re.sub(r'^개인정보\s+보호법\s*$', '', text, flags=re.MULTILINE)
    
    # "법제처 N 국가법령정보센터\n개인정보 보호법" 패턴 (연속으로 나타나는 경우)
    text = re.sub(r'법제처\s+\d+\s+국가법령정보센터\s*\n\s*개인정보\s+보호법\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'법제처\s+\d+\s+국가법령정보센터\s*\n\s*개인정보\s+보호법\s+시행령\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'법제처\s+\d+\s+국가법령정보센터\s*\n\s*개인정보의\s+안전성\s+확보조치\s+기준\s*\n?', '', text, flags=re.MULTILINE)
    
    # 주요 문서 제목 패턴 제거 (공통)
    # "개인정보 보호법" 단독 라인 (줄바꿈 포함)
    text = re.sub(r'^개인정보\s+보호법\s*\n?', '', text, flags=re.MULTILINE)
    # "개인정보 보호법 시행령" 단독 라인
    text = re.sub(r'^개인정보\s+보호법\s+시행령\s*\n?', '', text, flags=re.MULTILINE)
    # "개인정보의 안전성 확보조치 기준" 단독 라인
    text = re.sub(r'^개인정보의\s+안전성\s+확보조치\s+기준\s*\n?', '', text, flags=re.MULTILINE)
    # "개인정보 질의응답 모음집" 단독 라인
    text = re.sub(r'^개인정보\s+질의응답\s+모음집\s*\n?', '', text, flags=re.MULTILINE)
    
    # 문서 제목이 제공된 경우, 해당 제목과 일치하는 패턴도 제거
    if document_title:
        # 제목에서 특수문자 이스케이프
        escaped_title = re.escape(document_title)
        # 제목만 있는 라인 제거 (줄바꿈 포함)
        text = re.sub(rf'^{escaped_title}\s*\n?', '', text, flags=re.MULTILINE)
        # 제목 + 줄바꿈 패턴 제거
        text = re.sub(rf'{escaped_title}\s*\n', '', text, flags=re.MULTILINE)
    
    # 연속된 공백을 하나로
    text = re.sub(r' +', ' ', text)
    # 연속된 줄바꿈을 최대 2개로
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def clean_guides_text(text: str, document_title: str = "") -> str:
    """해설서/가이드라인 문서의 텍스트를 정리합니다.
    
    - 페이지 헤더/푸터 제거
      * 로마숫자 + 제목 + 페이지 번호 (예: "Ⅲ. 가명정보 35")
      * 숫자 + 제목 (예: "24 개인정보 질의응답 모음집")
      * 질문 끝 + 페이지 번호 (예: "~요? N")
    - 파일 제목과 일치하는 헤더/푸터 제거
    - 연속된 공백/줄바꿈 정리
    
    Args:
        text: 원본 텍스트
        document_title: 문서 제목 (헤더/푸터 제거용)
        
    Returns:
        정리된 텍스트
    """
    # 로마숫자 패턴 (Ⅰ, Ⅱ, Ⅲ, Ⅳ, Ⅴ, Ⅵ, Ⅶ, Ⅷ, Ⅸ, Ⅹ 등) + 제목 + 페이지 번호
    # 예: "Ⅲ. 가명정보 35"
    text = re.sub(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*[^\n]+\s+\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 숫자 + 제목 패턴 (줄의 시작에 숫자, 그 다음 공백 없이 제목, 페이지 번호)
    # 예: "24 개인정보 질의응답 모음집"
    text = re.sub(r'^\d+\s+[^\n]+\s*$', '', text, flags=re.MULTILINE)
    
    # 질문 끝 + 페이지 번호 패턴
    # 예: "~요? N", "~까요? 123" 등
    text = re.sub(r'[~～]\s*[^\n]*[?？]\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 페이지 번호만 있는 라인 제거 (숫자만 있는 라인)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 주요 문서 제목 패턴 제거 (공통)
    # "개인정보 보호법" 단독 라인
    text = re.sub(r'^개인정보\s+보호법\s*\n?', '', text, flags=re.MULTILINE)
    # "개인정보 보호법 시행령" 단독 라인
    text = re.sub(r'^개인정보\s+보호법\s+시행령\s*\n?', '', text, flags=re.MULTILINE)
    # "개인정보의 안전성 확보조치 기준" 단독 라인
    text = re.sub(r'^개인정보의\s+안전성\s+확보조치\s+기준\s*\n?', '', text, flags=re.MULTILINE)
    # "개인정보 질의응답 모음집" 단독 라인
    text = re.sub(r'^개인정보\s+질의응답\s+모음집\s*\n?', '', text, flags=re.MULTILINE)
    
    # 문서 제목이 제공된 경우, 해당 제목과 일치하는 패턴도 제거
    if document_title:
        # 제목에서 특수문자 이스케이프
        escaped_title = re.escape(document_title)
        # 제목만 있는 라인 제거 (줄바꿈 포함)
        text = re.sub(rf'^{escaped_title}\s*\n?', '', text, flags=re.MULTILINE)
        # 제목 + 줄바꿈 패턴 제거
        text = re.sub(rf'{escaped_title}\s*\n', '', text, flags=re.MULTILINE)
    
    # 연속된 공백을 하나로
    text = re.sub(r' +', ' ', text)
    # 연속된 줄바꿈을 최대 2개로
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def extract_structure_markers(text: str) -> List[tuple]:
    """법률 문서의 구조 마커를 추출합니다.
    
    제N조(, 제N조의M(, 제N장, 제N절 등의 패턴을 찾습니다.
    제N조는 "제N조(" 또는 "제N조의M(" 패턴을 기준으로 분할합니다.
    
    Args:
        text: 텍스트
        
    Returns:
        (마커 타입, 번호, 위치) 튜플 리스트
    """
    markers = []
    
    # 제N조의M( 패턴 (예: "제31조의2(") - 먼저 검색 (더 구체적인 패턴)
    article_with_ui_pattern = r'제\s*(\d+)\s*조의\s*(\d+)\s*\('
    for match in re.finditer(article_with_ui_pattern, text):
        # 조 번호와 "의" 뒤의 번호를 조합하여 고유 식별자 생성
        article_num = int(match.group(1))
        ui_num = int(match.group(2))
        # "조의" 패턴은 별도로 표시
        markers.append(("조의", article_num * 1000 + ui_num, match.start()))
    
    # 제N조( 패턴 (괄호가 있는 경우만 - 예: "제2조(정의)")
    article_pattern = r'제\s*(\d+)\s*조\s*\('
    for match in re.finditer(article_pattern, text):
        # "제N조의M(" 패턴과 겹치지 않는 경우만 추가
        pos = match.start()
        # 이미 "제N조의M(" 패턴으로 찾은 것과 겹치는지 확인
        is_overlap = any(
            m[0] == "조의" and 
            m[1] // 1000 == int(match.group(1)) and 
            abs(m[2] - pos) < 10
            for m in markers
        )
        if not is_overlap:
            markers.append(("조", int(match.group(1)), pos))
    
    # 제N장 패턴
    chapter_pattern = r'제\s*(\d+)\s*장'
    for match in re.finditer(chapter_pattern, text):
        markers.append(("장", int(match.group(1)), match.start()))
    
    # 제N절 패턴
    section_pattern = r'제\s*(\d+)\s*절'
    for match in re.finditer(section_pattern, text):
        markers.append(("절", int(match.group(1)), match.start()))
    
    # 제N항 패턴 (조 내부)
    paragraph_pattern = r'제\s*(\d+)\s*항'
    for match in re.finditer(paragraph_pattern, text):
        markers.append(("항", int(match.group(1)), match.start()))
    
    # 위치 순으로 정렬
    markers.sort(key=lambda x: x[2])
    
    return markers


def extract_section_metadata(section_text: str) -> dict:
    """섹션 텍스트에서 조/절/장 정보를 추출합니다.
    
    Args:
        section_text: 섹션 텍스트
        
    Returns:
        메타데이터 딕셔너리 (조, 조항제목, 절, 절제목, 장, 장제목)
    """
    metadata = {}
    
    # 제N장 패턴 찾기 (예: "제1장 총칙", "제2장 개인정보 보호정책의 수립 등")
    # 패턴: "제N장" 다음에 공백이나 줄바꿈 후 제목이 올 수 있음
    # 예: "제2장 개인정보 보호정책의 수립 등" 또는 "제3장 개인정보의 처리\n 제1절..."
    chapter_pattern = r'제\s*(\d+)\s*장\s+([^\n]+?)(?=\n\s*제\s*\d+[절조]|\n\n|$)'
    chapter_match = re.search(chapter_pattern, section_text)
    if chapter_match:
        metadata["장"] = int(chapter_match.group(1))
        # 제목 추출
        chapter_title = chapter_match.group(2).strip()
        # 불필요한 공백 정리
        chapter_title = re.sub(r'\s+', ' ', chapter_title)
        metadata["장제목"] = chapter_title
    
    # 제N절 패턴 찾기 (예: "제1절 개인정보의 수집, 이용, 제공 등")
    # 패턴: "제N절" 다음에 공백이나 줄바꿈 후 제목이 올 수 있음
    # 예: "제1절 개인정보의 수집, 이용, 제공 등"
    section_pattern = r'제\s*(\d+)\s*절\s+([^\n]+?)(?=\n\s*제\s*\d+[장조]|\n\n|$)'
    section_match = re.search(section_pattern, section_text)
    if section_match:
        metadata["절"] = int(section_match.group(1))
        # 제목 추출
        section_title = section_match.group(2).strip()
        # 불필요한 공백 정리
        section_title = re.sub(r'\s+', ' ', section_title)
        metadata["절제목"] = section_title
    
    # 제N조의M(제목) 패턴 찾기
    article_with_ui_pattern = r'제\s*(\d+)\s*조의\s*(\d+)\s*\(([^)]+)\)'
    article_ui_match = re.search(article_with_ui_pattern, section_text)
    if article_ui_match:
        metadata["조"] = f"{article_ui_match.group(1)}조의{article_ui_match.group(2)}"
        metadata["조항제목"] = article_ui_match.group(3).strip()
    else:
        # 제N조(제목) 패턴 찾기
        article_pattern = r'제\s*(\d+)\s*조\s*\(([^)]+)\)'
        article_match = re.search(article_pattern, section_text)
        if article_match:
            metadata["조"] = int(article_match.group(1))
            metadata["조항제목"] = article_match.group(2).strip()
    
    return metadata


def split_into_logical_sections(text: str) -> List[tuple]:
    """텍스트를 논리적 섹션으로 분할합니다.
    
    4.2 전처리 및 청킹 전략에 따라 법률 구조를 존중하여 분할합니다.
    "제N조(" 또는 "제N조의M(" 패턴을 기준으로 분할합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        (섹션 텍스트, 메타데이터) 튜플 리스트
    """
    text = clean_text(text, "")
    
    # "제N조(" 또는 "제N조의M(" 패턴의 시작 위치를 모두 찾기
    # 패턴: 제N조(설명) 또는 제N조의M(설명)
    # 다음 "제N조(" 또는 "제N조의M(" 패턴 전까지를 하나의 섹션으로
    
    # 모든 "제N조(" 또는 "제N조의M(" 패턴의 시작 위치 찾기
    split_positions = []
    
    # 제N조의M( 패턴 찾기 (더 구체적인 패턴 먼저)
    pattern1 = r'제\s*\d+\s*조의\s*\d+\s*\('
    for match in re.finditer(pattern1, text):
        split_positions.append(match.start())
    
    # 제N조( 패턴 찾기
    pattern2 = r'제\s*\d+\s*조\s*\('
    for match in re.finditer(pattern2, text):
        pos = match.start()
        # 이미 "제N조의M(" 패턴으로 찾은 위치와 겹치지 않는지 확인
        # (예: "제31조의2(" 안에 "제31조("가 포함될 수 있음)
        is_overlap = any(
            abs(pos - sp) < 10 for sp in split_positions
        )
        if not is_overlap:
            split_positions.append(pos)
    
    # 위치 순으로 정렬
    split_positions.sort()
    
    if not split_positions:
        # 패턴이 없으면 전체를 하나의 섹션으로
        return [(text, {})]
    
    # 각 위치를 기준으로 섹션 분할
    sections = []
    
    # 첫 번째 섹션 이전의 내용
    if split_positions[0] > 0:
        prefix = text[:split_positions[0]].strip()
        if prefix:
            sections.append((prefix, {}))
    
    # 각 섹션 추출 (현재 위치부터 다음 위치 전까지)
    for i in range(len(split_positions)):
        start_pos = split_positions[i]
        end_pos = split_positions[i + 1] if i + 1 < len(split_positions) else len(text)
        
        section = text[start_pos:end_pos].strip()
        if section:
            # 섹션에서 메타데이터 추출
            section_metadata = extract_section_metadata(section)
            sections.append((section, section_metadata))
    
    return sections


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """문서 리스트를 전처리합니다.
    
    문서 타입에 따라 적절한 전처리 함수를 적용합니다:
    - guideline: clean_guides_text() 사용
    - 기타: clean_text() 사용
    
    문서 제목을 메타데이터에서 추출하여 헤더/푸터 제거에 활용합니다.
    
    Args:
        documents: 원본 Document 리스트
        
    Returns:
        전처리된 Document 리스트
    """
    processed = []
    
    for doc in documents:
        # 문서 타입 확인
        doc_type = doc.metadata.get("document_type", "")
        
        # 문서 제목 추출 (title 또는 source에서)
        document_title = doc.metadata.get("title", "")
        if not document_title:
            document_title = doc.metadata.get("source", "")
        
        # guideline 타입이면 guides 전처리 함수 사용
        if doc_type == "guideline":
            cleaned_text = clean_guides_text(doc.page_content, document_title)
        else:
            # 법률 문서 등은 기존 전처리 함수 사용
            cleaned_text = clean_text(doc.page_content, document_title)
        
        # 새로운 Document 생성 (메타데이터 유지)
        processed_doc = Document(
            page_content=cleaned_text,
            metadata=doc.metadata.copy()
        )
        
        processed.append(processed_doc)
    
    logger.info(f"Preprocessed {len(processed)} documents")
    return processed

