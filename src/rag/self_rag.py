"""Self-RAG 체인 모듈

5.3 RAG 체인 및 Self-RAG 체인을 구현합니다.
Self-check 및 수정 기능을 제공합니다.
JSON 기반 2차 검색 구조를 사용합니다.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, LLM_MODEL
from src.prompts import (
    FINAL_ANSWER_TEMPLATE,
    SELF_CHECK_TEMPLATE,
    SYSTEM_PROMPT_FINAL_ANSWER,
    SYSTEM_PROMPT_SELF_RAG,
)
from src.rag.chains import create_rag_chain, query_rag_chain

logger = logging.getLogger(__name__)


def create_self_check_chain():
    """Self-check 체인을 생성합니다.
    
    5.2.2 Self-RAG 시스템 프롬프트를 사용합니다.
    
    Returns:
        Self-check를 수행하는 LLM 체인
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_SELF_RAG),
        ("human", SELF_CHECK_TEMPLATE),
    ])
    
    chain = prompt | llm
    
    return chain


def format_context_for_self_check(sources: List[Document]) -> str:
    """검색된 문서를 Self-check용 컨텍스트로 포맷팅합니다.
    
    Args:
        sources: 검색된 Document 리스트
        
    Returns:
        포맷팅된 컨텍스트 문자열
    """
    context_parts = []
    
    for i, doc in enumerate(sources, 1):
        source_info = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("document_type", "Unknown")
        
        context_parts.append(
            f"[{i}] 출처: {source_info} ({doc_type})\n"
            f"내용: {doc.page_content[:500]}..."  # 처음 500자만
        )
    
    return "\n\n".join(context_parts)


def extract_referenced_documents(answer: str, sources: List[Document]) -> List[Document]:
    """답변 텍스트에서 실제로 인용된 문서만 필터링합니다.
    
    답변 텍스트에서 조 번호(예: 제15조, 제17조 제1항)나 문서명을 추출하고,
    해당 메타데이터와 일치하는 문서만 반환합니다.
    
    Args:
        answer: 생성된 답변 텍스트
        sources: 검색된 모든 Document 리스트
        
    Returns:
        실제로 답변에 인용된 Document 리스트
    """
    if not answer or not sources:
        return sources
    
    # 답변에서 조 번호 추출 (예: 제15조, 제17조 제1항, 제15조제1항 등)
    jo_pattern = r'제(\d+)조'
    jo_numbers = set()
    for match in re.finditer(jo_pattern, answer):
        jo_num = match.group(1)
        jo_numbers.add(jo_num)
        jo_numbers.add(int(jo_num))  # 숫자로도 저장
    
    # 답변에서 문서명 추출 (개인정보보호법, 시행령 등)
    document_keywords = []
    if '개인정보보호법' in answer or '개인정보 보호법' in answer:
        document_keywords.append('개인정보보호법')
    if '시행령' in answer:
        document_keywords.append('시행령')
    if '시행규칙' in answer:
        document_keywords.append('시행규칙')
    if '가이드라인' in answer:
        document_keywords.append('가이드라인')
    
    # 필터링된 문서 리스트
    filtered_docs = []
    
    for doc in sources:
        doc_jo = doc.metadata.get('조', None)
        doc_source = doc.metadata.get('source', '')
        doc_type = doc.metadata.get('document_type', '')
        
        # 조 번호로 매칭
        if doc_jo is not None:
            # doc_jo가 문자열인 경우 숫자로 변환 시도
            try:
                doc_jo_num = int(str(doc_jo).replace('조', '').strip())
                if doc_jo_num in jo_numbers or str(doc_jo_num) in jo_numbers:
                    filtered_docs.append(doc)
                    continue
            except (ValueError, AttributeError):
                pass
            
            # 문자열로 직접 비교
            if str(doc_jo) in answer:
                filtered_docs.append(doc)
                continue
        
        # 문서명으로 매칭
        for keyword in document_keywords:
            if keyword in doc_source or keyword in doc_type or keyword in str(doc.metadata):
                filtered_docs.append(doc)
                break
    
    # 매칭된 문서가 없으면 상위 3개만 반환 (최소한의 참고 문서 표시)
    if not filtered_docs:
        logger.info("답변에서 명시적으로 인용된 문서를 찾을 수 없어 상위 문서를 반환합니다.")
        return sources[:3] if len(sources) > 3 else sources
    
    # 중복 제거 (같은 문서가 여러 번 매칭될 수 있음)
    seen = set()
    unique_docs = []
    for doc in filtered_docs:
        # 문서를 고유하게 식별하기 위해 조 번호와 출처 조합 사용
        doc_id = (doc.metadata.get('조'), doc.metadata.get('source'))
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    
    logger.info(f"답변에서 인용된 문서: {len(unique_docs)}개 (전체 검색 결과: {len(sources)}개)")
    return unique_docs


def parse_self_check_json(check_text: str) -> Optional[Dict]:
    """Self-check 결과에서 JSON을 파싱합니다.
    
    Args:
        check_text: Self-check LLM 응답 텍스트
        
    Returns:
        파싱된 JSON 딕셔너리 또는 None (파싱 실패 시)
    """
    try:
        # JSON 코드 블록 제거 (```json ... ``` 형식)
        if "```json" in check_text:
            start = check_text.find("```json") + 7
            end = check_text.find("```", start)
            if end != -1:
                check_text = check_text[start:end].strip()
        elif "```" in check_text:
            start = check_text.find("```") + 3
            end = check_text.find("```", start)
            if end != -1:
                check_text = check_text[start:end].strip()
        
        # JSON 파싱
        decision = json.loads(check_text)
        
        # 필수 필드 검증
        if not isinstance(decision, dict):
            return None
        
        # 기본값 설정
        decision.setdefault("need_more_context", False)
        decision.setdefault("followup_query", "")
        decision.setdefault("final_answer", "")
        decision.setdefault("reason", "")
        
        return decision
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning(f"Self-check JSON 파싱 실패: {e}")
        logger.debug(f"파싱 실패한 텍스트: {check_text[:500]}")
        return None


def create_final_answer_chain():
    """최종 답변 재작성을 위한 체인을 생성합니다.
    
    Returns:
        최종 답변 재작성 LLM 체인
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_FINAL_ANSWER),
        ("human", FINAL_ANSWER_TEMPLATE),
    ])
    
    chain = prompt | llm
    return chain


def self_rag_query(question: str, progress_callback=None) -> Dict:
    """Self-RAG 파이프라인으로 질문을 처리합니다.
    
    5.3 RAG 체인 및 Self-RAG 체인에 따라:
    1. 기본 RAG 체인 실행
    2. Self-check 체인으로 JSON 형식 검증
    3. 정보 부족 시 추가 검색 수행
    4. 최종 답변 재작성
    
    Args:
        question: 사용자 질문
        progress_callback: 진행 상황 콜백 함수 (선택적)
        
    Returns:
        {
            "answer": str,
            "sources": List[Document],
            "self_check_result": str 또는 Dict,
            "corrected": bool
        }
    """
    logger.info(f"Self-RAG 파이프라인 시작: {question}")
    print(f"[진행] Self-RAG 파이프라인 시작: {question}")
    
    # 1. 기본 RAG 체인 실행
    try:
        if progress_callback:
            progress_callback("검색기 로드 중...")
        print("[진행] 검색기 로드 중...")
        
        if progress_callback:
            progress_callback("관련 문서 검색 중...")
        print("[진행] 관련 문서 검색 중...")
        
        rag_result = query_rag_chain(question)
        initial_answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])
        
        if not initial_answer:
            logger.warning("RAG 체인에서 빈 답변이 반환되었습니다.")
            initial_answer = "답변을 생성할 수 없습니다. 인덱스가 구축되었는지 확인하세요."
        
        logger.info("초기 답변 생성 완료, Self-check 시작...")
        print(f"[진행] 초기 답변 생성 완료 (소스 {len(sources)}개)")
        
        if progress_callback:
            progress_callback("답변 생성 완료, 검증 중...")
    except Exception as e:
        logger.error(f"RAG 체인 실행 중 오류: {e}", exc_info=True)
        print(f"[오류] RAG 체인 실행 중 오류: {e}")
        raise
    
    # 2. Self-check (JSON 형식)
    first_context = format_context_for_self_check(sources)
    decision: Optional[Dict] = None
    check_text_raw = ""
    corrected_by_self_check = False
    
    try:
        print("[진행] Self-check 체인 생성 중...")
        self_check_chain = create_self_check_chain()
        
        print("[진행] Self-check 실행 중...")
        self_check_result = self_check_chain.invoke({
            "question": question,
            "initial_answer": initial_answer,
            "context": first_context,
        })
        print("[진행] Self-check 완료")
        
        # Gemini 응답 형식에 맞게 처리
        if hasattr(self_check_result, 'content'):
            check_text_raw = self_check_result.content
        elif isinstance(self_check_result, str):
            check_text_raw = self_check_result
        elif hasattr(self_check_result, 'text'):
            check_text_raw = self_check_result.text
        else:
            check_text_raw = str(self_check_result)
        
        # JSON 파싱
        decision = parse_self_check_json(check_text_raw)
        
    except Exception as e:
        logger.warning(f"Self-check 실행 중 오류 발생: {e}", exc_info=True)
        # Self-check 실패 시 원래 답변 반환
        return {
            "answer": initial_answer,
            "sources": extract_referenced_documents(initial_answer, sources),
            "self_check_result": f"Self-check 실행 실패: {str(e)}",
            "corrected": False,
        }
    
    # JSON 파싱 실패 시 fallback
    if decision is None:
        logger.warning("Self-check JSON 파싱 실패, 초기 답변 사용")
        return {
            "answer": initial_answer,
            "sources": extract_referenced_documents(initial_answer, sources),
            "self_check_result": check_text_raw,
            "corrected": False,
        }
    
    # 3. Self-check 결과 처리
    need_more_context = decision.get("need_more_context", False)
    followup_query = decision.get("followup_query", "").strip()
    final_answer_from_check = decision.get("final_answer", "").strip()
    
    # Self-check에서 답변 수정 여부 확인
    if final_answer_from_check and final_answer_from_check != initial_answer:
        corrected_by_self_check = True
        logger.info("Self-check에서 답변이 수정되었습니다.")
    
    # need_more_context가 False이면 여기서 종료
    if not need_more_context:
        logger.info("Self-check: 추가 검색 불필요")
        final_answer = final_answer_from_check if final_answer_from_check else initial_answer
        used_sources = extract_referenced_documents(final_answer, sources)
        
        return {
            "answer": final_answer,
            "sources": used_sources,
            "self_check_result": decision,
            "corrected": corrected_by_self_check,
        }
    
    # followup_query가 비어있으면 방어적으로 처리
    if not followup_query:
        logger.warning("Self-check에서 추가 검색이 필요하다고 했지만 followup_query가 비어있음")
        final_answer = final_answer_from_check if final_answer_from_check else initial_answer
        used_sources = extract_referenced_documents(final_answer, sources)
        
        return {
            "answer": final_answer,
            "sources": used_sources,
            "self_check_result": decision,
            "corrected": corrected_by_self_check,
        }
    
    # 4. 추가 검색 수행 (need_more_context가 True이고 followup_query가 유효한 경우)
    logger.info(f"Self-check: 추가 검색 필요 - followup_query: '{followup_query}'")
    print(f"[진행] 추가 검색 수행: '{followup_query}'")
    
    if progress_callback:
        progress_callback("추가 검색 수행 중...")
    
    try:
        second_rag_result = query_rag_chain(followup_query)
        second_answer = second_rag_result.get("answer", "")
        second_sources = second_rag_result.get("sources", [])
        
        logger.info(f"추가 검색 완료: {len(second_sources)}개 문서 검색됨")
        print(f"[진행] 추가 검색 완료: {len(second_sources)}개 문서 검색됨")
        
    except Exception as e:
        logger.error(f"추가 검색 중 오류 발생: {e}", exc_info=True)
        print(f"[오류] 추가 검색 실패: {e}")
        # 추가 검색 실패 시 Self-check 결과만 사용
        final_answer = final_answer_from_check if final_answer_from_check else initial_answer
        used_sources = extract_referenced_documents(final_answer, sources)
        
        return {
            "answer": final_answer,
            "sources": used_sources,
            "self_check_result": decision,
            "corrected": corrected_by_self_check,
        }
    
    # 5. 최종 답변 재작성
    logger.info("최종 답변 재작성 시작...")
    print("[진행] 최종 답변 재작성 중...")
    
    if progress_callback:
        progress_callback("최종 답변 재작성 중...")
    
    try:
        second_context = format_context_for_self_check(second_sources)
        final_chain = create_final_answer_chain()
        
        final_result = final_chain.invoke({
            "question": question,
            "initial_answer": initial_answer,
            "self_check_answer": final_answer_from_check if final_answer_from_check else initial_answer,
            "first_context": first_context,
            "second_context": second_context,
        })
        
        # 최종 답변 추출
        if hasattr(final_result, 'content'):
            corrected_answer = final_result.content
        elif isinstance(final_result, str):
            corrected_answer = final_result
        elif hasattr(final_result, 'text'):
            corrected_answer = final_result.text
        else:
            corrected_answer = str(final_result)
        
        corrected_answer = corrected_answer.strip()
        
        # 최종 수정 여부 확인
        corrected_by_final = corrected_answer != initial_answer
        corrected = corrected_by_self_check or corrected_by_final
        
        logger.info(f"최종 답변 재작성 완료 (수정됨: {corrected})")
        print(f"[진행] 최종 답변 재작성 완료")
        
    except Exception as e:
        logger.error(f"최종 답변 재작성 중 오류 발생: {e}", exc_info=True)
        print(f"[오류] 최종 답변 재작성 실패: {e}")
        # 재작성 실패 시 Self-check 결과 사용
        corrected_answer = final_answer_from_check if final_answer_from_check else initial_answer
        corrected = corrected_by_self_check
    
    # 6. 최종 소스 문서 필터링 (1차 + 2차 검색 결과 통합)
    all_sources = sources + second_sources
    used_sources = extract_referenced_documents(corrected_answer, all_sources)
    
    logger.info(f"Self-RAG 파이프라인 완료 (최종 소스: {len(used_sources)}개)")
    print(f"[진행] Self-RAG 파이프라인 완료")
    
    return {
        "answer": corrected_answer,
        "sources": used_sources,
        "self_check_result": decision,
        "corrected": corrected,
    }

