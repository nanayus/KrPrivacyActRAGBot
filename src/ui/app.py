"""Streamlit 웹 UI

간단한 웹 인터페이스를 제공합니다.
"""
import json
import re
from typing import List, Tuple

import streamlit as st
from langchain_core.documents import Document

from src.rag.self_rag import self_rag_query


def sort_sources_by_importance(answer: str, sources: List[Document]) -> List[Document]:
    """답변에서 인용된 순서와 중요도를 기준으로 소스를 정렬합니다.
    
    Args:
        answer: 최종 답변 텍스트
        sources: 소스 문서 리스트
        
    Returns:
        중요도 순으로 정렬된 소스 문서 리스트
    """
    if not answer or not sources:
        return sources
    
    # 답변에서 조 번호 추출 (인용된 순서)
    jo_pattern = r'제(\d+)조'
    jo_numbers_in_answer = []
    for match in re.finditer(jo_pattern, answer):
        jo_num = int(match.group(1))
        if jo_num not in jo_numbers_in_answer:
            jo_numbers_in_answer.append(jo_num)
    
    # 각 소스의 중요도 점수 계산
    source_scores: List[Tuple[Document, float]] = []
    
    for doc in sources:
        score = 0.0
        doc_jo = doc.metadata.get('조', None)
        
        # 1. 답변에서 인용된 조 번호와 일치하는 경우 높은 점수
        if doc_jo is not None:
            try:
                doc_jo_num = int(str(doc_jo).replace('조', '').strip())
                if doc_jo_num in jo_numbers_in_answer:
                    # 인용된 순서에 따라 점수 부여 (먼저 인용된 것이 높은 점수)
                    position = jo_numbers_in_answer.index(doc_jo_num)
                    score += 1000 - (position * 10)
            except (ValueError, AttributeError):
                pass
        
        # 2. final_score가 있으면 추가 점수
        final_score = doc.metadata.get('final_score', 0.0)
        if final_score:
            score += final_score * 100
        
        # 3. 조항제목이 답변에 포함된 경우 추가 점수
        jo_title = doc.metadata.get('조항제목', '')
        if jo_title and jo_title in answer:
            score += 50
        
        # 4. 문서 내용이 답변과 겹치는 키워드가 많은 경우 추가 점수
        doc_content = doc.page_content[:200]  # 처음 200자만 확인
        answer_words = set(answer.split())
        doc_words = set(doc_content.split())
        common_words = answer_words & doc_words
        if common_words:
            score += len(common_words) * 0.5
        
        source_scores.append((doc, score))
    
    # 점수 순으로 정렬 (내림차순)
    source_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in source_scores]


st.set_page_config(
    page_title="개인정보보호법 RAG 시스템",
    page_icon="📜",
    layout="wide",
)

st.title("📜 개인정보보호법 RAG 시스템")
st.markdown("한국 개인정보보호법 및 관련 고시·가이드라인 기반 질의응답 시스템")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("이 시스템은 개인정보보호법 관련 문서를 기반으로 답변을 제공합니다.")
    
    st.markdown("---")
    st.markdown("### 📚 사용 가능한 문서")
    st.markdown("- 개인정보보호법")
    st.markdown("- 시행령·시행규칙")
    st.markdown("- 가이드라인")
    st.markdown("- FAQ")
    
    st.markdown("---")
    st.markdown("### ℹ️ 안내")
    st.markdown("""
    - 모든 답변은 제공된 문서에만 기반합니다
    - 법조문 번호와 출처가 함께 표시됩니다
    - Self-RAG로 답변 품질을 자동 검증합니다
    """)

# 메인 영역
query = st.text_input(
    "질문을 입력하세요:",
    placeholder="예: 개인정보 보유기간은 얼마나 되나요?",
    key="query_input"
)

if st.button("🔍 검색", type="primary", use_container_width=True):
    if not query:
        st.warning("질문을 입력해주세요.")
    else:
        # 진행 상황 표시용 컨테이너
        status_container = st.empty()
        progress_container = st.empty()
        
        try:
            # 1단계: 검색기 로드
            with status_container.container():
                st.info("🔍 검색기 로드 중...")
            progress_container.progress(10)
            
            # 2단계: 문서 검색
            with status_container.container():
                st.info("📚 관련 문서 검색 중...")
            progress_container.progress(30)
            
            # 3단계: 답변 생성
            with status_container.container():
                st.info("💬 답변 생성 중...")
            progress_container.progress(50)
            
            result = self_rag_query(query)
            
            # 4단계: Self-check
            with status_container.container():
                st.info("✅ 답변 검증 중...")
            progress_container.progress(80)
            
            # 완료
            status_container.empty()
            progress_container.empty()
            
            # 답변 표시
            st.markdown("### 💬 답변")
            answer = result["answer"]
            st.markdown(answer)
            
            # Self-check 결과
            if result.get("corrected"):
                st.info("✅ Self-check에서 답변이 검증 및 수정되었습니다.")
            
            # 소스 문서 (최종 답변 기준으로 중요도 순 정렬)
            st.markdown("### 📄 참고 문서")
            sources = result.get("sources", [])
            
            if sources:
                # 답변에서 인용된 순서와 중요도를 기준으로 정렬
                sorted_sources = sort_sources_by_importance(answer, sources)
                
                for i, doc in enumerate(sorted_sources, 1):
                    # 문서 정보 추출
                    jo = doc.metadata.get('조', 'N/A')
                    jo_title = doc.metadata.get('조항제목', 'N/A')
                    source = doc.metadata.get('source', doc.metadata.get('source_file', 'Unknown'))
                    doc_type = doc.metadata.get('document_type', 'Unknown')
                    
                    # 문서 제목 생성
                    if jo != 'N/A':
                        doc_title = f"제{jo}조"
                        if jo_title != 'N/A':
                            doc_title += f" ({jo_title})"
                    else:
                        doc_title = source
                    
                    # 점수 정보 (있는 경우)
                    final_score = doc.metadata.get('final_score', None)
                    dense_score = doc.metadata.get('dense_score', None)
                    sparse_score = doc.metadata.get('sparse_score', None)
                    
                    # 중요도 표시
                    importance_badge = ""
                    if i <= 3:
                        importance_badge = " 🔥" if i == 1 else " ⭐" if i == 2 else " 📌"
                    
                    with st.expander(f"📄 {i}. {doc_title}{importance_badge}", expanded=(i <= 2)):
                        # 메타데이터 정보
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**조**: {jo}")
                            st.markdown(f"**조항제목**: {jo_title}")
                        with col2:
                            st.markdown(f"**문서 타입**: {doc_type}")
                            st.markdown(f"**출처**: {source}")
                        
                        # 점수 정보 (있는 경우)
                        if final_score is not None:
                            score_col1, score_col2, score_col3 = st.columns(3)
                            with score_col1:
                                st.metric("최종 점수", f"{final_score:.3f}")
                            if dense_score is not None:
                                with score_col2:
                                    st.metric("Dense 점수", f"{dense_score:.3f}")
                            if sparse_score is not None:
                                with score_col3:
                                    st.metric("Sparse 점수", f"{sparse_score:.3f}")
                        
                        st.markdown("**내용**:")
                        st.text(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
            else:
                st.warning("참고 문서를 찾을 수 없습니다.")
            
            # Self-check 상세 결과 (JSON 구조화)
            self_check_result = result.get("self_check_result", {})
            with st.expander("🔍 Self-check 상세 결과", expanded=False):
                if isinstance(self_check_result, dict):
                    # JSON 구조화된 결과 표시
                    st.markdown("#### 📊 Self-check 분석 결과")
                    
                    # need_more_context
                    need_more = self_check_result.get("need_more_context", False)
                    if need_more:
                        st.warning("⚠️ 추가 검색이 필요하다고 판단되었습니다.")
                    else:
                        st.success("✅ 현재 정보로 충분하다고 판단되었습니다.")
                    
                    # followup_query
                    followup = self_check_result.get("followup_query", "").strip()
                    if followup:
                        st.markdown("**추가 검색 질의문**:")
                        st.info(f"`{followup}`")
                    
                    # final_answer
                    final_answer_check = self_check_result.get("final_answer", "").strip()
                    if final_answer_check and final_answer_check != query:
                        st.markdown("**Self-check가 제안한 답변**:")
                        st.markdown(f'<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4;">{final_answer_check}</div>', unsafe_allow_html=True)
                    
                    # reason
                    reason = self_check_result.get("reason", "").strip()
                    if reason:
                        st.markdown("**판단 이유**:")
                        st.info(reason)
                    
                    # 원본 JSON (디버깅용)
                    with st.expander("🔧 원본 JSON (디버깅용)", expanded=False):
                        st.json(self_check_result)
                else:
                    # 문자열 형태의 결과 (fallback)
                    st.text(str(self_check_result))
                
        except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {str(e)}")
                with st.expander("🔍 상세 오류 정보"):
                    st.exception(e)
                st.info("💡 해결 방법:\n- GOOGLE_API_KEY가 설정되어 있는지 확인하세요\n- 인덱스가 구축되어 있는지 확인하세요 (`python3 -m src.ingestion.build_index`)")

# 예시 질문
st.markdown("---")
st.markdown("### 💡 예시 질문")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("보유기간은?", use_container_width=True):
        st.session_state.query_input = "개인정보 보유기간은 얼마나 되나요?"

with col2:
    if st.button("위탁 시 고려사항?", use_container_width=True):
        st.session_state.query_input = "개인정보처리 위탁 시 고려사항은 무엇인가요?"

with col3:
    if st.button("국외이전 절차?", use_container_width=True):
        st.session_state.query_input = "개인정보를 국외로 이전할 때 필요한 절차는?"

