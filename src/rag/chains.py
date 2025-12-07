"""RAG 체인 모듈

5.3 RAG 체인 및 Self-RAG 체인을 구현합니다.
기본 RAG 체인을 정의합니다.
"""
import logging

from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, LLM_MODEL
from src.prompts import SYSTEM_PROMPT_RAG, USER_QUESTION_TEMPLATE
from src.retrieval.retriever import load_hybrid_retriever

logger = logging.getLogger(__name__)


def create_rag_chain():
    """RAG 체인을 생성합니다.
    
    5.3 RAG 체인 및 Self-RAG 체인에 따라:
    - 하이브리드 검색기 사용
    - SYSTEM_PROMPT_RAG를 시스템 메시지로 사용
    
    Returns:
        RetrievalQA 체인
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # LLM 초기화
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    
    # 검색기 로드 (후처리 로직 포함)
    print("[진행] 하이브리드 검색기 로드 중...")
    retriever, vectorstore, dense_retriever, bm25_retriever = load_hybrid_retriever(use_postprocessing=True)
    print("[진행] 하이브리드 검색기 로드 완료 (후처리 로직 포함)")
    
    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_RAG),
        ("human", USER_QUESTION_TEMPLATE),
    ])
    
    # RetrievalQA 체인 생성
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        logger.info("RAG 체인 생성 완료")
        return qa_chain
    except Exception as e:
        logger.error(f"RAG 체인 생성 중 오류: {e}", exc_info=True)
        raise ValueError(f"RAG 체인 생성 실패: {str(e)}. 인덱스가 구축되었는지 확인하세요.")


def query_rag_chain(question: str, chain=None) -> dict:
    """RAG 체인에 질문을 실행합니다.
    
    Args:
        question: 사용자 질문
        chain: RAG 체인 (None이면 새로 생성)
        
    Returns:
        {"answer": str, "sources": List[Document]} 형태의 딕셔너리
    """
    if chain is None:
        chain = create_rag_chain()
    
    logger.info(f"질문 처리 중: {question}")
    print(f"[진행] RAG 체인 실행 중: {question}")
    
    try:
        # RetrievalQA는 "query" 키를 사용
        print("[진행] LLM에 질문 전송 중...")
        
        result = chain.invoke({"query": question})
        print("[진행] LLM 응답 수신 완료")
        
        # LLM 응답 전체 로깅
        logger.info(f"[LLM 응답] 원본 응답 타입: {type(result)}")
        print(f"[LLM 응답] 원본 응답 타입: {type(result)}")
        if isinstance(result, dict):
            logger.info(f"[LLM 응답] 응답 키: {result.keys()}")
            print(f"[LLM 응답] 응답 키: {result.keys()}")
            for key, value in result.items():
                if key != "source_documents":  # source_documents는 너무 길 수 있음
                    logger.info(f"[LLM 응답] {key}: {str(value)[:500]}")  # 처음 500자만
                    print(f"[LLM 응답] {key}: {str(value)[:500]}")
        else:
            logger.info(f"[LLM 응답] 응답 내용: {str(result)[:500]}")
            print(f"[LLM 응답] 응답 내용: {str(result)[:500]}")
        
        # 결과 파싱 (다양한 형식 대응)
        if isinstance(result, dict):
            answer = result.get("result", result.get("answer", ""))
            sources = result.get("source_documents", result.get("sources", []))
        else:
            # 문자열로 반환된 경우
            answer = str(result) if result else ""
            sources = []
        
        # 최종 답변 로깅
        logger.info(f"[LLM 응답] 최종 답변: {answer}")
        print(f"[LLM 응답] 최종 답변: {answer}")
        
        # 검색된 문서 정보 로깅 (실제 사용된 참고 문서)
        logger.info(f"실제 사용된 참고 문서 수: {len(sources)}")
        print(f"[진행] 실제 사용된 참고 문서 수: {len(sources)}")
        
        if sources:
            total_context_length = sum(len(doc.page_content) for doc in sources)
            logger.info(f"참고 문서들의 총 컨텍스트 길이: {total_context_length}자")
            print(f"[진행] 참고 문서들의 총 컨텍스트 길이: {total_context_length}자")
            
            for i, doc in enumerate(sources[:5], 1):  # 상위 5개 출력
                # 메타데이터 정보 추출
                jo = doc.metadata.get('조', 'N/A')
                jo_title = doc.metadata.get('조항제목', 'N/A')
                source = doc.metadata.get('source', doc.metadata.get('source_file', 'N/A'))
                chunk_idx = doc.metadata.get('chunk_index', 'N/A')
                
                # 점수 정보 추출
                dense_score = doc.metadata.get('dense_score', None)
                sparse_score = doc.metadata.get('sparse_score', None)
                final_score = doc.metadata.get('final_score', None)
                
                # 문서 내용 미리보기 (처음 150자)
                content_preview = doc.page_content[:150].replace('\n', ' ').strip()
                if len(doc.page_content) > 150:
                    content_preview += "..."
                
                # 점수 정보 포맷팅
                score_info = ""
                if final_score is not None:
                    score_info = f", 최종점수: {final_score:.4f}"
                if dense_score is not None and sparse_score is not None:
                    score_info = f", Dense: {dense_score:.4f}, Sparse: {sparse_score:.4f}, Final: {final_score:.4f}"
                
                # 출력
                doc_info = (
                    f"  [{i}] 조: {jo}, 조항제목: {jo_title}, "
                    f"출처: {source}, 청크: {chunk_idx}, "
                    f"길이: {len(doc.page_content)}자{score_info}\n"
                    f"       내용: {content_preview}"
                )
                logger.info(doc_info)
                print(f"[진행] {doc_info}")
                logger.debug(f"전체 내용: {doc.page_content}")
        else:
            logger.warning("검색된 문서가 없습니다!")
            print("[경고] 검색된 문서가 없습니다!")
            
        logger.info(f"답변 생성 완료 (소스 {len(sources)}개)")
        
        # 검색 결과가 없거나 답변이 비어있는 경우
        if not sources:
            logger.warning("검색된 문서가 없습니다!")
            print("[경고] 검색된 문서가 없습니다!")
            answer = "검색된 문서가 없습니다. 인덱스가 구축되었는지 확인하세요."
        elif not answer or answer.strip() == "":
            logger.warning("답변이 비어있습니다!")
            print("[경고] 답변이 비어있습니다!")
            # 검색된 문서가 있지만 답변이 없는 경우, 문서 내용 요약
            answer = f"검색된 문서는 {len(sources)}개이지만 답변을 생성할 수 없었습니다. "
            answer += "검색된 문서의 조 번호: " + ", ".join([
                str(doc.metadata.get('조', 'N/A')) for doc in sources[:3]
            ])
        
        return {
            "answer": answer,
            "sources": sources,
        }
    except Exception as e:
        logger.error(f"RAG 체인 실행 중 오류: {e}", exc_info=True)
        print(f"[오류] RAG 체인 실행 중 오류: {e}")
        raise

