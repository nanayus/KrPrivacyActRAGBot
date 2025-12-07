"""FastAPI 서버

간단한 쿼리 엔드포인트를 제공합니다.
"""
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.self_rag import self_rag_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="개인정보보호법 RAG API",
    description="한국 개인정보보호법 관련 문서 기반 질의응답 API",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    """쿼리 요청 모델"""
    question: str


class SourceInfo(BaseModel):
    """소스 정보 모델"""
    source: str
    document_type: str
    content_preview: str


class QueryResponse(BaseModel):
    """쿼리 응답 모델"""
    answer: str
    sources: List[SourceInfo]
    self_check_result: str
    corrected: bool


@app.get("/")
def root():
    """루트 엔드포인트"""
    return {
        "message": "개인정보보호법 RAG API",
        "version": "0.1.0",
        "endpoints": {
            "/query": "POST - 질문을 보내고 답변을 받습니다."
        }
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """질문을 처리하고 답변을 반환합니다.
    
    Self-RAG 파이프라인을 사용하여:
    1. 하이브리드 검색으로 관련 문서 검색
    2. RAG 체인으로 초기 답변 생성
    3. Self-check로 답변 검증 및 수정
    
    Args:
        request: QueryRequest 객체 (question 필드 포함)
        
    Returns:
        QueryResponse 객체 (answer, sources, self_check_result, corrected)
    """
    try:
        logger.info(f"쿼리 수신: {request.question}")
        
        # Self-RAG 파이프라인 실행
        result = self_rag_query(request.question)
        
        # 소스 정보 포맷팅
        sources_info = []
        for doc in result["sources"]:
            sources_info.append(SourceInfo(
                source=doc.metadata.get("source", "Unknown"),
                document_type=doc.metadata.get("document_type", "Unknown"),
                content_preview=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            ))
        
        response = QueryResponse(
            answer=result["answer"],
            sources=sources_info,
            self_check_result=result["self_check_result"],
            corrected=result["corrected"],
        )
        
        logger.info("쿼리 처리 완료")
        return response
        
    except Exception as e:
        logger.error(f"쿼리 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)


