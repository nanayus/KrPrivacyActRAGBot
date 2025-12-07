"""평가 스크립트

6. 평가 및 개선 계획에 따라 테스트 케이스를 실행하고 결과를 출력합니다.

사용법:
    python -m src.eval.evaluate
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from src.eval.test_cases import TEST_CASES
from src.rag.self_rag import self_rag_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate():
    """테스트 케이스를 실행하고 결과를 출력합니다."""
    logger.info(f"총 {len(TEST_CASES)}개의 테스트 케이스 실행 시작")
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        query = test_case["query"]
        notes = test_case.get("notes", "")
        
        logger.info(f"\n[{i}/{len(TEST_CASES)}] 질문: {query}")
        logger.info(f"참고사항: {notes}")
        
        try:
            # Self-RAG 파이프라인 실행
            result = self_rag_query(query)
            
            # 결과 저장
            eval_result = {
                "query": query,
                "notes": notes,
                "answer": result["answer"],
                "corrected": result["corrected"],
                "self_check_result": result["self_check_result"],
                "num_sources": len(result["sources"]),
                "sources": [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "document_type": doc.metadata.get("document_type", "Unknown"),
                    }
                    for doc in result["sources"]
                ],
            }
            
            results.append(eval_result)
            
            # 콘솔 출력
            print("\n" + "=" * 80)
            print(f"질문: {query}")
            print("-" * 80)
            print(f"답변:\n{result['answer']}")
            print("-" * 80)
            print(f"소스 수: {len(result['sources'])}")
            print(f"수정 여부: {result['corrected']}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"테스트 케이스 {i} 실행 중 오류: {e}", exc_info=True)
            results.append({
                "query": query,
                "notes": notes,
                "error": str(e),
            })
    
    # JSONL 로그 저장
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.jsonl"
    
    with open(log_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"평가 결과가 {log_file}에 저장되었습니다.")
    logger.info("평가 완료!")


if __name__ == "__main__":
    evaluate()


