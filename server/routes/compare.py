"""Compare route — RAG vs non-RAG side-by-side.

POST /api/compare — Answer a question with and without retrieval.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_playground.config.providers.base import ChatMessage
from rag_playground.m01_llm_basics.chat import chat_completion
from rag_playground.m07_rag_pipeline.generate import generate_answer
from server.deps import get_state

router = APIRouter(prefix="/api", tags=["compare"])


class CompareRequest(BaseModel):
    question: str
    top_k: int = 3


class CompareResponse(BaseModel):
    question: str
    rag_answer: str
    rag_elapsed_ms: float
    rag_sources: list[dict]
    llm_only_answer: str
    llm_only_elapsed_ms: float


@router.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """Answer a question both with RAG and without (LLM-only).

    Side-by-side comparison shows the value of retrieval.
    """
    state = get_state()
    import time

    # RAG answer (with retrieval)
    rag_response = await generate_answer(
        query=req.question,
        store=state.vector_store,
        top_k=req.top_k,
    )

    # LLM-only answer (no retrieval)
    start = time.perf_counter()
    messages = [ChatMessage(role="user", content=req.question)]
    llm_answer = await chat_completion(messages, temperature=0.1)
    llm_elapsed = (time.perf_counter() - start) * 1000

    return CompareResponse(
        question=req.question,
        rag_answer=rag_response.answer,
        rag_elapsed_ms=round(rag_response.elapsed_ms, 1),
        rag_sources=[
            {
                "content": s.chunk.content[:200],
                "score": round(s.score, 4),
                "rank": s.rank,
            }
            for s in rag_response.sources
        ],
        llm_only_answer=llm_answer,
        llm_only_elapsed_ms=round(llm_elapsed, 1),
    )
