"""Chat routes — RAG query and streaming endpoints.

POST /api/chat      — Full RAG: question → answer + sources
GET  /api/chat/stream — SSE streaming RAG response
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_playground.m07_rag_pipeline.generate import (
    generate_answer,
    generate_streaming_answer,
)
from server.deps import get_state

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    elapsed_ms: float


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a question and get a RAG answer with sources."""
    state = get_state()

    response = await generate_answer(
        query=req.question,
        store=state.vector_store,
        top_k=req.top_k,
    )

    return ChatResponse(
        answer=response.answer,
        sources=[
            {
                "content": s.chunk.content[:300],
                "score": round(s.score, 4),
                "rank": s.rank,
                "source": s.chunk.metadata.get("source", "unknown"),
            }
            for s in response.sources
        ],
        elapsed_ms=round(response.elapsed_ms, 1),
    )


@router.get("/chat/stream")
async def chat_stream(question: str, top_k: int = 3):
    """Stream a RAG answer via Server-Sent Events."""
    state = get_state()

    async def event_stream():
        try:
            async for token in generate_streaming_answer(
                query=question,
                store=state.vector_store,
                top_k=top_k,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0)  # Yield control
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
