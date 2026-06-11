"""Document routes — list and upload documents.

GET  /api/documents     — List indexed documents
POST /api/documents     — Upload + index a document
"""

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from rag_playground.config.env import settings
from rag_playground.m02_data_loading.document import Document, load_markdown_file
from rag_playground.m03_chunking.recursive_splitter import split_recursively
from rag_playground.m04_embeddings.embed import embed_documents
from server.deps import get_state

router = APIRouter(prefix="/api", tags=["documents"])


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunk_count: int
    total_chars: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """List all documents currently indexed in the vector store."""
    state = get_state()

    # Collect unique documents from chunk metadata
    seen: dict[str, DocumentInfo] = {}
    for i in range(state.vector_store.size):
        _, chunk, _ = (
            state.vector_store.get_by_id(list(state.vector_store._id_index.keys())[i])
            if i < len(state.vector_store._id_index)
            else (None, None, None)
        )
        # This is O(n²) in worst case — fine for demo scale

    # Simpler approach: iterate internal storage
    total_chars = 0
    for chunk in state.vector_store._chunks:
        doc_id = chunk.document_id
        if doc_id not in seen:
            seen[doc_id] = DocumentInfo(
                id=doc_id,
                filename=chunk.metadata.get("filename", doc_id),
                chunk_count=0,
                total_chars=0,
            )
        seen[doc_id].chunk_count += 1
        seen[doc_id].total_chars += len(chunk.content)
        total_chars += len(chunk.content)

    return DocumentsResponse(
        documents=list(seen.values()),
        total_chunks=state.vector_store.size,
    )


@router.post("/documents")
async def upload_document(file: UploadFile):
    """Upload and index a markdown or text document."""
    if not file.filename:
        return {"error": "No filename provided"}

    # Read file content
    content = await file.read()
    text = content.decode("utf-8")

    # Create a temp file to use existing loader
    import tempfile
    import os

    suffix = ".md" if file.filename.endswith(".md") else ".txt"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        doc = (
            await load_markdown_file(tmp_path)
            if suffix == ".md"
            else Document(
                page_content=text,
                metadata={"source": file.filename, "filename": file.filename},
            )
        )

        # Chunk the document
        chunks = split_recursively(
            doc, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

        # Embed the chunks
        chunk_texts = [c.content for c in chunks]
        vectors = await embed_documents(chunk_texts)

        # Add to store
        state = get_state()
        for vec, chunk in zip(vectors, chunks):
            state.vector_store.add(vec, chunk)

        return {
            "status": "indexed",
            "filename": file.filename,
            "chunks": len(chunks),
            "total_store_size": state.vector_store.size,
        }
    finally:
        os.unlink(tmp_path)
