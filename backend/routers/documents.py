import hashlib
import json
import asyncio
from pathlib import PurePosixPath

from fastapi import APIRouter, UploadFile, File, HTTPException
from sse_starlette.sse import EventSourceResponse

from db.chroma_client import delete_document, get_all_chunks
from models.schemas import DocumentInfo
from services.pipeline import ingest_document
from services.retriever import invalidate_bm25_cache

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory document registry (maps doc_id -> metadata)
_documents: dict[str, dict] = {}
_registry_loaded = False
_registry_lock = asyncio.Lock()


def _rebuild_registry() -> None:
    """Rebuild the document registry from ChromaDB on first access."""
    global _registry_loaded
    if _registry_loaded:
        return
    chunks = get_all_chunks()
    for chunk in chunks:
        doc_id = chunk["meta"]["doc_id"]
        if doc_id not in _documents:
            _documents[doc_id] = {
                "id": doc_id,
                "filename": chunk["meta"]["filename"],
                "content_type": chunk["meta"].get("content_type", "unknown"),
                "chunk_count": 0,
                "status": "ready",
                "file_hash": chunk["meta"].get("file_hash", ""),
            }
        _documents[doc_id]["chunk_count"] += 1
    _registry_loaded = True


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document. Returns SSE stream of processing events."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    safe_filename = PurePosixPath(file.filename).name if file.filename else "unnamed"
    file_hash = hashlib.sha256(content).hexdigest()

    async with _registry_lock:
        _rebuild_registry()
    for doc in _documents.values():
        if doc["filename"] == safe_filename or doc.get("file_hash") == file_hash:
            raise HTTPException(status_code=409, detail="File already uploaded.")

    async def event_stream():
        doc_id = None
        async for event in ingest_document(content, safe_filename, file.content_type):
            if event.step == "complete" and event.status == "done":
                doc_id = event.doc_id
                _documents[doc_id] = {
                    "id": doc_id,
                    "filename": safe_filename,
                    "content_type": file.content_type or "unknown",
                    "chunk_count": event.chunk_count or 0,
                    "status": "ready",
                    "file_hash": file_hash,
                }
                invalidate_bm25_cache()
            elif event.status == "error":
                pass  # Don't register failed documents

            yield {
                "event": event.step,
                "data": json.dumps(event.model_dump()),
            }

    return EventSourceResponse(event_stream())


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    async with _registry_lock:
        _rebuild_registry()
    return [DocumentInfo(**doc) for doc in _documents.values()]


@router.delete("/{doc_id}")
async def delete_doc(doc_id: str):
    async with _registry_lock:
        _rebuild_registry()
    if doc_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document(doc_id)
    del _documents[doc_id]
    invalidate_bm25_cache()
    return {"status": "deleted", "doc_id": doc_id}
