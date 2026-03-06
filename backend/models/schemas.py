from pydantic import BaseModel


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str
    chunk_count: int
    status: str


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    n_results: int = 5


class SourceCitation(BaseModel):
    document: str
    chunk_text: str
    score: float
    page: int | None = None
    section: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    session_id: str


class CompareRequest(BaseModel):
    question: str
    doc_ids: list[str]
    n_results: int = 3


class ProcessingEvent(BaseModel):
    step: str
    status: str
    detail: str | None = None
    progress: float | None = None
    chunk_count: int | None = None
    doc_id: str | None = None
    file_hash: str | None = None
