import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from routers.documents import router as documents_router
from routers.query import router as query_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Multi-Document RAG Pipeline", version="0.1.0", lifespan=lifespan)

_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001")
allowed_origins = [o.strip() for o in _origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
