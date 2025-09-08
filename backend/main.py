# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from core.config import settings
from db.database import engine, Base
from db.init_db import init_db
from routers import health, runs, graphs, pipeline, query, results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up GraphLLM Backend...")
    Base.metadata.create_all(bind=engine)
    init_db()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(runs.router, prefix="/runs", tags=["runs"])
app.include_router(graphs.router, prefix="/graphs", tags=["graphs"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(results.router, prefix="/results", tags=["results"])

@app.get("/")
def root():
    return {"message": "GraphLLM Backend API", "version": "0.1.0"}