from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.services.evidence_registry_service import initialize_evidence_registry
from app.services.matx_database_service import initialize_matx_database
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_evidence_registry()
    initialize_matx_database()
    yield
app = FastAPI(
    title="MAT-X API",
    description=(
        "AI-Assisted Materials State Reconstruction "
        "and Active Investigation Engine"
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router, prefix="/api/v1")
@app.get("/")
def root():
    return {
        "project": "MAT-X",
        "version": "1.0.0",
        "message": "Materials intelligence backend is running.",
        "documentation": "/docs",
    }
