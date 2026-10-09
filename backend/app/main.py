
from fastapi import FastAPI

from app.api.uavs import router as uavs_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description=(
        "AI-driven Integrated Vehicle Health "
        "Management platform for UAVs."
    ),
    version=settings.app_version,
)


@app.get("/")
def root():
    return {
        "project": "AeroVital",
        "message": "UAV health management API is running",
        "status": "operational",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.environment,
    }


app.include_router(
    uavs_router,
    prefix=settings.api_prefix,
)
