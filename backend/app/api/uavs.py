
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import UAV
from app.schemas.uav import UAVCreate, UAVResponse

router = APIRouter(prefix="/uavs", tags=["UAVs"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_model=list[UAVResponse])
def list_uavs(db: Session = Depends(get_db)):
    statement = select(UAV).order_by(UAV.id)
    return db.scalars(statement).all()


@router.post(
    "/",
    response_model=UAVResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_uav(
    payload: UAVCreate,
    db: Session = Depends(get_db),
):
    existing = db.scalar(
        select(UAV).where(UAV.uav_code == payload.uav_code)
    )

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A UAV with this code already exists.",
        )

    uav = UAV(**payload.model_dump())
    db.add(uav)
    db.commit()
    db.refresh(uav)
    return uav
