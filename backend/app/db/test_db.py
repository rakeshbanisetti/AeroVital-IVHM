
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import UAV


def test_uav_database() -> None:
    with SessionLocal() as session:
        uav = session.scalar(
            select(UAV).where(UAV.uav_code == "UAV-TEST-001")
        )

        if uav is None:
            uav = UAV(
                uav_code="UAV-TEST-001",
                model_name="AeroVital Test UAV",
                manufacturer="AeroVital Research",
            )
            session.add(uav)
            session.commit()
            session.refresh(uav)

        saved_uav = session.scalar(
            select(UAV).where(UAV.uav_code == "UAV-TEST-001")
        )

        print("Database read/write test passed.")
        print("UAV code:", saved_uav.uav_code)
        print("Model:", saved_uav.model_name)
        print("Record ID:", saved_uav.id)


if __name__ == "__main__":
    test_uav_database()
