
from app.db.database import Base, engine
from app.db.models import UAV


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
    print("Tables:", list(Base.metadata.tables.keys()))


if __name__ == "__main__":
    init_db()
