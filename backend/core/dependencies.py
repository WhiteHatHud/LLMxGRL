# DB session and settings dependencies
# core/dependencies.py
from sqlalchemy.orm import Session
from db.database import SessionLocal
from core.config import settings

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_settings():
    return settings