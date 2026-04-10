import os
import secrets
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_wBGIHVX57uQp@ep-blue-bonus-a4mv4f1l.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- USER TABLE ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    api_key = Column(String, unique=True, index=True)


# --- RUN TABLE (replaces flat files on disk) ---
class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, index=True)       # links run to a user
    run_id = Column(String, index=True)        # e.g. "run_96a5d8a7"
    payload_json = Column(Text)                # full JSON payload as string
    saved_at = Column(DateTime, default=datetime.utcnow)


# --- CREATE TABLES ---
Base.metadata.create_all(bind=engine)


# --- DB SESSION ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- API KEY GENERATOR ---
def generate_api_key():
    return secrets.token_urlsafe(32)