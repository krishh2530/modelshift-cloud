from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import secrets

# Make sure to replace YOUR_ACTUAL_PASSWORD!
SQLALCHEMY_DATABASE_URL = "postgresql://neondb_owner:npg_wBGIHVX57uQp@ep-blue-bonus-a4mv4f1l.us-east-1.aws.neon.tech/neondb?sslmode=require"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 1. DEFINE THE USER TABLE ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    api_key = Column(String, unique=True, index=True)

# --- 2. CREATE THE TABLE IN NEON ---
Base.metadata.create_all(bind=engine)

# --- 3. DATABASE CONNECTION FUNCTION ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 4. API KEY GENERATOR ---
def generate_api_key():
    return secrets.token_urlsafe(32)