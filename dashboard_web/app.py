# dashboard_web/app.py
from __future__ import annotations
import sys
import subprocess
import json
import os
import re
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi.security import APIKeyHeader
from fastapi import Security, FastAPI, HTTPException, Query, Response, Depends, BackgroundTasks, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
import numpy as np

from pydantic import BaseModel
import bcrypt
from sqlalchemy.orm import Session
from dashboard_web.database import get_db, User, Run, generate_api_key
from dashboard_web.email_alert import send_critical_alert

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class UserCreate(BaseModel):
    email: str
    password: str

class SDKLogin(BaseModel):
    email: str
    password: str

# -------------------------------------------------------------------
# Paths & Constants (kept for static files / templates only)
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,80}$")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("[+] Server started. NeonDB persistence active.")
    print("="*60 + "\n")
    yield

app = FastAPI(title="ModelShift-Lite Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def no_cache_headers(resp: Response) -> None:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"

def _safe_number(value: Any) -> Optional[float]:
    try:
        if value is None: return None
        return float(value)
    except (TypeError, ValueError): return None

def _safe_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    return None

def _parse_iso_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip(): return None
    s = value.strip()
    try:
        if s.endswith("Z"): s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except: return None

def _sort_dt_key(dt: Optional[datetime]) -> datetime:
    if dt is None: return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None: return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _best_status(run: Dict[str, Any]) -> str:
    if not isinstance(run, dict): return "UNKNOWN"
    top = _safe_str(run.get("status")) or _safe_str(run.get("state"))
    if top: return top
    decision = run.get("decision")
    if isinstance(decision, dict):
        if d_status := _safe_str(decision.get("status")): return d_status
    return "UNKNOWN"

def _summary_obj(run: Dict[str, Any]) -> Dict[str, Any]:
    raw = run.get("summary")
    return raw if isinstance(raw, dict) else {}

def normalize_history_item(run: Dict[str, Any], saved_at: str) -> Dict[str, Any]:
    status = _safe_str(run.get("status")) or "UNKNOWN"
    pred_ks = _safe_number(run.get("drifted_pred_ks")) or _safe_number(run.get("pred_ks"))
    delta_entropy = _safe_number(run.get("drifted_entropy_change")) or _safe_number(run.get("delta_entropy"))
    return {
        "saved_at": saved_at,
        "run_id": run.get("run_id"),
        "generated_at": run.get("generated_at"),
        "status": status,
        "clean_health": _safe_number(run.get("clean_health")),
        "drifted_health": _safe_number(run.get("drifted_health")),
        "drifted_pred_ks": pred_ks,
        "drifted_entropy_change": delta_entropy,
        "drifted_last_window_feature": run.get("drifted_last_window_feature"),
        "drifted_last_window_ks": _safe_number(run.get("drifted_last_window_ks")),
    }

# -------------------------------------------------------------------
# DB helpers — replace all file I/O
# -------------------------------------------------------------------
def db_save_run(db: Session, api_key: str, run_id: str, payload: Dict[str, Any]) -> None:
    """Insert or update a run row in NeonDB."""
    existing = db.query(Run).filter(Run.api_key == api_key, Run.run_id == run_id).first()
    if existing:
        existing.payload_json = json.dumps(payload, ensure_ascii=False)
        existing.saved_at = datetime.utcnow()
    else:
        row = Run(
            api_key=api_key,
            run_id=run_id,
            payload_json=json.dumps(payload, ensure_ascii=False),
            saved_at=datetime.utcnow(),
        )
        db.add(row)
    db.commit()

def db_get_latest_two(db: Session, api_key: str):
    """Return the two most recent Run rows for this user."""
    rows = (
        db.query(Run)
        .filter(Run.api_key == api_key)
        .order_by(Run.saved_at.desc())
        .limit(2)
        .all()
    )
    return rows

def db_get_run(db: Session, api_key: str, run_id: str) -> Optional[Dict[str, Any]]:
    row = db.query(Run).filter(Run.api_key == api_key, Run.run_id == run_id).first()
    if not row: return None
    try: return json.loads(row.payload_json)
    except: return None

def db_get_history(db: Session, api_key: str, n: int = 50) -> List[Dict[str, Any]]:
    rows = (
        db.query(Run)
        .filter(Run.api_key == api_key)
        .order_by(Run.saved_at.desc())
        .limit(n)
        .all()
    )
    items = []
    for row in rows:
        try:
            payload = json.loads(row.payload_json)
            item = normalize_history_item(payload, row.saved_at.isoformat())
            items.append(item)
        except:
            continue
    return items

def db_clear_history(db: Session, api_key: str) -> None:
    db.query(Run).filter(Run.api_key == api_key).delete()
    db.commit()

# -------------------------------------------------------------------
# Security & Auth
# -------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_current_user(api_key: str = Security(api_key_header), db: Session = Depends(get_db)):
    if not api_key: raise HTTPException(status_code=403, detail="No API Key provided.")
    clean_key = api_key.replace("Bearer ", "").replace("bearer ", "").strip('"').strip("'").strip()
    user = db.query(User).filter(User.api_key == clean_key).first()
    if not user: raise HTTPException(status_code=403, detail="Invalid API Key. Access Denied.")
    return user

def get_user_from_query(api_key: str = Query(None), db: Session = Depends(get_db)):
    if not api_key: raise HTTPException(status_code=403, detail="No API Key provided.")
    clean_key = api_key.replace("Bearer ", "").replace("bearer ", "").strip('"').strip("'").strip()
    user = db.query(User).filter(User.api_key == clean_key).first()
    if not user: raise HTTPException(status_code=403, detail="Invalid API Key.")
    return user

@app.post("/api/auth/signup")
def create_user(user: UserCreate, response: Response, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), salt).decode('utf-8')
    new_user = User(email=user.email, hashed_password=hashed_password, api_key=generate_api_key())
    db.add(new_user)
    db.commit()
    response.set_cookie(
        key="ms_session", value=new_user.api_key,
        httponly=True, samesite="lax", secure=True, max_age=86400 * 30
    )
    return {"message": "User created", "email": new_user.email, "api_key": new_user.api_key}

@app.post("/api/auth/login")
def login_user(user: UserCreate, response: Response, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not bcrypt.checkpw(user.password.encode('utf-8'), db_user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Account doesn't exist or Invalid Password")
    if not getattr(db_user, "api_key", None):
        db_user.api_key = generate_api_key()
        db.commit()
    response.set_cookie(
        key="ms_session", value=db_user.api_key,
        httponly=True, samesite="lax", secure=True, max_age=86400 * 30
    )
    return {"message": "Auth successful", "email": db_user.email, "api_key": db_user.api_key}

@app.post("/api/auth/logout")
def logout_user(response: Response):
    response.delete_cookie("ms_session", samesite="lax", secure=True)
    return {"message": "Logged out"}

@app.post("/api/v1/sdk_login")
def sdk_login_endpoint(credentials: SDKLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not bcrypt.checkpw(credentials.password.encode('utf-8'), user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not getattr(user, "api_key", None):
        user.api_key = generate_api_key()
        db.commit()
    return {"api_key": user.api_key}

# -------------------------------------------------------------------
# MAIN API — all reads/writes go to NeonDB now
# -------------------------------------------------------------------
@app.post("/api/v1/track")
def receive_drift_data(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    run_id = payload.get("run_id")
    if not run_id: raise HTTPException(status_code=400, detail="Missing run_id")

    # Save to NeonDB — survives Render restarts forever
    db_save_run(db, user.api_key, run_id, payload)

    # Fire email alert if critical
    status = payload.get("status", "")
    if status == "CRITICAL_DRIFT":
        background_tasks.add_task(
            send_critical_alert,
            recipient_email=user.email,
            run_id=run_id,
            feature=payload.get("drifted_last_window_feature") or "Unknown",
            ks_score=payload.get("drifted_last_window_ks") or 0.0,
            health=payload.get("drifted_health") or 0.0
        )
    return {"status": "success", "run_id": run_id}

@app.get("/api/results")
def api_results(response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db_get_latest_two(db, user.api_key)
    no_cache_headers(response)
    if not rows:
        return {"latest": {}, "previous": {}, "live_connected": False}
    latest = json.loads(rows[0].payload_json) if rows else {}
    previous = json.loads(rows[1].payload_json) if len(rows) > 1 else {}
    return {"latest": latest, "previous": previous, "live_connected": bool(latest)}

@app.get("/api/history")
def api_history(response: Response, n: int = Query(default=50, ge=1, le=200), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db_get_history(db, user.api_key, n)
    no_cache_headers(response)
    return {"runs": items, "source": "neondb"}

@app.post("/api/history/clear")
def api_history_clear(response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_clear_history(db, user.api_key)
    no_cache_headers(response)
    return {"ok": True, "message": "History cleared from database"}

@app.get("/api/run/{run_id}")
def api_run(run_id: str, response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not SAFE_RUN_ID_RE.match(run_id): raise HTTPException(status_code=400)
    data = db_get_run(db, user.api_key, run_id)
    if not data: raise HTTPException(status_code=404, detail="Run not found")
    no_cache_headers(response)
    return data

# -------------------------------------------------------------------
# DATASET VIEWER
# -------------------------------------------------------------------
@app.get("/dataset/{run_id}", response_class=HTMLResponse)
def view_dataset(run_id: str, user: User = Depends(get_user_from_query), db: Session = Depends(get_db)):
    if not SAFE_RUN_ID_RE.match(run_id): raise HTTPException(status_code=400)
    data = db_get_run(db, user.api_key, run_id)
    if not data:
        return HTMLResponse("<h2 style='color:white;text-align:center;padding:50px;'>Run data not found.</h2>", status_code=404)

    records_after = data.get("dataset_sample", [])
    records_before = data.get("baseline_sample", [])

    if not records_after:
        return HTMLResponse("<h2 style='color:white;text-align:center;padding:50px;'>No dataset sample attached to this run.</h2>", status_code=404)

    import math as _math
    df_after = pd.DataFrame(records_after)
    df_before = pd.DataFrame(records_before) if records_before else pd.DataFrame(columns=df_after.columns)

    after_html = ["<table class='dataset-table'><thead><tr><th>#</th>"]
    for col in df_after.columns: after_html.append(f"<th>{col}</th>")
    after_html.append("</tr></thead><tbody>")

    for i, row_after in enumerate(records_after):
        after_html.append("<tr>")
        after_html.append(f"<td><b>{i}</b></td>")
        row_before = records_before[i] if i < len(records_before) else None
        for col in df_after.columns:
            val_after = row_after.get(col)
            val_before = row_before.get(col) if row_before else None
            is_diff = False
            if pd.isna(val_after) and pd.isna(val_before): is_diff = False
            elif pd.isna(val_after) != pd.isna(val_before): is_diff = True
            else:
                try:
                    if not _math.isclose(float(val_after), float(val_before), abs_tol=1e-3): is_diff = True
                except:
                    if str(val_after).strip() != str(val_before).strip(): is_diff = True
            td_class = "changed" if is_diff else ""
            display_val = f"{float(val_after):.4f}" if isinstance(val_after, (float, int, np.number)) and not pd.isna(val_after) else str(val_after)
            after_html.append(f"<td class='{td_class}'>{display_val}</td>")
        after_html.append("</tr>")
    after_html.append("</tbody></table>")
    before_html = df_before.round(4).to_html(classes="dataset-table", index=True) if not df_before.empty else "<h3 style='color:white'>No Baseline Data.</h3>"

    return HTMLResponse(content=f"""
    <!DOCTYPE html><html><head>
    <title>ModelShift - Dataset Inspector</title>
    <style>
      body{{font-family:'Segoe UI',monospace;background:#0b0c0d;color:#e9eef2;padding:20px}}
      .tabs{{display:flex;gap:10px;margin-bottom:15px}}
      .tab-btn{{background:#1d2023;color:#9aa4ad;border:1px solid #2a2d30;padding:10px 20px;cursor:pointer}}
      .tab-btn.active{{background:rgba(209,31,31,0.2);border-color:#d11f1f;color:white;font-weight:bold}}
      .tab-content{{display:none;overflow:auto;max-height:80vh;border:1px solid #2a2d30;background:#0f1112}}
      .tab-content.active{{display:block}}
      .dataset-table{{border-collapse:collapse;width:100%;font-size:12px}}
      .dataset-table th,.dataset-table td{{border:1px solid #2a2d30;padding:6px 10px;text-align:right;white-space:nowrap}}
      .dataset-table th{{background:#1d2023;position:sticky;top:0;color:white}}
      .changed{{background-color:rgba(209,31,31,0.5)!important;color:white;font-weight:bold}}
      .note{{font-size:13px;color:#9aa4ad;margin-bottom:15px;border-left:3px solid #d11f1f;padding-left:10px}}
    </style></head><body>
    <h1>Run ID: {run_id}</h1>
    <div class="note">15-row sample (Private to {user.email}). Red = difference &gt; 0.001</div>
    <div class="tabs">
      <button class="tab-btn active" onclick="showTab('after',this)">SHEET 1: LIVE (DRIFTED)</button>
      <button class="tab-btn" onclick="showTab('before',this)">SHEET 2: BASELINE (CLEAN)</button>
    </div>
    <div id="after" class="tab-content active">{"".join(after_html)}</div>
    <div id="before" class="tab-content">{before_html}</div>
    <script>
      function showTab(t,b){{
        document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));
        document.getElementById(t).classList.add('active');b.classList.add('active');
      }}
    </script></body></html>
    """)

# -------------------------------------------------------------------
# SELF TEST (kept, uses subprocess)
# -------------------------------------------------------------------
@app.get("/api/selftest")
def api_selftest(response: Response, user: User = Depends(get_current_user)):
    no_cache_headers(response)
    return {"ok": True, "message": "Self-test endpoint active."}

@app.post("/api/selftest/run")
def api_selftest_run(response: Response, user: User = Depends(get_current_user)):
    no_cache_headers(response)
    return {"ok": True, "message": "Self-test triggered (no local file system needed)."}

@app.get("/api/health")
def api_health(response: Response):
    return {"ok": True, "service": "modelshift-saas", "storage": "neondb"}

# -------------------------------------------------------------------
# UI PAGE ROUTING
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, ms_session: Optional[str] = Cookie(default=None), db: Session = Depends(get_db)):
    if not ms_session:
        return RedirectResponse(url="/login", status_code=302)
    user = db.query(User).filter(User.api_key == ms_session).first()
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request=request, name="index.html")
@app.get("/guide", response_class=HTMLResponse)
def guide_page(request: Request):
    return templates.TemplateResponse(request=request, name="guide.html")
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html")