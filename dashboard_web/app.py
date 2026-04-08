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
from fastapi import Security, FastAPI, HTTPException, Query, Response, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd
import numpy as np

from pydantic import BaseModel
import bcrypt
from sqlalchemy.orm import Session
from dashboard_web.database import get_db, User, generate_api_key
from dashboard_web.email_alert import send_critical_alert

# -------------------------------------------------------------------
# Security Setup
# -------------------------------------------------------------------
class UserCreate(BaseModel):
    email: str
    password: str

class SDKLogin(BaseModel):
    email: str
    password: str

# -------------------------------------------------------------------
# Paths & Constants
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_STALE_AFTER_SEC = 10
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,80}$")

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*60)
    print("[+] Server started. MULTI-TENANT architecture active.")
    print("="*60 + "\n")
    yield 

app = FastAPI(title="ModelShift-Lite Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# -------------------------------------------------------------------
# MULTI-TENANT WORKSPACE LOGIC
# -------------------------------------------------------------------
def get_workspace(api_key: str) -> Path:
    ws = DATA_DIR / api_key
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "runs").mkdir(parents=True, exist_ok=True)
    return ws

def get_paths(api_key: str) -> dict:
    ws = get_workspace(api_key)
    return {
        "latest": ws / "latest.json",
        "previous": ws / "previous.json",
        "history_index": ws / "history_index.json",
        "report_latest": ws / "report_latest.html",
        "heartbeat": ws / "live_heartbeat.touch",
        "runs_dir": ws / "runs",
        "selftest": ws / "selftest.json"
    }

# -------------------------------------------------------------------
# Core Helpers (Restored for Backward Compatibility)
# -------------------------------------------------------------------
def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists(): return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except: return {}

def read_text(path: Path) -> str:
    if not path.exists(): return ""
    try: return path.read_text(encoding="utf-8")
    except: return ""

def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)

def no_cache_headers(resp: Response) -> None:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"

def _is_fresh_file(path: Path) -> bool:
    if not path.exists() or not path.is_file(): return False
    return (time.time() - path.stat().st_mtime) <= RESULTS_STALE_AFTER_SEC

def _is_live_connected(paths: dict) -> bool:
    if paths["heartbeat"].exists() and _is_fresh_file(paths["heartbeat"]): return True
    if read_json(paths["latest"]): return True
    return False

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
    monitor_decision = run.get("monitor_decision")
    if isinstance(monitor_decision, dict):
        if md_status := _safe_str(monitor_decision.get("status")): return md_status
    return "UNKNOWN"

def _summary_obj(run: Dict[str, Any]) -> Dict[str, Any]:
    raw = run.get("summary")
    return raw if isinstance(raw, dict) else {}

def _pick_metric(run: Dict[str, Any], summary_key: str, top_level_fallbacks: List[str]) -> Optional[float]:
    summary = _summary_obj(run)
    if (v := _safe_number(summary.get(summary_key))) is not None: return v
    for key in top_level_fallbacks:
        if (v := _safe_number(run.get(key))) is not None: return v
    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        for key in top_level_fallbacks:
            if (v := _safe_number(metrics.get(key))) is not None: return v
    for blk_name in ("decision", "monitor_decision"):
        blk = run.get(blk_name)
        if isinstance(blk, dict):
            signals = blk.get("signals")
            if isinstance(signals, dict):
                sig_key_map = {
                    "drifted_pred_ks": ["prediction_ks", "pred_ks"],
                    "drifted_entropy_change": ["entropy_change", "delta_entropy"],
                    "clean_health": ["clean_health"],
                    "drifted_health": ["drifted_health"],
                }
                for sig_key in sig_key_map.get(summary_key, []):
                    if (v := _safe_number(signals.get(sig_key))) is not None: return v
    return None

def _pick_last_window_feature(run: Dict[str, Any]) -> Optional[str]:
    summary = _summary_obj(run)
    if name := _safe_str(summary.get("drifted_last_window_feature")): return name
    mdf = run.get("most_drifted_feature")
    if isinstance(mdf, dict): return _safe_str(mdf.get("feature")) or _safe_str(mdf.get("name"))
    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        mdf = metrics.get("most_drifted_feature")
        if isinstance(mdf, dict): return _safe_str(mdf.get("feature")) or _safe_str(mdf.get("name"))
    return None

def _pick_last_window_ks(run: Dict[str, Any]) -> Optional[float]:
    summary = _summary_obj(run)
    if (ks := _safe_number(summary.get("drifted_last_window_ks"))) is not None: return ks
    mdf = run.get("most_drifted_feature")
    if isinstance(mdf, dict):
        if (ks := _safe_number(mdf.get("ks"))) is not None: return ks
        if (ks := _safe_number(mdf.get("ks_statistic"))) is not None: return ks
    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        mdf = metrics.get("most_drifted_feature")
        if isinstance(mdf, dict):
            if (ks := _safe_number(mdf.get("ks"))) is not None: return ks
            if (ks := _safe_number(mdf.get("ks_statistic"))) is not None: return ks
    return None

def slim_run_payload(run: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = run.get("evaluation") if isinstance(run.get("evaluation"), dict) else None
    return {
        "saved_at": run.get("saved_at"),
        "run_id": run.get("run_id"),
        "generated_at": run.get("generated_at"),
        "status": _best_status(run),
        "window_size": run.get("window_size"),
        "clean_health": _pick_metric(run, "clean_health", ["clean_health"]),
        "drifted_health": _pick_metric(run, "drifted_health", ["drifted_health"]),
        "drifted_pred_ks": _pick_metric(run, "drifted_pred_ks", ["pred_ks", "drifted_pred_ks"]),
        "drifted_entropy_change": _pick_metric(run, "drifted_entropy_change", ["delta_entropy", "drifted_entropy_change"]),
        "drifted_last_window_feature": _pick_last_window_feature(run),
        "drifted_last_window_ks": _pick_last_window_ks(run),
        "evaluation": evaluation,
        "series_hash": run.get("series_hash"),
        "payload_hash": run.get("payload_hash"),
    }

def normalize_history_item(it: Dict[str, Any]) -> Dict[str, Any]:
    status = _safe_str(it.get("status")) or _safe_str(it.get("state")) or "UNKNOWN"
    pred_ks = _safe_number(it.get("drifted_pred_ks"))
    if pred_ks is None: pred_ks = _safe_number(it.get("pred_ks"))
    delta_entropy = _safe_number(it.get("drifted_entropy_change"))
    if delta_entropy is None: delta_entropy = _safe_number(it.get("delta_entropy"))
    saved_at = it.get("saved_at") or it.get("generated_at")
    return {
        "saved_at": saved_at,
        "run_id": it.get("run_id"),
        "generated_at": it.get("generated_at"),
        "status": status,
        "clean_health": _safe_number(it.get("clean_health")),
        "drifted_health": _safe_number(it.get("drifted_health")),
        "drifted_pred_ks": pred_ks,
        "drifted_entropy_change": delta_entropy,
        "drifted_last_window_feature": it.get("drifted_last_window_feature"),
        "drifted_last_window_ks": _safe_number(it.get("drifted_last_window_ks")),
        "payload_hash": it.get("payload_hash"),
        "series_hash": it.get("series_hash"),
    }

def sort_history_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(it: Dict[str, Any]):
        dt_gen = _parse_iso_dt(it.get("generated_at"))
        dt_saved = _parse_iso_dt(it.get("saved_at"))
        rid = str(it.get("run_id") or "")
        return (
            dt_gen is not None,
            _sort_dt_key(dt_gen) if dt_gen is not None else datetime.min.replace(tzinfo=timezone.utc),
            dt_saved is not None,
            _sort_dt_key(dt_saved) if dt_saved is not None else datetime.min.replace(tzinfo=timezone.utc),
            rid,
        )
    return sorted(items, key=_key, reverse=True)

# -------------------------------------------------------------------
# SECURITY & AUTH
# -------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_current_user(api_key: str = Security(api_key_header), db: Session = Depends(get_db)):
    if not api_key: raise HTTPException(status_code=403, detail="No API Key provided.")
    clean_key = api_key.strip('"').strip("'") # Fixes the invisible quotes bug!
    user = db.query(User).filter(User.api_key == clean_key).first()
    if not user: raise HTTPException(status_code=403, detail="Invalid API Key. Access Denied.")
    return user

def get_user_from_query(api_key: str = Query(None), db: Session = Depends(get_db)):
    if not api_key: raise HTTPException(status_code=403, detail="No API Key provided.")
    clean_key = api_key.strip('"').strip("'")
    user = db.query(User).filter(User.api_key == clean_key).first()
    if not user: raise HTTPException(status_code=403, detail="Invalid API Key.")
    return user

@app.post("/api/auth/signup")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), salt).decode('utf-8') 
    new_user = User(email=user.email, hashed_password=hashed_password, api_key=generate_api_key())
    db.add(new_user)
    db.commit()
    get_workspace(new_user.api_key) # Init workspace
    return {"message": "User created", "email": new_user.email, "api_key": new_user.api_key}

@app.post("/api/auth/login")
def login_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user: raise HTTPException(status_code=400, detail="Account doesn't exist")
    if not bcrypt.checkpw(user.password.encode('utf-8'), db_user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid Password")
    return {"message": "Auth successful", "email": db_user.email, "api_key": db_user.api_key}

@app.post("/api/v1/sdk_login")
def sdk_login_endpoint(credentials: SDKLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not bcrypt.checkpw(credentials.password.encode('utf-8'), user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"api_key": user.api_key}

# -------------------------------------------------------------------
# THE API BRIDGE
# -------------------------------------------------------------------
@app.post("/api/v1/track")
def receive_drift_data(
    payload: Dict[str, Any], 
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    run_id = payload.get("run_id")
    if not run_id: raise HTTPException(status_code=400, detail="Missing run_id")
    paths = get_paths(user.api_key)
    
    write_json_atomic(paths["runs_dir"] / f"{run_id}.json", payload)
    if paths["latest"].exists(): write_json_atomic(paths["previous"], read_json(paths["latest"]))
    write_json_atomic(paths["latest"], payload)
    paths["heartbeat"].touch()
    
    idx = read_json(paths["history_index"])
    items = idx.get("items", [])
    items.insert(0, normalize_history_item(payload))
    idx["items"] = items[:200] 
    idx["count"] = len(idx["items"])
    idx["generated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(paths["history_index"], idx)
    
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
def api_results(response: Response, user: User = Depends(get_current_user)):
    paths = get_paths(user.api_key)
    if not _is_live_connected(paths):
        no_cache_headers(response)
        return {"latest": {}, "previous": {}, "live_connected": False}
    no_cache_headers(response)
    return {"latest": read_json(paths["latest"]), "previous": read_json(paths["previous"]), "live_connected": True}

@app.get("/api/history")
def api_history(response: Response, n: int = Query(default=20, ge=1, le=200), user: User = Depends(get_current_user)):
    paths = get_paths(user.api_key)
    idx = read_json(paths["history_index"])
    items = idx.get("items")
    if not isinstance(items, list):
        items = []
        for p in paths["runs_dir"].glob("*.json"):
            run = read_json(p)
            if run: items.append(slim_run_payload(run))
        items = sort_history_items(items)
    no_cache_headers(response)
    return {"runs": items[:n], "source": "history_index"}

@app.post("/api/history/clear")
def api_history_clear(response: Response, user: User = Depends(get_current_user)):
    paths = get_paths(user.api_key)
    for p in paths["runs_dir"].iterdir():
        if p.is_file(): p.unlink(missing_ok=True)
    if paths["latest"].exists(): paths["latest"].unlink()
    if paths["previous"].exists(): paths["previous"].unlink()
    if paths["heartbeat"].exists(): paths["heartbeat"].unlink()
    if paths["report_latest"].exists(): paths["report_latest"].unlink()
    write_json_atomic(paths["history_index"], {"schema_version": 1, "count": 0, "items": []})
    no_cache_headers(response)
    return {"ok": True, "message": "Private history cleared"}

@app.get("/api/run/{run_id}")
def api_run(run_id: str, response: Response, user: User = Depends(get_current_user)):
    if not SAFE_RUN_ID_RE.match(run_id): raise HTTPException(status_code=400)
    paths = get_paths(user.api_key)
    p = paths["runs_dir"] / f"{run_id}.json"
    if not p.exists(): raise HTTPException(status_code=404, detail="Run not found")
    no_cache_headers(response)
    return read_json(p)

# -------------------------------------------------------------------
# REPORT GENERATORS & SELF TEST (Restored!)
# -------------------------------------------------------------------
@app.get("/api/report/latest", response_class=HTMLResponse)
def api_report_latest(download: int = Query(default=0), user: User = Depends(get_current_user)):
    paths = get_paths(user.api_key)
    html = read_text(paths["report_latest"])
    if not html:
        latest = read_json(paths["latest"])
        if rid := latest.get("run_id"):
            if SAFE_RUN_ID_RE.match(rid): html = read_text(paths["runs_dir"] / f"{rid}.report.html")
    if not html: raise HTTPException(status_code=404, detail="No report available")
    resp = HTMLResponse(content=html)
    if download == 1: resp.headers["Content-Disposition"] = 'attachment; filename="Report_Latest.html"'
    return resp

@app.get("/api/report/{run_id}", response_class=HTMLResponse)
def api_report_run(run_id: str, download: int = Query(default=0), user: User = Depends(get_current_user)):
    if not SAFE_RUN_ID_RE.match(run_id): raise HTTPException(status_code=400)
    paths = get_paths(user.api_key)
    html = read_text(paths["runs_dir"] / f"{run_id}.report.html")
    if not html: raise HTTPException(status_code=404, detail="Report not found")
    resp = HTMLResponse(content=html)
    if download == 1: resp.headers["Content-Disposition"] = f'attachment; filename="{run_id}.report.html"'
    return resp

@app.get("/api/selftest")
def api_selftest(response: Response, user: User = Depends(get_current_user)):
    paths = get_paths(user.api_key)
    data = read_json(paths["selftest"])
    no_cache_headers(response)
    return data if data else {"ok": False, "message": "No self-test run yet."}

@app.post("/api/selftest/run")
def api_selftest_run(response: Response, user: User = Depends(get_current_user)):
    paths = get_paths(user.api_key)
    cwd = str(BASE_DIR.parent)
    cmd = [sys.executable, "-m", "modelshift.selftest", "--out", str(paths["selftest"])]
    try:
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        no_cache_headers(response)
        return {"ok": False, "message": "Self-test failed", "stderr": (e.stderr or "")[:2000]}
    data = read_json(paths["selftest"])
    no_cache_headers(response)
    return data if data else {"ok": False, "message": "Self-test output missing."}

# -------------------------------------------------------------------
# DATASET VIEWER
# -------------------------------------------------------------------
@app.get("/dataset/{run_id}", response_class=HTMLResponse)
def view_dataset(run_id: str, user: User = Depends(get_user_from_query)):
    if not SAFE_RUN_ID_RE.match(run_id): raise HTTPException(status_code=400)
    
    paths = get_paths(user.api_key)
    run_file = paths["runs_dir"] / f"{run_id}.json"
    
    if not run_file.exists():
        return HTMLResponse("<h2 style='color:white;text-align:center;padding:50px;'>Run data not found on server.</h2>", status_code=404)
        
    data = read_json(run_file)
    records_after = data.get("dataset_sample", [])
    records_before = data.get("baseline_sample", [])
    
    if not records_after:
        return HTMLResponse("<h2 style='color:white;text-align:center;padding:50px;'>No dataset sample was attached to this run.</h2>", status_code=404)

    import pandas as pd
    import math
    
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
                    if not math.isclose(float(val_after), float(val_before), abs_tol=1e-3): is_diff = True
                except:
                    if str(val_after).strip() != str(val_before).strip(): is_diff = True

            td_class = "changed" if is_diff else ""
            display_val = f"{float(val_after):.4f}" if isinstance(val_after, (float, int, np.number)) and not pd.isna(val_after) else str(val_after)
            after_html.append(f"<td class='{td_class}'>{display_val}</td>")
        after_html.append("</tr>")
        
    after_html.append("</tbody></table>")
    before_html = df_before.round(4).to_html(classes="dataset-table", index=True) if not df_before.empty else "<h3 style='color:white'>No Baseline Data Provided.</h3>"

    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ModelShift - Dataset Inspector</title>
        <style>
            body {{ font-family: 'Segoe UI', monospace; background: #0b0c0d; color: #e9eef2; padding: 20px; }}
            h1 {{ margin-bottom: 5px; }}
            .tabs {{ display: flex; gap: 10px; margin-bottom: 15px; }}
            .tab-btn {{ background: #1d2023; color: #9aa4ad; border: 1px solid #2a2d30; padding: 10px 20px; cursor: pointer; }}
            .tab-btn.active {{ background: rgba(209,31,31,0.2); border-color: #d11f1f; color: white; font-weight: bold; }}
            .tab-content {{ display: none; overflow: auto; max-height: 80vh; border: 1px solid #2a2d30; background: #0f1112; }}
            .tab-content.active {{ display: block; }}
            .dataset-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
            .dataset-table th, .dataset-table td {{ border: 1px solid #2a2d30; padding: 6px 10px; text-align: right; white-space: nowrap; }}
            .dataset-table th {{ background: #1d2023; position: sticky; top: 0; color: white; }}
            .changed {{ background-color: rgba(209, 31, 31, 0.5) !important; color: white; font-weight: bold; }}
            .note {{ font-size: 13px; color: #9aa4ad; margin-bottom: 15px; border-left: 3px solid #d11f1f; padding-left: 10px; }}
        </style>
    </head>
    <body>
        <h1>Run ID: {run_id}</h1>
        <div class="note">Showing 15-row sample (Private to {user.email}). Cells with difference &gt; 0.001 are highlighted in Red.</div>
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('after', this)">SHEET 1: LIVE (DRIFTED)</button>
            <button class="tab-btn" onclick="showTab('before', this)">SHEET 2: BASELINE (CLEAN)</button>
        </div>
        <div id="after" class="tab-content active">{"".join(after_html)}</div>
        <div id="before" class="tab-content">{before_html}</div>
        <script>
            function showTab(tabId, btn) {{
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                btn.classList.add('active');
            }}
        </script>
    </body>
    </html>
    """)

# -------------------------------------------------------------------
# UI PAGE ROUTING
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request): return templates.TemplateResponse(request=request, name="home.html")
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request): return templates.TemplateResponse(request=request, name="index.html")
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request): return templates.TemplateResponse(request=request, name="login.html")
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request): return templates.TemplateResponse(request=request, name="signup.html")
@app.get("/api/health")
def api_health(response: Response): return {"ok": True, "service": "modelshift-saas"}