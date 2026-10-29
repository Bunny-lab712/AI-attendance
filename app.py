"""
AI Facial Recognition Attendance System  —  Pro Edition
--------------------------------------------------------
Stack : Streamlit + DeepFace (Facenet) + OpenCV + SQLite + Pandas

Install:
    pip install "streamlit>=1.50" deepface tf-keras opencv-python pandas numpy openpyxl

Run:
    streamlit run app.py

Optional admin password (protects the Admin tab):
    Windows (PowerShell):  $env:ATTENDANCE_ADMIN_PASSWORD="mypass"
    macOS / Linux:         export ATTENDANCE_ADMIN_PASSWORD="mypass"
"""

import base64
import io
import os
import re
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime, timedelta
from datetime import time as dtime
from html import escape
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="AI Attendance Pro", page_icon="📷", layout="wide")

FACES_DIR = "registered_faces"
CAPTURES_DIR = "daily_captures"
DB_FILE = "attendance.db"
MODEL_NAME = "Facenet"
ADMIN_PASSWORD = os.environ.get("ATTENDANCE_ADMIN_PASSWORD", "")

# These two are overridden from the sidebar controls further down
MATCH_THRESHOLD = 0.40   # DeepFace's default cosine threshold for Facenet
DETECTOR = "opencv"

MIN_FACE_PX = 40         # ignore tiny/false face detections smaller than this (pixels)

STATUS_COLORS = {"Present": "#10b981", "Late": "#f59e0b", "Excused": "#3b82f6"}

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)


def ensure_theme_file() -> None:
    """Write a light-theme config the first time so widgets match the custom CSS (applies on next start)."""
    cfg_dir, cfg = ".streamlit", os.path.join(".streamlit", "config.toml")
    if os.path.exists(cfg):
        return
    try:
        os.makedirs(cfg_dir, exist_ok=True)
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(
                '[theme]\nbase = "light"\nprimaryColor = "#6366f1"\n'
                'backgroundColor = "#f8fafc"\nsecondaryBackgroundColor = "#eef2ff"\n'
                'textColor = "#0f172a"\nfont = "sans serif"\n'
            )
    except OSError:
        pass


ensure_theme_file()

# =============================================================================
# 1. CUSTOM CSS
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(1100px 560px at 8% -8%, #e0e7ff 0%, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, #fce7f3 0%, transparent 55%),
        #f8fafc;
    color: #0f172a;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1220px; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp .stMarkdown p { color: #0f172a; }

/* ---------- HERO ---------- */
.hero {
    display: flex; justify-content: space-between; align-items: center; gap: 24px;
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    border-radius: 26px; padding: 30px 36px; margin-bottom: 20px;
    box-shadow: 0 20px 45px rgba(99, 102, 241, .35);
    position: relative; overflow: hidden;
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.10);
}
.hero::before { width: 260px; height: 260px; right: -60px; top: -90px; }
.hero::after  { width: 160px; height: 160px; left: 38%; bottom: -80px; }
.hero-left, .hero-right { position: relative; z-index: 1; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,.2); color: #fff; padding: 5px 14px;
    border-radius: 999px; font-size: .78rem; font-weight: 600; letter-spacing: .3px; margin-bottom: 12px;
}
.hero-title { color: #fff !important; font-size: 2rem; font-weight: 800; line-height: 1.15; }
.hero-sub { color: rgba(255,255,255,.85) !important; font-size: .95rem; margin-top: 8px; }
.hero-right { min-width: 230px; text-align: right; }
.hero-rate-label { color: rgba(255,255,255,.85) !important; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.hero-rate { color: #fff !important; font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 4px 0 10px; }
.progress { height: 10px; background: rgba(255,255,255,.25); border-radius: 999px; overflow: hidden; }
.progress > span { display: block; height: 100%; background: #fff; border-radius: 999px; }

/* ---------- STAT CARDS ---------- */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 22px; }
.stat {
    display: flex; align-items: center; gap: 14px; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 20px; padding: 18px 20px; box-shadow: 0 10px 28px rgba(15,23,42,.06);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stat:hover { transform: translateY(-4px); box-shadow: 0 16px 34px rgba(99,102,241,.18); }
.stat-icon { width: 52px; height: 52px; border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; }
.stat-label { color: #64748b !important; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; }
.stat-value { color: #0f172a !important; font-size: 1.8rem; font-weight: 800; line-height: 1.1; }

/* ---------- SECTION HEADERS ---------- */
.sec { display: flex; align-items: center; gap: 12px; margin: 4px 0 14px; }
.sec-bar { width: 6px; height: 36px; border-radius: 6px; background: linear-gradient(180deg, #6366f1, #ec4899); }
.sec-title { color: #0f172a !important; font-size: 1.25rem; font-weight: 700; line-height: 1.2; }
.sec-sub { color: #64748b !important; font-size: .85rem; }

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,.75); backdrop-filter: blur(8px);
    padding: 6px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 6px 20px rgba(15,23,42,.05);
}
.stTabs [data-baseweb="tab"] { height: 44px; border-radius: 12px; padding: 0 20px; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; color: #475569; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; box-shadow: 0 6px 16px rgba(99,102,241,.4); }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- BUTTONS ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 0; font-weight: 600; padding: .55rem 1.3rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 6px 16px rgba(99,102,241,.35); transition: all .2s ease;
}
.stButton > button p, .stDownloadButton > button p { color: #fff !important; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(99,102,241,.5); }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: .45; box-shadow: none; transform: none; }

/* ---------- INPUTS ---------- */
.stTextInput input, .stDateInput input, .stTimeInput input, .stTextInput input[type="password"] {
    border-radius: 12px !important; background: #fff !important; border: 1px solid #e2e8f0 !important;
}
.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
    border-radius: 12px !important; background: #fff !important; border: 1px solid #e2e8f0 !important;
}
.stTextInput input:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important; }

/* ---------- CAMERA ---------- */
[data-testid="stCameraInput"] {
    background: #fff; border: 2px dashed #c7d2fe; border-radius: 22px; padding: 10px;
    box-shadow: 0 10px 28px rgba(15,23,42,.06);
}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img { border-radius: 16px; }

/* ---------- MISC WIDGETS ---------- */
[data-testid="stDataFrame"] {
    border-radius: 16px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 8px 24px rgba(15,23,42,.05);
}
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stSidebar"] { background: rgba(255,255,255,.85); border-right: 1px solid #e2e8f0; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
hr { border-color: #e2e8f0; }

/* ---------- BADGES ---------- */
.badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .72rem; font-weight: 700; }
.b-present { background: #d1fae5; color: #047857; }
.b-late    { background: #fef3c7; color: #b45309; }
.b-excused { background: #dbeafe; color: #1d4ed8; }

/* ---------- RESULT CARD ---------- */
.result { display: flex; align-items: center; gap: 18px; padding: 18px 22px; margin-top: 14px; border-radius: 22px;
    background: #fff; border: 1px solid #e2e8f0; box-shadow: 0 14px 34px rgba(15,23,42,.08); }
.result.ok  { border-left: 8px solid #10b981; }
.result.late{ border-left: 8px solid #f59e0b; }
.result-imgs { display: flex; gap: 10px; }
.result-imgs div { text-align: center; font-size: .68rem; color: #64748b; font-weight: 600; }
.result-imgs img { width: 92px; height: 92px; object-fit: cover; border-radius: 16px; display: block; margin-bottom: 4px; border: 3px solid #e0e7ff; }
.result-title { color: #0f172a !important; font-size: 1.25rem; font-weight: 800; }
.result-sub { color: #64748b !important; font-size: .88rem; margin-top: 4px; }
.score { margin-left: auto; text-align: center; }
.score b { display: block; font-size: 1.8rem; font-weight: 800; color: #6366f1; line-height: 1; }
.score span { font-size: .7rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }

/* ---------- FEED ---------- */
.feed-item { display: flex; align-items: center; gap: 12px; padding: 10px 14px; margin-bottom: 10px;
    background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; box-shadow: 0 4px 14px rgba(15,23,42,.04); }
.feed-item img, .avatar { width: 44px; height: 44px; border-radius: 50%; object-fit: cover; flex-shrink: 0; }
.avatar { background: linear-gradient(135deg, #6366f1, #ec4899); color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; }
.feed-name { color: #0f172a !important; font-weight: 700; font-size: .95rem; }
.feed-time { color: #64748b !important; font-size: .78rem; }
.feed-right { margin-left: auto; }
.empty { text-align: center; color: #94a3b8 !important; padding: 28px 10px; background: #fff; border: 1px dashed #cbd5e1; border-radius: 16px; font-size: .9rem; }

/* ---------- GALLERY ---------- */
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 16px; margin: 6px 0 14px; }
.gal-item { background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 10px; text-align: center;
    box-shadow: 0 6px 18px rgba(15,23,42,.05); transition: transform .2s ease; }
.gal-item:hover { transform: translateY(-4px) scale(1.02); box-shadow: 0 14px 28px rgba(99,102,241,.2); }
.gal-item img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 12px; }
.gal-title { color: #0f172a !important; font-weight: 700; font-size: .88rem; margin-top: 8px; }
.gal-sub { color: #64748b !important; font-size: .74rem; }

/* ---------- SIDEBAR BRAND ---------- */
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.brand-logo { width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
    background: linear-gradient(135deg, #6366f1, #ec4899); box-shadow: 0 8px 18px rgba(99,102,241,.4); }
.brand-name { color: #0f172a !important; font-weight: 800; font-size: 1.1rem; line-height: 1.1; }
.brand-sub { color: #64748b !important; font-size: .75rem; }
.side-chip { display: flex; justify-content: space-between; background: #eef2ff; border-radius: 12px; padding: 9px 14px; margin-bottom: 8px; }
.side-chip span { color: #475569 !important; font-size: .82rem; font-weight: 600; }
.side-chip b { color: #4f46e5 !important; }

/* ---------- LIVE MODE ---------- */
.live-pill { display: inline-flex; align-items: center; gap: 8px; background: #fee2e2; padding: 5px 14px;
    border-radius: 999px; font-weight: 700; font-size: .78rem; color: #b91c1c !important; }
.live-dot { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; animation: pulse 1.2s infinite; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,.6); }
    70%  { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
[data-testid="stImage"] img { border-radius: 16px; }
.live-status { margin-top: 10px; color: #334155 !important; font-size: .9rem; }

.footer { text-align: center; color: #94a3b8 !important; font-size: .8rem; margin: 30px 0 6px; }

@media (max-width: 800px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .hero { flex-direction: column; align-items: flex-start; }
    .hero-right { text-align: left; width: 100%; }
    .hero-title { font-size: 1.5rem; }
    .result { flex-direction: column; text-align: center; }
    .score { margin-left: 0; }
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# =============================================================================
# 2. DATABASE LAYER
# =============================================================================
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Students (
                ID                TEXT PRIMARY KEY,
                Name              TEXT NOT NULL,
                Registration_Date TEXT NOT NULL,
                Profile_Pic_Path  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Attendance (
                Log_ID           INTEGER PRIMARY KEY AUTOINCREMENT,
                Student_ID       TEXT NOT NULL,
                Date             TEXT NOT NULL,
                Time             TEXT NOT NULL,
                Capture_Pic_Path TEXT,
                Status           TEXT NOT NULL DEFAULT 'Present',
                UNIQUE (Student_ID, Date),
                FOREIGN KEY (Student_ID) REFERENCES Students(ID)
            )
            """
        )
        conn.commit()


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_students() -> pd.DataFrame:
    return query_df("SELECT ID, Name, Registration_Date, Profile_Pic_Path FROM Students ORDER BY Name")


def student_exists(student_id: str) -> bool:
    with closing(get_conn()) as conn:
        return conn.execute("SELECT 1 FROM Students WHERE ID = ?", (student_id,)).fetchone() is not None


def already_marked(student_id: str, date_str: str) -> bool:
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT 1 FROM Attendance WHERE Student_ID = ? AND Date = ?", (student_id, date_str)
        ).fetchone()
    return row is not None


def insert_student(student_id: str, name: str, profile_path: str) -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "INSERT INTO Students (ID, Name, Registration_Date, Profile_Pic_Path) VALUES (?, ?, ?, ?)",
            (student_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), profile_path),
        )
        conn.commit()


def insert_attendance(student_id, date_str, time_str, capture_path, status="Present") -> None:
    with closing(get_conn()) as conn:
        conn.execute(
            "INSERT INTO Attendance (Student_ID, Date, Time, Capture_Pic_Path, Status) VALUES (?, ?, ?, ?, ?)",
            (student_id, date_str, time_str, capture_path, status),
        )
        conn.commit()


def delete_student(student_id: str) -> None:
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT Profile_Pic_Path FROM Students WHERE ID = ?", (student_id,)).fetchone()
        conn.execute("DELETE FROM Attendance WHERE Student_ID = ?", (student_id,))
        conn.execute("DELETE FROM Students WHERE ID = ?", (student_id,))
        conn.commit()
    if row and row[0] and os.path.exists(row[0]):
        os.remove(row[0])


def today_stats() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    total = len(get_students())
    df = query_df("SELECT Status, COUNT(*) AS c FROM Attendance WHERE Date = ? GROUP BY Status", (today,))
    counts = dict(zip(df["Status"], df["c"])) if not df.empty else {}
    present = int(counts.get("Present", 0) + counts.get("Late", 0))
    late = int(counts.get("Late", 0))
    recorded = int(sum(counts.values()))
    absent = max(total - recorded, 0)
    rate = round(present / total * 100) if total else 0
    return {"total": total, "present": present, "late": late, "absent": absent, "rate": rate}


init_db()


# =============================================================================
# 3. IMAGE + AI HELPERS
# =============================================================================
def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")


def decode_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the captured image.")
    return img


def largest_face(faces: list) -> dict:
    return max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])


def extract_embeddings(img: np.ndarray, enforce: bool = True, spoof_check: bool = False) -> list:
    """
    Detect faces + compute Facenet embeddings.
    Raises ValueError if enforce=True and no face is found (or a spoof is detected when spoof_check=True).
    """
    kwargs = dict(
        img_path=img, model_name=MODEL_NAME, detector_backend=DETECTOR, enforce_detection=enforce
    )
    if spoof_check:
        kwargs["anti_spoofing"] = True
    try:
        return DeepFace.represent(**kwargs)
    except TypeError:
        if not spoof_check:
            raise
        st.warning("Anti-spoofing needs a newer DeepFace (pip install -U deepface torch). Running without it.")
        kwargs.pop("anti_spoofing")
        return DeepFace.represent(**kwargs)


@st.cache_data(show_spinner=False)
def profile_embedding(path: str, mtime: float, detector: str):
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        faces = DeepFace.represent(
            img_path=img, model_name=MODEL_NAME, detector_backend=detector, enforce_detection=False
        )
        return largest_face(faces)["embedding"]
    except Exception:
        return None


def cosine_distance(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_best_match(embedding, students: pd.DataFrame):
    """Return (student_row, distance) for the closest registered face within threshold, else (None, best_distance)."""
    best_row, best_dist = None, None
    for _, row in students.iterrows():
        path = row["Profile_Pic_Path"]
        if not os.path.exists(path):
            continue
        ref = profile_embedding(path, os.path.getmtime(path), DETECTOR)
        if ref is None:
            continue
        dist = cosine_distance(embedding, ref)
        if best_dist is None or dist < best_dist:
            best_row, best_dist = row, dist
    if best_row is not None and best_dist <= MATCH_THRESHOLD:
        return best_row, best_dist
    return None, best_dist


def save_capture(img: np.ndarray, student_id: str, now: datetime) -> str:
    """Save snapshot to daily_captures/YYYY-MM-DD/<ID>_<HHMMSS>.jpg"""
    day_dir = os.path.join(CAPTURES_DIR, now.strftime("%Y-%m-%d"))
    os.makedirs(day_dir, exist_ok=True)
    path = os.path.join(day_dir, f"{safe_name(student_id)}_{now.strftime('%H%M%S')}.jpg")
    cv2.imwrite(path, img)
    return path


# ---------- Multi-face recognition (live video / video file / class photo) ----------
def resize_max(img: np.ndarray, max_w: int = 960) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)))


def load_known(students: pd.DataFrame) -> list:
    """Return [(student_dict, embedding), ...] for every registered student with a usable photo."""
    known = []
    for _, row in students.iterrows():
        path = row["Profile_Pic_Path"]
        if not os.path.exists(path):
            continue
        emb = profile_embedding(path, os.path.getmtime(path), DETECTOR)
        if emb is not None:
            known.append((row.to_dict(), emb))
    return known


def recognize_frame(frame: np.ndarray, known: list) -> list:
    """Find every face in a frame and match each against the known students."""
    try:
        faces = DeepFace.represent(
            img_path=frame, model_name=MODEL_NAME, detector_backend=DETECTOR, enforce_detection=True
        )
    except ValueError:
        return []   # no face in this frame
    results = []
    for f in faces:
        a = f["facial_area"]
        box = (int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"]))
        if box[2] < MIN_FACE_PX or box[3] < MIN_FACE_PX:
            continue
        best, best_d = None, None
        for stu, emb in known:
            d = cosine_distance(f["embedding"], emb)
            if best_d is None or d < best_d:
                best, best_d = stu, d
        matched = best is not None and best_d <= MATCH_THRESHOLD
        results.append({"box": box, "student": best if matched else None, "distance": best_d})
    return results


def annotate_frame(frame: np.ndarray, results: list, marked_ids: set) -> np.ndarray:
    """Draw boxes + name labels. Green = marked, indigo = recognised, red = unknown."""
    out = frame.copy()
    for r in results:
        x, y, w, h = r["box"]
        stu = r["student"]
        if stu is None:
            color, label = (68, 68, 239), "Unknown"
        else:
            sid, name = str(stu["ID"]), str(stu["Name"])
            label = name if name.isascii() else f"ID {sid}"
            color = (129, 185, 16) if sid in marked_ids else (241, 102, 99)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y - th - 12, 0)
        cv2.rectangle(out, (x, top), (x + tw + 10, top + th + 12), color, -1)
        cv2.putText(out, label, (x + 5, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def mark_present(student: dict, img: np.ndarray, now: datetime, face_box=None):
    """
    Mark a recognised student present (saves a cropped face snapshot for auditing).
    Returns ("marked", status) or ("already", None).
    """
    sid = str(student["ID"])
    date_str = now.strftime("%Y-%m-%d")
    if already_marked(sid, date_str):
        return "already", None

    snap = img
    if face_box is not None:
        x, y, w, h = face_box
        m = int(0.4 * max(w, h))
        H, W = img.shape[:2]
        crop = img[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
        if crop.size:
            snap = crop

    status = "Late" if (late_on and now.time() > late_after) else "Present"
    path = save_capture(snap, sid, now)
    try:
        insert_attendance(sid, date_str, now.strftime("%H:%M:%S"), path, status)
    except sqlite3.IntegrityError:
        if os.path.exists(path):
            os.remove(path)
        return "already", None
    return "marked", status


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def thumb_b64(path: str, mtime: float, size: int = 240) -> str:
    """Square-cropped base64 thumbnail for use inside custom HTML cards."""
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    img = cv2.resize(img[y:y + s, x:x + s], (size, size))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def thumb(path, size: int = 240) -> str:
    if path and isinstance(path, str) and os.path.exists(path):
        return thumb_b64(path, os.path.getmtime(path), size)
    return ""


# =============================================================================
# 4. HTML COMPONENT HELPERS
# =============================================================================
def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="sec"><span class="sec-bar"></span><div><div class="sec-title">{escape(title)}</div>'
        f'<div class="sec-sub">{escape(subtitle)}</div></div></div>',
        unsafe_allow_html=True,
    )


def badge(status: str) -> str:
    cls = {"Present": "b-present", "Late": "b-late", "Excused": "b-excused"}.get(status, "b-present")
    return f'<span class="badge {cls}">{escape(str(status))}</span>'


def hero_html(stats: dict) -> str:
    now = datetime.now()
    greeting = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 17 else "Good evening"
    return (
        '<div class="hero"><div class="hero-left">'
        '<div class="hero-badge">🤖 Powered by DeepFace · Facenet</div>'
        f'<div class="hero-title">{greeting}! Ready to take attendance?</div>'
        f'<div class="hero-sub">{now.strftime("%A, %d %B %Y")} · {now.strftime("%I:%M %p")}</div></div>'
        '<div class="hero-right"><div class="hero-rate-label">Today\'s attendance</div>'
        f'<div class="hero-rate">{stats["rate"]}%</div>'
        f'<div class="progress"><span style="width:{stats["rate"]}%"></span></div></div></div>'
    )


def stats_html(stats: dict) -> str:
    cards = [
        ("🎓", "Registered", stats["total"], "linear-gradient(135deg,#e0e7ff,#c7d2fe)"),
        ("✅", "Present Today", stats["present"], "linear-gradient(135deg,#d1fae5,#a7f3d0)"),
        ("⏰", "Late Arrivals", stats["late"], "linear-gradient(135deg,#fef3c7,#fde68a)"),
        ("🚫", "Absent", stats["absent"], "linear-gradient(135deg,#fee2e2,#fecaca)"),
    ]
    inner = "".join(
        f'<div class="stat"><div class="stat-icon" style="background:{bg}">{icon}</div>'
        f'<div><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div></div>'
        for icon, label, value, bg in cards
    )
    return f'<div class="stats">{inner}</div>'


def sidebar_stats_html(stats: dict) -> str:
    n_caps = sum(len(files) for _, _, files in os.walk(CAPTURES_DIR))
    rows = [
        ("Registered students", stats["total"]),
        ("Present today", stats["present"]),
        ("Stored check-in photos", n_caps),
    ]
    return "".join(f'<div class="side-chip"><span>{k}</span><b>{v}</b></div>' for k, v in rows)


def result_html(name, sid, time_str, status, score, profile_path, capture_path) -> str:
    cls = "late" if status == "Late" else "ok"
    prof, cap = thumb(profile_path, 200), thumb(capture_path, 200)
    imgs = ""
    if prof:
        imgs += f'<div><img src="{prof}"/>Registered</div>'
    if cap:
        imgs += f'<div><img src="{cap}"/>Live capture</div>'
    return (
        f'<div class="result {cls}"><div class="result-imgs">{imgs}</div><div>'
        f'<div class="result-title">Welcome, {escape(str(name))}! 🎉</div>'
        f'<div class="result-sub">ID {escape(str(sid))} · {escape(time_str)} · {badge(status)}</div></div>'
        f'<div class="score"><b>{score}%</b><span>Match score</span></div></div>'
    )


def feed_html(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="empty">No check-ins yet today.<br>Be the first! 🚀</div>'
    items = []
    for r in df.itertuples():
        src = thumb(r.Profile_Pic_Path, 96)
        name = r.Name if r.Name else r.Student_ID
        avatar = f'<img src="{src}"/>' if src else f'<div class="avatar">{escape(str(name)[:1].upper())}</div>'
        items.append(
            f'<div class="feed-item">{avatar}<div><div class="feed-name">{escape(str(name))}</div>'
            f'<div class="feed-time">🕒 {escape(str(r.Time))}</div></div>'
            f'<div class="feed-right">{badge(r.Status)}</div></div>'
        )
    return "".join(items)


def render_live_session_log(slot, entries: list) -> None:
    with slot.container():
        if not entries:
            st.markdown('<div class="empty">Waiting for faces…</div>', unsafe_allow_html=True)
            return

        for entry in entries:
            image_col, details_col = st.columns([1, 4], vertical_alignment="center")
            profile_path = entry["Profile_Pic_Path"]
            if profile_path and os.path.isfile(profile_path):
                image_col.image(profile_path, width=64)
            else:
                image_col.caption("No profile photo")

            details_col.markdown(
                f"**{escape(str(entry['Name']))}**  \n"
                f"🕒 {escape(str(entry['Time']))} · {badge(entry['Status'])}",
                unsafe_allow_html=True,
            )


def gallery_html(items: list) -> str:
    """items: list of (image_path, title, subtitle)"""
    cells = []
    for path, title, sub in items:
        src = thumb(path, 240)
        img = f'<img src="{src}"/>' if src else '<div class="avatar" style="width:100%;height:120px;border-radius:12px">?</div>'
        cells.append(
            f'<div class="gal-item">{img}<div class="gal-title">{escape(str(title))}</div>'
            f'<div class="gal-sub">{escape(str(sub))}</div></div>'
        )
    return f'<div class="gallery">{"".join(cells)}</div>'


# =============================================================================
# 5. SIDEBAR  (settings + quick stats)
# =============================================================================
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">📷</div><div><div class="brand-name">AI Attendance</div>'
        '<div class="brand-sub">Pro Edition</div></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**⚙️ Settings**")
    MATCH_THRESHOLD = st.slider(
        "Match tolerance", 0.20, 0.60, 0.40, 0.01,
        help="Cosine distance threshold. Lower = stricter matching (fewer false positives).",
    )
    DETECTOR = st.selectbox(
        "Face detector", ["opencv", "mtcnn", "retinaface"], index=0,
        help="opencv is fastest. mtcnn / retinaface are more accurate but slower (first use downloads weights).",
    )
    late_on = st.toggle("Track late arrivals", value=True)
    late_after = st.time_input("Late after", value=dtime(9, 30), disabled=not late_on)
    anti_spoof = st.toggle(
        "Anti-spoofing (Daily Scan)", value=False,
        help="Rejects photos/screens held up to the camera. Needs a recent DeepFace and PyTorch (pip install torch).",
    )
    st.divider()
    st.markdown("**📊 Quick stats**")
    sidebar_stats_ph = st.empty()

# =============================================================================
# 6. HEADER PLACEHOLDERS (filled at the end so they always show fresh numbers)
# =============================================================================
hero_ph = st.empty()
stats_ph = st.empty()

tab_scan, tab_live, tab_reg, tab_admin, tab_analytics = st.tabs(
    ["📸 Daily Scan", "🎥 Live Classroom", "📝 Registration", "🛠️ Admin Panel", "📈 Analytics"]
)

# -----------------------------------------------------------------------------
# TAB 1: DAILY SCAN
# -----------------------------------------------------------------------------
with tab_scan:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        section("Mark Daily Attendance", "Look at the camera and press “Take Photo”")
        students = get_students()

        if students.empty:
            st.warning("No students registered yet. Please use the Registration tab first.")
        else:
            shot = st.camera_input("Scan Face", key="scan_cam", label_visibility="collapsed")

            if shot is not None:
                try:
                    img = decode_image(shot)

                    with st.spinner("🧠 AI is analysing the face and verifying identity..."):
                        faces = extract_embeddings(img, enforce=True, spoof_check=anti_spoof)
                        face = largest_face(faces)
                        student, distance = find_best_match(face["embedding"], students)

                    if len(faces) > 1:
                        st.warning("Multiple faces detected — using the largest one.")

                    if student is None:
                        st.error("❌ Face not recognized. Please register first or try again.")
                    else:
                        now = datetime.now()
                        date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                        sid, sname = str(student["ID"]), student["Name"]
                        status = "Late" if (late_on and now.time() > late_after) else "Present"
                        score = int(max(0, min(100, (1 - distance / (2 * MATCH_THRESHOLD)) * 100)))

                        if already_marked(sid, date_str):
                            st.warning(f"⚠️ Attendance already recorded for **{sname}** today.")
                        else:
                            capture_path = save_capture(img, sid, now)
                            try:
                                insert_attendance(sid, date_str, time_str, capture_path, status)
                                st.markdown(
                                    result_html(sname, sid, time_str, status, score,
                                                student["Profile_Pic_Path"], capture_path),
                                    unsafe_allow_html=True,
                                )
                                st.toast(f"{sname} checked in ({status})", icon="✅")
                            except sqlite3.IntegrityError:
                                if os.path.exists(capture_path):
                                    os.remove(capture_path)
                                st.warning(f"⚠️ Attendance already recorded for **{sname}** today.")

                except ValueError as e:
                    if "spoof" in str(e).lower():
                        st.error("🛡️ Spoof attempt detected — please show your real face (no photos or screens).")
                    else:
                        st.error("😕 No face detected in the image. Please face the camera and try again.")
                except Exception as e:
                    st.error(f"Unexpected error while scanning: {e}")

    with col_r:
        section("Live Check-in Feed", "Today's latest arrivals")
        feed_df = query_df(
            """
            SELECT a.Time, a.Status, a.Student_ID, s.Name, s.Profile_Pic_Path
            FROM Attendance a LEFT JOIN Students s ON s.ID = a.Student_ID
            WHERE a.Date = ? ORDER BY a.Time DESC LIMIT 6
            """,
            (datetime.now().strftime("%Y-%m-%d"),),
        )
        st.markdown(feed_html(feed_df), unsafe_allow_html=True)
        st.info("💡 Tip: good lighting and a front-facing pose give the best match score.")

# -----------------------------------------------------------------------------
# TAB 1b: LIVE CLASSROOM  (multi-face: webcam stream / video file / class photo)
# -----------------------------------------------------------------------------
with tab_live:
    section("Live Classroom Mode", "Identify many students at once — from a webcam, a recorded video or a class photo")
    live_students = get_students()

    if live_students.empty:
        st.warning("No students registered yet. Please use the Registration tab first.")
    else:
        mode = st.radio(
            "Mode", ["🔴 Live webcam", "🎞️ Upload video", "🖼️ Class photo"],
            horizontal=True, label_visibility="collapsed", key="live_mode",
        )

        # ============================ MODE 1: LIVE WEBCAM ============================
        # ============================ MODE 1: LIVE WEBCAM (CLOUD-READY) ============================
        if mode.startswith("🔴"):
            st.info("Live cloud scanner activated. Please allow camera permissions in your browser.")
            
            with st.spinner("Loading registered faces..."):
                known = load_known(live_students)

            if not known:
                st.error("No usable registered face photos were found.")
            else:
                # The callback that processes the video feed in real-time
                def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
                    # Convert browser frame to OpenCV format
                    img = frame.to_ndarray(format="bgr24")
                    img = resize_max(img, 800)
                    
                    # Run DeepFace recognition
                    results = recognize_frame(img, known)
                    
                    marked_ids = set()
                    for r in results:
                        stu = r["student"]
                        if stu is not None:
                            sid = str(stu["ID"])
                            # Mark them present in the database if not already marked
                            if not already_marked(sid, datetime.now().strftime("%Y-%m-%d")):
                                mark_present(stu, img, datetime.now(), r["box"])
                            marked_ids.add(sid)
                            
                    # Draw boxes and names on the video feed
                    annotated_img = annotate_frame(img, results, marked_ids)
                    
                    # Return the processed frame to the browser
                    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

                # Start the cloud-compatible video stream
                webrtc_streamer(
                    key="cloud_live_attendance",
                    video_frame_callback=process_frame,
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }
                )
            if start:
                with st.spinner("Loading registered faces (first run can take a moment)..."):
                    known = load_known(live_students)

                if not known:
                    st.error("No usable registered face photos were found.")
                else:
                    cap = cv2.VideoCapture(int(cam_index))
                    if not cap.isOpened():
                        st.error("Could not open the camera. Check the camera index/permissions or close other apps using it.")
                    else:
                        marked_ids, counts, session_log = set(), {}, []
                        last_results, frame_no, t0 = [], 0, time.time()
                        try:
                            while time.time() - t0 < max_seconds:
                                ok, frame = cap.read()
                                if not ok:
                                    st.error("The camera stream ended unexpectedly.")
                                    break
                                frame = resize_max(frame, 800)
                                frame_no += 1

                                if (frame_no - 1) % process_every == 0:
                                    last_results = recognize_frame(frame, known)
                                    seen = set()
                                    for r in last_results:
                                        stu = r["student"]
                                        if stu is None:
                                            continue
                                        sid = str(stu["ID"])
                                        seen.add(sid)
                                        if sid in marked_ids:
                                            continue
                                        counts[sid] = counts.get(sid, 0) + 1
                                        if counts[sid] >= confirm_frames:
                                            now = datetime.now()
                                            state, status = mark_present(stu, frame, now, r["box"])
                                            marked_ids.add(sid)
                                            if state == "marked":
                                                session_log.insert(0, {
                                                    "Time": now.strftime("%H:%M:%S"), "Status": status,
                                                    "Student_ID": sid, "Name": stu["Name"],
                                                    "Profile_Pic_Path": stu["Profile_Pic_Path"],
                                                })
                                                render_live_session_log(log_slot, session_log)
                                                st.toast(f"{stu['Name']} checked in ({status})", icon="✅")
                                    for sid in list(counts):      # require *consecutive* sightings
                                        if sid not in seen:
                                            counts[sid] = 0

                                    left = int(max(max_seconds - (time.time() - t0), 0))
                                    status_slot.markdown(
                                        '<div class="live-status"><span class="live-pill"><span class="live-dot"></span>LIVE</span>'
                                        f' &nbsp; Faces in view: <b>{len(last_results)}</b> · '
                                        f'Marked this session: <b>{len(session_log)}</b> · Time left: <b>{left}s</b></div>',
                                        unsafe_allow_html=True,
                                    )

                                video_slot.image(
                                    annotate_frame(frame, last_results, marked_ids), channels="BGR", width=800
                                )
                        finally:
                            cap.release()
                        st.success(f"✅ Live session finished — {len(session_log)} new student(s) marked.")

        # ============================ MODE 2: UPLOAD VIDEO ============================
        elif mode.startswith("🎞️"):
            vid = st.file_uploader("Upload a classroom video", type=["mp4", "avi", "mov", "mkv"])
            c1, c2, c3 = st.columns(3)
            every_sec = c1.slider("Analyse one frame every (sec)", 0.5, 5.0, 1.0, 0.5)
            max_frames = c2.slider("Max frames to analyse", 10, 300, 120, 10)
            confirm_v = c3.slider("Confirm in N frames", 1, 3, 2,
                                  help="A student must appear in at least N analysed frames to be marked.")

            if vid is not None and st.button("🔍 Analyse video & mark attendance"):
                with st.spinner("Loading registered faces..."):
                    known = load_known(live_students)

                if not known:
                    st.error("No usable registered face photos were found.")
                else:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid.name)[1] or ".mp4")
                    tmp.write(vid.getvalue())
                    tmp.close()
                    cap = cv2.VideoCapture(tmp.name)

                    found, counts, samples = {}, {}, []
                    unknown_hits, analysed, frame_i = 0, 0, 0
                    try:
                        if not cap.isOpened():
                            st.error("Could not read this video file.")
                        else:
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            step = max(int(fps * every_sec), 1)
                            progress = st.progress(0.0, text="Analysing video...")

                            while analysed < max_frames:
                                if not cap.grab():
                                    break
                                if frame_i % step == 0:
                                    ok, frame = cap.retrieve()
                                    if ok:
                                        frame = resize_max(frame, 960)
                                        results = recognize_frame(frame, known)
                                        analysed += 1
                                        secs = int(frame_i / fps)
                                        stamp = f"{secs // 60:02d}:{secs % 60:02d}"
                                        for r in results:
                                            stu = r["student"]
                                            if stu is None:
                                                unknown_hits += 1
                                                continue
                                            sid = str(stu["ID"])
                                            if sid in found:
                                                continue
                                            counts[sid] = counts.get(sid, 0) + 1
                                            if counts[sid] >= confirm_v:
                                                state, status = mark_present(stu, frame, datetime.now(), r["box"])
                                                found[sid] = {
                                                    "Student_ID": sid, "Name": stu["Name"], "First seen": stamp,
                                                    "Result": f"✅ Marked ({status})" if state == "marked"
                                                    else "ℹ️ Already marked earlier",
                                                }
                                                if state == "marked" and len(samples) < 6:
                                                    ann = annotate_frame(frame, results, set(found))
                                                    samples.append((cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                                                    f"{stu['Name']} @ {stamp}"))
                                        pct = min(frame_i / total, 1.0) if total else 0.0
                                        progress.progress(pct, text=f"Analysed {analysed} frame(s) · {len(found)} student(s) identified")
                                frame_i += 1
                            progress.progress(1.0, text="Done")
                    finally:
                        cap.release()
                        if os.path.exists(tmp.name):
                            os.remove(tmp.name)

                    newly = sum(1 for v in found.values() if v["Result"].startswith("✅"))
                    st.success(f"✅ Analysed {analysed} frame(s). Identified {len(found)} student(s); {newly} newly marked.")

                    if found:
                        st.markdown("##### 🎓 Identified students")
                        st.dataframe(pd.DataFrame(found.values()), width="stretch", hide_index=True)
                    today_s = datetime.now().strftime("%Y-%m-%d")
                    missing = [k[0] for k in known if str(k[0]["ID"]) not in found and not already_marked(str(k[0]["ID"]), today_s)]
                    if missing:
                        st.markdown("##### 🚫 Not seen in this video")
                        st.dataframe(pd.DataFrame([{"Student_ID": m["ID"], "Name": m["Name"]} for m in missing]),
                                     width="stretch", hide_index=True)
                    if unknown_hits:
                        st.caption(f"{unknown_hits} unrecognised face detection(s) across analysed frames.")
                    if samples:
                        st.markdown("##### 🖼️ Detection samples")
                        cols = st.columns(3)
                        for i, (im, cap_txt) in enumerate(samples):
                            cols[i % 3].image(im, caption=cap_txt, width=260)

        # ============================ MODE 3: CLASS PHOTO ============================
        else:
            src = st.radio("Photo source", ["Upload photo", "Take photo now"], horizontal=True, key="photo_src")
            if src == "Upload photo":
                photo = st.file_uploader("Upload a class / group photo", type=["jpg", "jpeg", "png"], key="group_upload")
            else:
                photo = st.camera_input("Group photo", key="group_cam")

            if photo is not None and st.button("🔍 Identify everyone & mark attendance"):
                try:
                    img = resize_max(decode_image(photo), 1600)
                    with st.spinner("🧠 Detecting and identifying every face..."):
                        known = load_known(live_students)
                        results = recognize_frame(img, known) if known else []

                    if not known:
                        st.error("No usable registered face photos were found.")
                    elif not results:
                        st.error("😕 No faces detected in this photo. Try a clearer or closer picture.")
                    else:
                        # keep the best-matching face per student (a person can't be marked twice)
                        best_per = {}
                        for r in results:
                            if r["student"] is None:
                                continue
                            sid = str(r["student"]["ID"])
                            if sid not in best_per or r["distance"] < best_per[sid]["distance"]:
                                best_per[sid] = r

                        rows, now, marked_ids = [], datetime.now(), set()
                        for sid, r in best_per.items():
                            state, status = mark_present(r["student"], img, now, r["box"])
                            marked_ids.add(sid)
                            rows.append({
                                "Student_ID": sid, "Name": r["student"]["Name"],
                                "Match score %": int(max(0, min(100, (1 - r["distance"] / (2 * MATCH_THRESHOLD)) * 100))),
                                "Result": f"✅ Marked ({status})" if state == "marked" else "ℹ️ Already marked earlier",
                            })
                        unknown_n = sum(1 for r in results if r["student"] is None)
                        newly = sum(1 for x in rows if x["Result"].startswith("✅"))

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Faces detected", len(results))
                        m2.metric("Students identified", len(rows))
                        m3.metric("Newly marked", newly)

                        st.image(cv2.cvtColor(annotate_frame(img, results, marked_ids), cv2.COLOR_BGR2RGB),
                                 caption="Green = marked · Red = unknown", width=820)
                        if rows:
                            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                        if unknown_n:
                            st.warning(f"{unknown_n} face(s) could not be matched to a registered student.")
                        st.toast(f"{newly} student(s) marked from the photo", icon="🖼️")
                except ValueError:
                    st.error("Could not read this image. Please try another file.")
                except Exception as e:
                    st.error(f"Unexpected error while analysing the photo: {e}")

# -----------------------------------------------------------------------------
# TAB 2: REGISTRATION
# -----------------------------------------------------------------------------
with tab_reg:
    st.session_state.setdefault("reg_nonce", 0)
    n = st.session_state["reg_nonce"]

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        section("Register a New Student", "Enter details and capture a clear front-facing photo")

        flash = st.session_state.pop("reg_flash", None)
        if flash:
            st.success(flash)

        c1, c2 = st.columns(2)
        with c1:
            new_id = st.text_input("Student ID", placeholder="e.g. 101", key=f"reg_id_{n}").strip()
        with c2:
            new_name = st.text_input("Student Name", placeholder="e.g. Priya Sharma", key=f"reg_name_{n}").strip()

        reg_shot = st.camera_input("Take Registration Photo", key=f"reg_cam_{n}")

        if st.button("💾 Save Registration"):
            if not (new_id and new_name and reg_shot):
                st.error("Please provide ID, Name, and capture a photo.")
            elif student_exists(new_id):
                st.error(f"Student ID **{new_id}** is already registered.")
            else:
                try:
                    img = decode_image(reg_shot)

                    with st.spinner("🧠 Validating face in the photo..."):
                        faces = extract_embeddings(img, enforce=True)
                        existing = get_students()
                        dup = None
                        if len(faces) == 1 and not existing.empty:
                            dup, _ = find_best_match(faces[0]["embedding"], existing)

                    if len(faces) > 1:
                        st.error("Multiple faces detected. Only one person should be in the frame.")
                    elif dup is not None:
                        st.error(f"This face is already registered as **{dup['Name']}** (ID: {dup['ID']}).")
                    else:
                        file_path = os.path.join(FACES_DIR, f"{safe_name(new_id)}_{safe_name(new_name)}.jpg")
                        cv2.imwrite(file_path, img)
                        try:
                            insert_student(new_id, new_name, file_path)
                            st.session_state["reg_flash"] = (
                                f"🎉 Registered **{new_name}** (ID: {new_id}). You can now scan attendance."
                            )
                            st.session_state["reg_nonce"] += 1   # clears the form + camera
                            st.rerun()
                        except sqlite3.IntegrityError:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.error(f"Student ID **{new_id}** is already registered.")
                except ValueError:
                    st.error("😕 No face detected in the registration photo. Please retake it.")
                except Exception as e:
                    st.error(f"Unexpected error during registration: {e}")

    with col_r:
        section("Photo Checklist", "For the most reliable recognition")
        st.markdown(
            '<div class="feed-item"><div class="avatar">1</div><div class="feed-name">Face the camera directly</div></div>'
            '<div class="feed-item"><div class="avatar">2</div><div class="feed-name">Use bright, even lighting</div></div>'
            '<div class="feed-item"><div class="avatar">3</div><div class="feed-name">Remove masks &amp; sunglasses</div></div>'
            '<div class="feed-item"><div class="avatar">4</div><div class="feed-name">Only one person in the frame</div></div>',
            unsafe_allow_html=True,
        )
        section("Recently Registered", "Latest 4 students")
        recent = query_df(
            "SELECT ID, Name, Profile_Pic_Path FROM Students ORDER BY Registration_Date DESC LIMIT 4"
        )
        if recent.empty:
            st.markdown('<div class="empty">No students yet.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                gallery_html([(r.Profile_Pic_Path, r.Name, f"ID {r.ID}") for r in recent.itertuples()]),
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# TAB 3: ADMIN PANEL
# -----------------------------------------------------------------------------
with tab_admin:
    section("Admin Panel", "Review records, audit photos, manage students and export reports")

    admin_ok = True
    if ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
        admin_ok = False
        pw = st.text_input("🔒 Admin password", type="password", key="admin_pw")
        if st.button("Unlock"):
            if pw == ADMIN_PASSWORD:
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    if admin_ok:
        top1, top2, top3 = st.columns([1.2, 1.2, 1])
        with top1:
            selected_date = st.date_input("📅 Filter by date", value=datetime.now().date(), key="admin_date")
        with top2:
            search = st.text_input("🔎 Search name or ID", key="admin_search")
        with top3:
            st.write("")
            show_all = st.checkbox("Show all dates", value=False)
        status_filter = st.multiselect("Status", list(STATUS_COLORS), default=list(STATUS_COLORS))

        sel_date = selected_date.strftime("%Y-%m-%d")
        base_sql = """
            SELECT a.Log_ID, a.Student_ID, s.Name, a.Date, a.Time, a.Status, a.Capture_Pic_Path
            FROM Attendance a LEFT JOIN Students s ON s.ID = a.Student_ID
        """
        if show_all:
            df_att = query_df(base_sql + " ORDER BY a.Date DESC, a.Time DESC")
            report_label = "all_dates"
        else:
            df_att = query_df(base_sql + " WHERE a.Date = ? ORDER BY a.Time DESC", (sel_date,))
            report_label = sel_date

        if not df_att.empty:
            df_att = df_att[df_att["Status"].isin(status_filter)]
            if search.strip():
                q = search.strip().lower()
                mask = df_att["Name"].fillna("").str.lower().str.contains(q, regex=False) | \
                    df_att["Student_ID"].astype(str).str.lower().str.contains(q, regex=False)
                df_att = df_att[mask]

        sub_att, sub_abs, sub_gal, sub_stu = st.tabs(
            ["📋 Attendance", "🚫 Absentees & Manual Entry", "🖼️ Capture Gallery", "🎓 Students"]
        )

        # ---- Attendance table + export + audit ----
        with sub_att:
            st.dataframe(df_att, width="stretch")
            st.caption(f"{len(df_att)} record(s) shown for **{'all dates' if show_all else sel_date}**.")

            d1, d2, _ = st.columns([1, 1, 2])
            with d1:
                st.download_button(
                    "⬇️ Download CSV", data=df_att.to_csv(index=False).encode("utf-8"),
                    file_name=f"attendance_{report_label}.csv", mime="text/csv", disabled=df_att.empty,
                )
            with d2:
                try:
                    st.download_button(
                        "⬇️ Download Excel", data=to_excel_bytes(df_att),
                        file_name=f"attendance_{report_label}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        disabled=df_att.empty,
                    )
                except ImportError:
                    st.caption("Install `openpyxl` to enable Excel export.")

            st.markdown("##### 🔍 Audit a check-in photo")
            if df_att.empty:
                st.info("No attendance records for this selection.")
            else:
                options = {
                    f"{r.Name or r.Student_ID} — {r.Date} {r.Time} ({r.Status})": r.Capture_Pic_Path
                    for r in df_att.itertuples()
                }
                choice = st.selectbox("Select a record", list(options.keys()))
                pic = options[choice]
                if pic and os.path.exists(pic):
                    st.image(pic, caption=pic, width=340)
                else:
                    st.info("No capture photo for this record (it may have been entered manually).")

        # ---- Absentees + manual entry ----
        with sub_abs:
            all_students = get_students()
            present_ids = set(
                query_df("SELECT Student_ID FROM Attendance WHERE Date = ?", (sel_date,))["Student_ID"].astype(str)
            )
            absentees = all_students[~all_students["ID"].astype(str).isin(present_ids)]

            st.markdown(f"##### Absent on **{sel_date}** — {len(absentees)} student(s)")
            if absentees.empty:
                st.success("🎉 Everyone has a record for this date.")
            else:
                st.markdown(
                    gallery_html([(r.Profile_Pic_Path, r.Name, f"ID {r.ID}") for r in absentees.itertuples()]),
                    unsafe_allow_html=True,
                )
                st.markdown("##### ✍️ Manual entry (teacher override)")
                m1, m2, m3 = st.columns([2, 1, 1])
                labels = {f"{r.Name} (ID {r.ID})": str(r.ID) for r in absentees.itertuples()}
                with m1:
                    who = st.selectbox("Student", list(labels.keys()), key="manual_student")
                with m2:
                    manual_status = st.selectbox("Status", list(STATUS_COLORS), key="manual_status")
                with m3:
                    st.write("")
                    if st.button("Add record"):
                        try:
                            insert_attendance(labels[who], sel_date, datetime.now().strftime("%H:%M:%S"),
                                              None, manual_status)
                            st.toast("Manual record added", icon="✍️")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.warning("A record already exists for this student on that date.")

        # ---- Capture gallery ----
        with sub_gal:
            gal_df = df_att[df_att["Capture_Pic_Path"].notna()] if not df_att.empty else df_att
            if gal_df.empty:
                st.markdown('<div class="empty">No capture photos for this selection.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    gallery_html([
                        (r.Capture_Pic_Path, r.Name or r.Student_ID, f"{r.Date} · {r.Time}")
                        for r in gal_df.head(40).itertuples()
                    ]),
                    unsafe_allow_html=True,
                )
                if len(gal_df) > 40:
                    st.caption("Showing the latest 40 photos.")

        # ---- Students ----
        with sub_stu:
            stu_df = get_students()
            if stu_df.empty:
                st.markdown('<div class="empty">No registered students.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    gallery_html([(r.Profile_Pic_Path, r.Name, f"ID {r.ID}") for r in stu_df.itertuples()]),
                    unsafe_allow_html=True,
                )
                st.dataframe(stu_df, width="stretch")

                with st.expander("🗑️ Remove a student"):
                    del_labels = {f"{r.Name} (ID {r.ID})": str(r.ID) for r in stu_df.itertuples()}
                    del_choice = st.selectbox("Student to remove", list(del_labels.keys()), key="del_student")
                    st.warning("This deletes the student, their profile photo and all their attendance records.")
                    confirm = st.checkbox("I understand, delete permanently", key="del_confirm")
                    if st.button("Delete student", disabled=not confirm):
                        delete_student(del_labels[del_choice])
                        st.toast("Student removed", icon="🗑️")
                        st.rerun()

            if os.path.exists(DB_FILE):
                with open(DB_FILE, "rb") as f:
                    st.download_button("💾 Download database backup", data=f.read(),
                                       file_name="attendance_backup.db", mime="application/octet-stream")

# -----------------------------------------------------------------------------
# TAB 4: ANALYTICS
# -----------------------------------------------------------------------------
with tab_analytics:
    section("Attendance Analytics", "Trends, punctuality and per-student performance")

    range_days = st.select_slider("Date range (days)", options=[7, 14, 30], value=7)
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(range_days - 1, -1, -1)]
    day_strs = [d.strftime("%Y-%m-%d") for d in days]
    d0, d1 = day_strs[0], day_strs[-1]

    df_status = query_df(
        "SELECT Date, Status, COUNT(*) AS Count FROM Attendance WHERE Date BETWEEN ? AND ? GROUP BY Date, Status",
        (d0, d1),
    )
    if df_status.empty:
        pivot = pd.DataFrame(0, index=day_strs, columns=["Present", "Late"])
    else:
        pivot = df_status.pivot_table(
            index="Date", columns="Status", values="Count", aggfunc="sum", fill_value=0
        ).reindex(day_strs, fill_value=0)
    cols = [c for c in STATUS_COLORS if c in pivot.columns] + [c for c in pivot.columns if c not in STATUS_COLORS]
    pivot = pivot[cols]
    pivot.index.name = "Date"
    total_trend = (pivot.get("Present", 0) + pivot.get("Late", 0)).rename("Students Present")

    k1, k2, k3 = st.columns(3)
    k1.metric("Check-ins in range", int(pivot.values.sum()))
    k2.metric("Avg. per day", f"{pivot.values.sum() / range_days:.1f}")
    late_total = int(pivot["Late"].sum()) if "Late" in pivot.columns else 0
    k3.metric("Late arrivals", late_total)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("##### 📊 Daily attendance by status")
        st.bar_chart(pivot, color=[STATUS_COLORS.get(c, "#94a3b8") for c in pivot.columns])
    with ch2:
        st.markdown("##### 📈 Attendance trend")
        st.area_chart(total_trend, color="#8b5cf6")

    ch3, ch4 = st.columns(2)
    with ch3:
        st.markdown("##### ⏱️ Check-in time distribution (by hour)")
        df_hour = query_df(
            "SELECT substr(Time, 1, 2) AS Hour, COUNT(*) AS Check_ins FROM Attendance "
            "WHERE Date BETWEEN ? AND ? GROUP BY Hour",
            (d0, d1),
        )
        hours = [f"{h:02d}" for h in range(24)]
        hour_series = (
            df_hour.set_index("Hour")["Check_ins"].reindex(hours, fill_value=0)
            if not df_hour.empty else pd.Series(0, index=hours, name="Check_ins")
        )
        hour_series.index.name = "Hour"
        st.bar_chart(hour_series, color="#ec4899")
    with ch4:
        st.markdown("##### 🎓 Days present per student")
        df_lb = query_df(
            """
            SELECT s.ID, s.Name,
                   COUNT(a.Log_ID) AS Days_Present,
                   COALESCE(SUM(CASE WHEN a.Status = 'Late' THEN 1 ELSE 0 END), 0) AS Late_Days
            FROM Students s
            LEFT JOIN Attendance a ON a.Student_ID = s.ID AND a.Date BETWEEN ? AND ?
            GROUP BY s.ID ORDER BY Days_Present DESC, s.Name
            """,
            (d0, d1),
        )
        if df_lb.empty:
            st.info("No students registered yet.")
        else:
            st.bar_chart(df_lb.set_index("Name")["Days_Present"], color="#6366f1")

    st.markdown("##### 🏆 Attendance leaderboard")
    if not df_lb.empty:
        with closing(get_conn()) as conn:
            class_days = conn.execute(
                "SELECT COUNT(DISTINCT Date) FROM Attendance WHERE Date BETWEEN ? AND ?", (d0, d1)
            ).fetchone()[0]
        df_lb["Attendance %"] = (df_lb["Days_Present"] / class_days * 100).round(0) if class_days else 0.0
        st.dataframe(
            df_lb,
            width="stretch",
            hide_index=True,
            column_config={
                "Attendance %": st.column_config.ProgressColumn(
                    "Attendance %", min_value=0, max_value=100, format="%.0f%%"
                )
            },
        )
        st.caption(f"Attendance % is based on {class_days} active class day(s) in the selected range.")
        if class_days:
            low = df_lb[df_lb["Attendance %"] < 75]
            if not low.empty:
                st.warning("⚠️ Below 75% attendance: " + ", ".join(low["Name"].astype(str)))

# =============================================================================
# 7. FILL PLACEHOLDERS WITH FRESH NUMBERS + FOOTER
# =============================================================================
_stats = today_stats()
hero_ph.markdown(hero_html(_stats), unsafe_allow_html=True)
stats_ph.markdown(stats_html(_stats), unsafe_allow_html=True)
sidebar_stats_ph.markdown(sidebar_stats_html(_stats), unsafe_allow_html=True)
st.markdown('<div class="footer">Built with Streamlit · DeepFace · OpenCV · SQLite</div>', unsafe_allow_html=True)
