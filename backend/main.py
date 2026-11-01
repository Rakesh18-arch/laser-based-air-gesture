from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

APP_NAME = "AIRWRITE Cloud API"
SESSION_COOKIE = "airwrite_session"
SESSION_SECONDS = 60 * 60 * 24 * 7
PBKDF2_ROUNDS = 310_000
MAX_BODY_BYTES = 1_000_000
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
ADMIN_EMAIL = os.environ.get("AIRWRITE_ADMIN_EMAIL", "").strip().lower()
ADMIN_PASSWORD = os.environ.get("AIRWRITE_ADMIN_PASSWORD", "")
ALLOWED_ORIGINS = [x.strip().rstrip("/") for x in os.environ.get(
    "AIRWRITE_ALLOWED_ORIGINS", "https://laserhandgesture.netlify.app"
).split(",") if x.strip()]

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    origin = request.headers.get("origin", "").rstrip("/")
    if request.method == "POST" and origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"ok": False, "error": "Request origin is not allowed."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    return response


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured. Add your Neon PostgreSQL connection string.")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=10)


def password_hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS).hex()


def password_matches(password: str, salt_hex: str, expected_hex: str) -> bool:
    try:
        actual = password_hash(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(actual, expected_hex)
    except (ValueError, TypeError):
        return False


def public_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"], "full_name": row["full_name"], "email": row["email"],
        "organisation": row["organisation"], "role": row["role"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "last_login": row["last_login"].isoformat() if row.get("last_login") else None,
        "login_count": row.get("login_count", 0),
    }


def init_database() -> None:
    if not ADMIN_EMAIL or len(ADMIN_PASSWORD) < 16:
        raise RuntimeError("Set AIRWRITE_ADMIN_EMAIL and a unique AIRWRITE_ADMIN_PASSWORD of at least 16 characters in the hosting dashboard.")
    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                full_name VARCHAR(100) NOT NULL,
                email VARCHAR(180) NOT NULL UNIQUE,
                organisation VARCHAR(140) NOT NULL DEFAULT '',
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                password_salt VARCHAR(64) NOT NULL,
                password_hash VARCHAR(128) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                last_login TIMESTAMPTZ,
                login_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS pilot_enquiries (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
                name VARCHAR(160) NOT NULL,
                email VARCHAR(180) NOT NULL,
                organisation VARCHAR(180) NOT NULL,
                role VARCHAR(120) NOT NULL,
                city VARCHAR(120) NOT NULL,
                country VARCHAR(120) NOT NULL,
                organisation_type VARCHAR(120) NOT NULL,
                deployment_size VARCHAR(120) NOT NULL,
                current_setup VARCHAR(500) NOT NULL,
                interest VARCHAR(180) NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
                event VARCHAR(80) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash VARCHAR(64) PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)
        if ADMIN_EMAIL and ADMIN_PASSWORD:
            row = db.execute("SELECT id, role FROM users WHERE email=%s", (ADMIN_EMAIL,)).fetchone()
            if row is None:
                salt = secrets.token_bytes(16)
                db.execute(
                    "INSERT INTO users(full_name,email,organisation,role,password_salt,password_hash,created_at) "
                    "VALUES(%s,%s,%s,'admin',%s,%s,%s)",
                    ("AIRWRITE Owner", ADMIN_EMAIL, "AIRWRITE", salt.hex(), password_hash(ADMIN_PASSWORD, salt), now_utc()),
                )
            elif row["role"] != "admin":
                raise RuntimeError("AIRWRITE_ADMIN_EMAIL already belongs to a non-admin account. Choose a dedicated owner email.")
        db.execute("DELETE FROM sessions WHERE expires_at < %s", (now_utc(),))


@app.on_event("startup")
def startup() -> None:
    init_database()


def error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status)


async def read_payload(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return error("Request body is too large.", 413)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return error("Expected a JSON object.", 400)
        return data
    except Exception:
        return error("Request must contain valid JSON.", 400)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def set_session(response: Response, user_id: int) -> None:
    token = secrets.token_urlsafe(32)
    with connect() as db:
        db.execute("INSERT INTO sessions(token_hash,user_id,expires_at,created_at) VALUES(%s,%s,%s,%s)",
                   (token_digest(token), user_id, now_utc() + timedelta(seconds=SESSION_SECONDS), now_utc()))
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=True, samesite="lax", path="/")


def current_user(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        return None
    with connect() as db:
        row = db.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=%s AND s.expires_at>%s",
            (token_digest(token), now_utc()),
        ).fetchone()
        if not row:
            return None
        db.execute("UPDATE sessions SET expires_at=%s WHERE token_hash=%s",
                   (now_utc() + timedelta(seconds=SESSION_SECONDS), token_digest(token)))
        return public_user(row)


def require_user(request: Request, admin: bool = False):
    user = current_user(request)
    if not user:
        return None, error("Please log in first.", 401)
    if admin and user["role"] != "admin":
        return None, error("Owner access is required.", 403)
    return user, None


@app.get("/health")
def health():
    try:
        with connect() as db:
            db.execute("SELECT 1")
        return {"ok": True, "service": APP_NAME, "database": "connected"}
    except Exception:
        return JSONResponse({"ok": False, "service": APP_NAME, "database": "unavailable"}, status_code=503)


@app.get("/api/me")
def me(request: Request):
    user = current_user(request)
    if not user:
        return error("Not signed in.", 401)
    return {"ok": True, "user": user}


@app.post("/api/register")
async def register(request: Request):
    data = await read_payload(request)
    if isinstance(data, JSONResponse):
        return data
    name = str(data.get("full_name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    organisation = str(data.get("organisation", "")).strip()
    password = str(data.get("password", ""))
    if not 2 <= len(name) <= 100:
        return error("Enter your full name (2–100 characters).")
    if "@" not in email or len(email) > 180 or any(ch.isspace() for ch in email):
        return error("Enter a valid email address.")
    if len(organisation) > 140 or not 8 <= len(password) <= 128:
        return error("Password must be 8–128 characters and organisation must be under 140 characters.")
    if data.get("consent") is not True:
        return error("Please accept the account and cloud-data notice.")
    salt = secrets.token_bytes(16)
    try:
        with connect() as db:
            now = now_utc()
            row = db.execute(
                "INSERT INTO users(full_name,email,organisation,role,password_salt,password_hash,created_at,last_login,login_count) "
                "VALUES(%s,%s,%s,'user',%s,%s,%s,%s,1) RETURNING *",
                (name, email, organisation, salt.hex(), password_hash(password, salt), now, now),
            ).fetchone()
            db.execute("INSERT INTO activity_log(user_id,event,created_at) VALUES(%s,%s,%s)", (row["id"], "registered", now))
    except psycopg.errors.UniqueViolation:
        return error("An account with that email already exists. Please log in.", 409)
    response = JSONResponse({"ok": True, "message": "Your AIRWRITE account is ready.", "user": public_user(row)}, status_code=201)
    set_session(response, row["id"])
    return response


@app.post("/api/login")
async def login(request: Request):
    data = await read_payload(request)
    if isinstance(data, JSONResponse):
        return data
    identifier = str(data.get("identifier", "")).strip().lower()
    password = str(data.get("password", ""))
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE email=%s", (identifier,)).fetchone()
        if not row or not password_matches(password, row["password_salt"], row["password_hash"]):
            db.execute("INSERT INTO activity_log(user_id,event,created_at) VALUES(NULL,%s,%s)", ("failed_login", now_utc()))
            return error("Email or password is incorrect.", 401)
        now = now_utc()
        db.execute("UPDATE users SET last_login=%s,login_count=login_count+1 WHERE id=%s", (now, row["id"]))
        db.execute("INSERT INTO activity_log(user_id,event,created_at) VALUES(%s,%s,%s)", (row["id"], "login", now))
        updated = db.execute("SELECT * FROM users WHERE id=%s", (row["id"],)).fetchone()
    response = JSONResponse({"ok": True, "message": "Welcome back.", "user": public_user(updated)})
    set_session(response, updated["id"])
    return response


@app.post("/api/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        with connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=%s", (token_digest(token),))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, secure=True, samesite="lax")
    return response


@app.get("/api/admin/users")
def admin_users(request: Request):
    user, denied = require_user(request, admin=True)
    if denied:
        return denied
    with connect() as db:
        rows = db.execute("SELECT id,full_name,email,organisation,created_at,last_login,login_count FROM users WHERE role='user' ORDER BY created_at DESC").fetchall()
    users = [{**r, "created_at": r["created_at"].isoformat(), "last_login": r["last_login"].isoformat() if r["last_login"] else None} for r in rows]
    return {"ok": True, "users": users, "count": len(users)}


@app.get("/api/admin/enquiries")
def admin_enquiries(request: Request):
    user, denied = require_user(request, admin=True)
    if denied:
        return denied
    with connect() as db:
        rows = db.execute("SELECT id,name,email,organisation,role,city,country,organisation_type,deployment_size,current_setup,interest,message,created_at FROM pilot_enquiries ORDER BY created_at DESC").fetchall()
    enquiries = [{**r, "created_at": r["created_at"].isoformat()} for r in rows]
    return {"ok": True, "enquiries": enquiries, "count": len(enquiries)}


@app.get("/api/admin/export.csv")
def admin_export(request: Request):
    user, denied = require_user(request, admin=True)
    if denied:
        return denied
    with connect() as db:
        rows = db.execute("SELECT full_name,email,organisation,created_at,last_login,login_count FROM users WHERE role='user' ORDER BY created_at DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Email", "Organisation", "Registered at", "Last login", "Login count"])
    for row in rows:
        writer.writerow([row["full_name"], row["email"], row["organisation"], row["created_at"].isoformat(), row["last_login"].isoformat() if row["last_login"] else "", row["login_count"]])
    return StreamingResponse(iter(["\ufeff" + output.getvalue()]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=airwrite_registrations.csv"})


@app.post("/api/pilot")
async def pilot(request: Request):
    user, denied = require_user(request)
    if denied:
        return denied
    data = await read_payload(request)
    if isinstance(data, JSONResponse):
        return data
    keys = ["name", "email", "organisation", "role", "city", "country", "organisation_type", "deployment_size", "current_setup", "interest", "message"]
    values = {key: str(data.get(key, "")).strip() for key in keys}
    if any(not values[key] for key in keys):
        return error("Please complete all required enquiry fields.")
    if len(values["message"]) > 5000:
        return error("Please keep the use-case message under 5,000 characters.")
    with connect() as db:
        db.execute(
            "INSERT INTO pilot_enquiries(user_id,name,email,organisation,role,city,country,organisation_type,deployment_size,current_setup,interest,message,created_at) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (user["id"], *(values[key] for key in keys), now_utc()),
        )
    return JSONResponse({"ok": True, "message": "Your pilot enquiry was saved to the AIRWRITE cloud database."}, status_code=201)
