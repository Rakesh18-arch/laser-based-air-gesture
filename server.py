from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import mimetypes
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "website"
DESKTOP_ROOT = ROOT / "desktop"
DATA_ROOT = ROOT / "data"
DB_PATH = DATA_ROOT / "airwrite_users.sqlite3"
OWNER_LOGIN_FILE = DATA_ROOT / "OWNER_LOGIN.txt"
HOST = "127.0.0.1"
PORT = int(os.environ.get("AIRWRITE_PORT", "8765"))
SESSION_COOKIE = "airwrite_session"
SESSION_SECONDS = 60 * 60 * 24 * 7
PBKDF2_ROUNDS = 310_000
MAX_BODY_BYTES = 1_000_000
REMOTE_ORIGINS = {x.strip().rstrip("/") for x in os.environ.get("AIRWRITE_ALLOWED_ORIGINS", "https://laserhandgesture.netlify.app").split(",") if x.strip()}

DATA_ROOT.mkdir(exist_ok=True)
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
_desktop_process: subprocess.Popen | None = None
_desktop_mode: str | None = None
_desktop_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> tuple[str | None, str | None]:
    with db_connect() as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                organisation TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'user',
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                login_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS pilot_enquiries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                organisation TEXT NOT NULL,
                role TEXT NOT NULL,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                organisation_type TEXT NOT NULL,
                deployment_size TEXT NOT NULL,
                current_setup TEXT NOT NULL,
                interest TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        owner = db.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
        if owner:
            return None, None

        password = secrets.token_urlsafe(15)
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
        db.execute(
            "INSERT INTO users (full_name,email,organisation,role,password_salt,password_hash,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            ("AIRWRITE Owner", "owner@airwrite.local", "AIRWRITE", "admin",
             salt.hex(), digest.hex(), utc_now()),
        )
        db.commit()
        OWNER_LOGIN_FILE.write_text(
            "AIRWRITE LOCAL OWNER LOGIN\n"
            "==========================\n"
            "Username: owner\n"
            f"Password: {password}\n\n"
            "Keep this file private. It is stored locally in the package's data folder.\n"
            "Use these credentials in the website Login screen to open the Owner Dashboard.\n",
            encoding="utf-8",
        )
        return "owner", password


def password_matches(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
        return hmac.compare_digest(actual.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def get_session(handler: BaseHTTPRequestHandler) -> dict | None:
    cookie = SimpleCookie()
    try:
        cookie.load(handler.headers.get("Cookie", ""))
        morsel = cookie.get(SESSION_COOKIE)
        token = morsel.value if morsel else ""
    except Exception:
        token = ""
    if not token:
        return None
    with _sessions_lock:
        item = _sessions.get(token)
        if not item:
            return None
        if item["expires"] < time.time():
            _sessions.pop(token, None)
            return None
        item["expires"] = time.time() + SESSION_SECONDS
        return {"token": token, "user_id": item["user_id"], "role": item["role"]}


def session_user(session: dict | None) -> dict | None:
    if not session:
        return None
    with db_connect() as db:
        row = db.execute(
            "SELECT id,full_name,email,organisation,role,created_at,last_login,login_count "
            "FROM users WHERE id=?", (session["user_id"],)
        ).fetchone()
    return dict(row) if row else None


class AirwriteHandler(BaseHTTPRequestHandler):
    server_version = "AIRWRITE-Local/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Keep a useful but compact local console log.
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _valid_host(self) -> bool:
        host = self.headers.get("Host", "").split(":", 1)[0].lower()
        return host in {"127.0.0.1", "localhost"}

    def _valid_origin(self) -> bool:
        origin = self.headers.get("Origin", "").rstrip("/")
        if not origin:
            return True
        return origin in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"} or origin in REMOTE_ORIGINS

    def _send_bytes(self, body: bytes, content_type: str, status=200, extra_headers=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store" if content_type.startswith(("application/json", "text/csv")) else "no-cache")
        origin = self.headers.get("Origin", "").rstrip("/")
        if origin and (origin in REMOTE_ORIGINS or origin in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        if extra_headers:
            for key, value in extra_headers:
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, payload: dict, status=200, extra_headers=None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status, extra_headers)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Request body is empty or too large.")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def require_login(self, admin=False):
        session = get_session(self)
        user = session_user(session)
        if not user:
            self.json_response({"ok": False, "error": "Please log in first."}, 401)
            return None
        if admin and user.get("role") != "admin":
            self.json_response({"ok": False, "error": "Owner access is required."}, 403)
            return None
        return user

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin", "").rstrip("/")
        if not self._valid_host() or origin not in REMOTE_ORIGINS:
            self.send_error(HTTPStatus.FORBIDDEN, "Origin not allowed")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        # Chrome may request this when a public HTTPS page contacts localhost.
        if self.headers.get("Access-Control-Request-Private-Network", "").lower() == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._valid_host():
            self.json_response({"ok": False, "error": "Local AIRWRITE portal only."}, 403)
            return
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/status":
            with _desktop_lock:
                running = _desktop_process is not None and _desktop_process.poll() is None
                pid = _desktop_process.pid if running else None
                desktop_mode = _desktop_mode if running else None
            cal_file = DESKTOP_ROOT / "laser_calibration_9points.txt"
            try:
                point_count = sum(1 for line in cal_file.read_text(encoding="utf-8").splitlines() if line.strip())
            except OSError:
                point_count = 0
            self.json_response({
                "ok": True,
                "portal": "online",
                "desktop_running": running,
                "desktop_pid": pid,
                "desktop_mode": desktop_mode,
                "calibration_profile_points": 5,
                "camera_profile_present": ((DESKTOP_ROOT / "camera_profile.txt").is_file() or (DESKTOP_ROOT / "droidcam_camera.txt").is_file()),
                "desktop_entry": "b.py",
                "portal_scope": "This computer only",
                "database_present": DB_PATH.is_file(),
            })
            return
        if path == "/api/me":
            session = get_session(self)
            user = session_user(session)
            if not user:
                self.json_response({"ok": False, "user": None}, 401)
            else:
                self.json_response({"ok": True, "user": user})
            return
        if path == "/api/admin/users":
            user = self.require_login(admin=True)
            if not user:
                return
            with db_connect() as db:
                rows = db.execute(
                    "SELECT id,full_name,email,organisation,created_at,last_login,login_count "
                    "FROM users WHERE role='user' ORDER BY created_at DESC"
                ).fetchall()
            self.json_response({"ok": True, "users": [dict(row) for row in rows], "count": len(rows)})
            return
        if path == "/api/admin/enquiries":
            user = self.require_login(admin=True)
            if not user:
                return
            with db_connect() as db:
                rows = db.execute(
                    "SELECT id,name,email,organisation,role,city,country,organisation_type,"
                    "deployment_size,current_setup,interest,message,created_at "
                    "FROM pilot_enquiries ORDER BY created_at DESC"
                ).fetchall()
            self.json_response({"ok": True, "enquiries": [dict(row) for row in rows], "count": len(rows)})
            return
        if path == "/api/admin/export.csv":
            user = self.require_login(admin=True)
            if not user:
                return
            with db_connect() as db:
                rows = db.execute(
                    "SELECT full_name,email,organisation,created_at,last_login,login_count "
                    "FROM users WHERE role='user' ORDER BY created_at DESC"
                ).fetchall()
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Name", "Email", "Organisation", "Registered at", "Last login", "Login count"])
            writer.writerows([tuple(row) for row in rows])
            self._send_bytes(output.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8",
                             200, [("Content-Disposition", 'attachment; filename="airwrite_registrations.csv"')])
            return

        if path.startswith("/api/"):
            self.json_response({"ok": False, "error": "API endpoint not found."}, 404)
            return
        if path == "/":
            path = "/index.html"
        relative = Path(path.lstrip("/"))
        file_path = (WEB_ROOT / relative).resolve()
        try:
            file_path.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return
        if not file_path.is_file():
            self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        self._send_bytes(file_path.read_bytes(), mime or "application/octet-stream")

    def do_POST(self) -> None:
        if not self._valid_host() or not self._valid_origin():
            self.json_response({"ok": False, "error": "Request blocked: AIRWRITE accepts local portal requests only."}, 403)
            return
        path = urlparse(self.path).path
        try:
            data = self.read_json() if path != "/api/logout" else {}
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.json_response({"ok": False, "error": str(exc)}, 400)
            return

        if path == "/api/register":
            name = str(data.get("full_name", "")).strip()
            email = str(data.get("email", "")).strip().lower()
            organisation = str(data.get("organisation", "")).strip()
            password = str(data.get("password", ""))
            role = str(data.get("account_type", "Personal")).strip()[:60]
            if len(name) < 2 or len(name) > 100:
                self.json_response({"ok": False, "error": "Enter your full name (2–100 characters)."}, 400); return
            if "@" not in email or len(email) > 180 or any(ch.isspace() for ch in email):
                self.json_response({"ok": False, "error": "Enter a valid email address."}, 400); return
            if len(organisation) > 140 or len(password) < 8 or len(password) > 128:
                self.json_response({"ok": False, "error": "Password must be 8–128 characters and organisation must be under 140 characters."}, 400); return
            if data.get("consent") is not True:
                self.json_response({"ok": False, "error": "Please accept the account and local-data notice."}, 400); return
            salt = secrets.token_bytes(16)
            digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
            try:
                with db_connect() as db:
                    now = utc_now()
                    cur = db.execute(
                        "INSERT INTO users(full_name,email,organisation,role,password_salt,password_hash,created_at,last_login,login_count) "
                        "VALUES(?,?,?,?,?,?,?,?,1)",
                        (name, email, organisation, "user", salt.hex(), digest.hex(), now, now),
                    )
                    user_id = cur.lastrowid
                    db.execute("INSERT INTO activity_log(user_id,event,created_at) VALUES(?,?,?)",
                               (user_id, "registered", utc_now()))
                    db.commit()
            except sqlite3.IntegrityError:
                self.json_response({"ok": False, "error": "An account with that email already exists. Please log in."}, 409); return
            token = secrets.token_urlsafe(32)
            with _sessions_lock:
                _sessions[token] = {"user_id": user_id, "role": "user", "expires": time.time() + SESSION_SECONDS}
            headers = [("Set-Cookie", f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={SESSION_SECONDS}")]
            self.json_response({"ok": True, "message": "Your AIRWRITE account is ready.",
                                "user": {"id": user_id, "full_name": name, "email": email, "organisation": organisation, "role": "user"}}, 201, headers)
            return

        if path == "/api/login":
            identifier = str(data.get("identifier", "")).strip().lower()
            password = str(data.get("password", ""))
            with db_connect() as db:
                row = db.execute(
                    "SELECT * FROM users WHERE email=? COLLATE NOCASE OR (?='owner' AND role='admin') LIMIT 1",
                    (identifier, identifier),
                ).fetchone()
                if not row or not password_matches(password, row["password_salt"], row["password_hash"]):
                    db.execute("INSERT INTO activity_log(user_id,event,created_at) VALUES(NULL,?,?)",
                               ("failed_login", utc_now()))
                    db.commit()
                    self.json_response({"ok": False, "error": "Email/owner username or password is incorrect."}, 401); return
                db.execute("UPDATE users SET last_login=?,login_count=login_count+1 WHERE id=?", (utc_now(), row["id"]))
                db.execute("INSERT INTO activity_log(user_id,event,created_at) VALUES(?,?,?)",
                           (row["id"], "login", utc_now()))
                db.commit()
                current = db.execute("SELECT id,full_name,email,organisation,role,created_at,last_login,login_count FROM users WHERE id=?", (row["id"],)).fetchone()
            token = secrets.token_urlsafe(32)
            with _sessions_lock:
                _sessions[token] = {"user_id": current["id"], "role": current["role"], "expires": time.time() + SESSION_SECONDS}
            headers = [("Set-Cookie", f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={SESSION_SECONDS}")]
            self.json_response({"ok": True, "message": "Welcome back.", "user": dict(current)}, 200, headers)
            return

        if path == "/api/logout":
            session = get_session(self)
            if session:
                with _sessions_lock:
                    _sessions.pop(session["token"], None)
            headers = [("Set-Cookie", f"{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")]
            self.json_response({"ok": True}, 200, headers)
            return

        if path == "/api/desktop/start":
            # A signed-in user on the hosted portal may start only the desktop
            # process on their own computer. The bridge accepts this one action
            # from the configured website origin; account/admin APIs remain local-authenticated.
            request_origin = self.headers.get("Origin", "").rstrip("/")
            if request_origin not in REMOTE_ORIGINS:
                user = self.require_login()
                if not user:
                    return
            mode = str(data.get("mode", "laser")).strip().lower()
            if mode not in {"laser", "hand"}:
                self.json_response({"ok": False, "error": "Choose Laser Writer or Hand Gesture mode."}, 400); return
            entry = DESKTOP_ROOT / "b.py"
            if not entry.is_file():
                self.json_response({"ok": False, "error": "The desktop entry file b.py is missing from the package."}, 500); return
            python_exe = Path(sys.executable)
            if not python_exe.exists():
                self.json_response({"ok": False, "error": "The local Python runtime could not be found. Run START_AIRWRITE.bat again."}, 500); return
            with _desktop_lock:
                global _desktop_process, _desktop_mode
                if _desktop_process is not None and _desktop_process.poll() is None:
                    if _desktop_mode and _desktop_mode != mode:
                        self.json_response({"ok": False, "error": f"AIRWRITE is already running in {_desktop_mode} mode. Close that desktop window before starting {mode} mode."}, 409); return
                    self.json_response({"ok": True, "status": "already_running", "pid": _desktop_process.pid,
                                        "mode": _desktop_mode or mode,
                                        "message": "AIRWRITE desktop is already running. Switch to its window."}); return
                try:
                    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
                    _desktop_process = subprocess.Popen(
                        [str(python_exe), str(entry), "--mode", mode], cwd=str(DESKTOP_ROOT),
                        creationflags=creation_flags,
                        env={**os.environ, "PYTHONUTF8": "1", "AIRWRITE_MODE": mode},
                    )
                    _desktop_mode = mode
                except OSError as exc:
                    self.json_response({"ok": False, "error": f"Could not launch b.py: {exc}"}, 500); return
                pid = _desktop_process.pid
            launch_name = "Hand Gesture Studio" if mode == "hand" else "Laser Writer"
            self.json_response({"ok": True, "status": "started", "pid": pid, "mode": mode,
                                "message": f"AIRWRITE {launch_name} started. The desktop window will open separately; camera access stays local to this computer."})
            return

        if path == "/api/pilot":
            user = self.require_login()
            if not user:
                return
            keys = ["name", "email", "organisation", "role", "city", "country", "organisation_type",
                    "deployment_size", "current_setup", "interest", "message"]
            values = {key: str(data.get(key, "")).strip() for key in keys}
            if any(not values[key] for key in keys):
                self.json_response({"ok": False, "error": "Please complete all required enquiry fields."}, 400); return
            if len(values["message"]) > 5000:
                self.json_response({"ok": False, "error": "Please keep the use-case message under 5,000 characters."}, 400); return
            with db_connect() as db:
                db.execute(
                    "INSERT INTO pilot_enquiries(user_id,name,email,organisation,role,city,country,organisation_type,"
                    "deployment_size,current_setup,interest,message,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (user["id"], *(values[key] for key in keys), utc_now()),
                )
                db.commit()
            self.json_response({"ok": True, "message": "Your pilot enquiry was saved to the local AIRWRITE database."}, 201)
            return

        self.json_response({"ok": False, "error": "API endpoint not found."}, 404)


def main() -> None:
    owner_name, owner_password = init_database()
    print("=" * 64)
    print(" AIRWRITE LOCAL PORTAL — WEBSITE + DESKTOP BRIDGE")
    print("=" * 64)
    print(f"Website: http://{HOST}:{PORT}/")
    print(f"Database: {DB_PATH}")
    print(f"Desktop entry: {DESKTOP_ROOT / 'b.py'}")
    if owner_name and owner_password:
        print("\nOWNER LOGIN CREATED (save these credentials):")
        print(f"  Username: {owner_name}")
        print(f"  Password: {owner_password}")
        print(f"  Saved privately in: {OWNER_LOGIN_FILE}")
    elif OWNER_LOGIN_FILE.exists():
        print(f"Owner login details: {OWNER_LOGIN_FILE}")
    print("\nOnly this computer can connect. Keep this terminal window open while using AIRWRITE.")
    print("Press Ctrl+C here to stop the local portal. The desktop window is a separate process.")
    print("=" * 64)
    try:
        server = ThreadingHTTPServer((HOST, PORT), AirwriteHandler)
    except OSError as exc:
        print(f"\n[ERROR] Could not start the local AIRWRITE portal on {HOST}:{PORT}: {exc}")
        print("Close another AIRWRITE portal or the program using this port, then try again.")
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        return
    server.daemon_threads = True
    if os.environ.get("AIRWRITE_NO_BROWSER", "0") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}/", new=1)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping AIRWRITE local portal...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
