"""AIRWRITE desktop application with separate Laser and Hand launch modes.

Laser mode uses its confirmed DroidCam camera and calibrated spot detector.
Hand mode uses the dedicated MediaPipe hand_gesture_engine.py and its own
saved webcam profile. The two modes never share or overwrite camera profiles.
"""
from __future__ import annotations
import os, sys, time, math, threading, traceback, re, shutil, importlib.util, hashlib, json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load the detector from this exact project folder. A generic `import air_writer`
# can return an unrelated/stale module already present in sys.modules, which
# caused the observed AttributeError for build_color_mask during calibration.
DETECTOR_FILE = (ROOT / "air_writer.py").resolve()
if not DETECTOR_FILE.is_file():
    raise FileNotFoundError(
        "AIRWRITE detector file is missing beside the application: "
        f"{DETECTOR_FILE}. Restore air_writer.py from the original project archive."
    )
_DETECTOR_MODULE_NAME = "_airwrite_project_detector_" + hashlib.sha1(
    str(DETECTOR_FILE).encode("utf-8")
).hexdigest()[:12]
_DETECTOR_SPEC = importlib.util.spec_from_file_location(
    _DETECTOR_MODULE_NAME, str(DETECTOR_FILE)
)
if _DETECTOR_SPEC is None or _DETECTOR_SPEC.loader is None:
    raise ImportError(f"Could not create a loader for AIRWRITE detector: {DETECTOR_FILE}")
detector = importlib.util.module_from_spec(_DETECTOR_SPEC)
sys.modules[_DETECTOR_MODULE_NAME] = detector
try:
    _DETECTOR_SPEC.loader.exec_module(detector)
except Exception:
    sys.modules.pop(_DETECTOR_MODULE_NAME, None)
    raise
_REQUIRED_DETECTOR_API = ("build_color_mask", "find_candidates")
_MISSING_DETECTOR_API = [name for name in _REQUIRED_DETECTOR_API if not callable(getattr(detector, name, None))]
if _MISSING_DETECTOR_API:
    raise ImportError(
        "The air_writer.py beside this application is not the expected AIRWRITE detector. "
        f"Loaded: {getattr(detector, '__file__', DETECTOR_FILE)}; "
        f"missing functions: {', '.join(_MISSING_DETECTOR_API)}. "
        "Restore the original air_writer.py from your vishnu archive."
    )
DETECTOR_SOURCE_PATH = str(Path(detector.__file__).resolve())
print(f"[AIRWRITE] Loaded laser detector from: {DETECTOR_SOURCE_PATH}")

class LaserDwellGate:
    """Require fresh, consecutive, spatially stable laser samples on one target."""
    def __init__(self, dwell_seconds=0.20, stable_radius_px=28.0, max_gap_seconds=0.16, min_samples=1):
        self.dwell_seconds = float(dwell_seconds)
        self.stable_radius_px = float(stable_radius_px)
        self.max_gap_seconds = float(max_gap_seconds)
        self.min_samples = max(1, int(min_samples))
        self.reset()

    def reset(self):
        self.target = None
        self.started_at = None
        self.last_sample_at = None
        self.anchor = None
        self.points = []
        self.samples = 0

    def observe(self, sample_time, point, target):
        """Feed one *new camera detection*. Cached cursor frames must not count."""
        if point is None or target is None:
            self.reset()
            return
        sample_time = float(sample_time)
        px, py = float(point[0]), float(point[1])
        if self.last_sample_at is not None and sample_time <= self.last_sample_at + 1e-6:
            return
        restart = self.started_at is None
        if self.last_sample_at is not None:
            gap = sample_time - self.last_sample_at
            moved = math.hypot(px - self.anchor[0], py - self.anchor[1])
            restart = restart or gap > self.max_gap_seconds or target != self.target or moved > self.stable_radius_px
        if restart:
            self.target = target
            self.started_at = sample_time
            self.anchor = (px, py)
            self.points = [(px, py)]
            self.samples = 1
        else:
            self.points.append((px, py))
            self.points = self.points[-9:]
            self.samples += 1
        self.last_sample_at = sample_time

    def ready(self, current_time=None, trailing_release_delay=0.0):
        """Return true after a same-target dwell, allowing for detector/OFF latency.

        Laser detections stop when the beam is switched off. The app then waits
        for LASER_DETECTION_RELEASE_GRACE_SECONDS and LASER_OFF_CONFIRM_SECONDS
        before dispatching. Account for those known delays, but reject a stale
        last sample and never allow a target change or excessive movement to
        inherit the previous target's dwell.
        """
        if (self.started_at is None or self.last_sample_at is None
                or self.samples < self.min_samples):
            return False
        held = self.held_seconds
        if current_time is not None:
            now = float(current_time)
            tail = max(0.0, float(trailing_release_delay))
            if now - self.last_sample_at > self.max_gap_seconds:
                return False
            inferred_hold = max(0.0, now - self.started_at - tail)
            held = max(held, inferred_hold)
        return held >= self.dwell_seconds

    @property
    def held_seconds(self):
        if self.started_at is None or self.last_sample_at is None:
            return 0.0
        return max(0.0, self.last_sample_at - self.started_at)

    @property
    def stable_point(self):
        if not self.points:
            return None
        values = np.asarray(self.points, dtype=np.float32)
        return float(np.median(values[:, 0])), float(np.median(values[:, 1]))



class LatestCameraFrameReader:
    """Read one VideoCapture on a worker thread and publish only the newest frame.

    The UI thread never calls cap.read(). This avoids starving Tk's event loop while
    Windows Media Foundation/DroidCam waits for a frame. Exactly one reader owns
    the capture object after the initial open handshake.
    """
    def __init__(self, capture, first_frame=None):
        self.capture = capture
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame = first_frame.copy() if first_frame is not None else None
        self._sequence = 1 if first_frame is not None else 0
        self._timestamp = time.time() if first_frame is not None else 0.0
        self._failures = 0
        self._error = ""
        self._thread = threading.Thread(
            target=self._run, name="AIRWRITE-DroidCam-Reader", daemon=True
        )
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                ok, frame = self.capture.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    stamp = time.time()
                    with self._lock:
                        self._frame = frame
                        self._sequence += 1
                        self._timestamp = stamp
                        self._failures = 0
                        self._error = ""
                else:
                    with self._lock:
                        self._failures += 1
                        self._error = "VideoCapture.read() returned no frame"
                    self._stop_event.wait(0.012)
            except Exception as exc:
                with self._lock:
                    self._failures += 1
                    self._error = f"{type(exc).__name__}: {exc}"
                self._stop_event.wait(0.025)

    def snapshot(self):
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return frame, self._sequence, self._timestamp, self._failures, self._error

    def stop(self):
        self._stop_event.set()
        try:
            self._thread.join(timeout=0.40)
        except Exception:
            pass
        try:
            self.capture.release()
        except Exception:
            pass

APP_NAME = "Air Writer Smart Board"
DARK = "#10131a"
PANEL = "#191e28"
ACCENT = "#58d6c7"
TEXT = "#eef3f8"
MUTED = "#aab5c5"
CAL_FILE = ROOT / "laser_calibration_9points.txt"
# Calibration must use real, fresh detections from the current camera frame.
# A stale cursor location is never allowed to complete a calibration target.
CALIBRATION_MIN_CONFIDENCE = 0.35
# Require a deliberate, stable dwell over a radial-menu item before an ON→OFF
# selection can execute. This prevents a transient cursor crossing from firing.
RADIAL_MENU_DWELL_SECONDS = 0.20
RADIAL_MENU_HIT_RADIUS = 62.0
LASER_OFF_CONFIRM_SECONDS = 0.10
LASER_DETECTION_RELEASE_GRACE_SECONDS = 0.18
LASER_MIN_ACTIVE_SAMPLES = 1
LASER_ACTIVE_MAX_AGE_SECONDS = 0.50
def _configured_laser_click_dwell():
    """Dwell needed on one target before a confirmed laser OFF can click."""
    raw = os.environ.get("AIRWRITE_CLICK_DWELL_SECONDS", "0.30")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.30
    return max(0.20, min(1.00, value))


LASER_CLICK_DWELL_SECONDS = _configured_laser_click_dwell()
LASER_BUTTON_HIT_TOLERANCE_PX = 42.0
LASER_DWELL_STABILITY_PX = 64.0
LASER_DWELL_MAX_GAP_SECONDS = 0.65
RIGHT_CLICK_HOLD_SECONDS = 0.95  # Retained for older settings; confirmed OFF cycles now left-click.
MOUSE_CLICK_COOLDOWN_SECONDS = 0.26
# Keep automated OS pointer movement away from exact screen corners, which
# trigger PyAutoGUI's global fail-safe and previously disabled all click input.
DESKTOP_POINTER_SAFE_MARGIN = 5
RUNTIME_LOG_FILE = ROOT / "air_writer_runtime.log"
HAND_GESTURE_PROFILE_FILE = ROOT / "hand_gesture_profile.json"
# Version 4 deliberately re-runs the six documented lessons once for users
# who launched the previous build before the current first-use learning flow.
HAND_GESTURE_ACADEMY_VERSION = 4
HAND_ACADEMY_HOLD_SECONDS = 0.90
HAND_ACADEMY_STEPS = (
    ("verify", "Verify one hand", "Hold one hand naturally in the camera view. No special palm-facing pose is required.", "Hand verified"),
    ("index", "Index-finger pointer", "Extend your index finger and relax the other fingers. Keep it steady until verified.", "Pointer verified"),
    ("pinch", "Thumb + index pinch", "Bring your thumb and index fingertip together. This draws, clicks and selects.", "Pinch verified"),
    ("pan", "V-sign pan", "Raise index and middle fingers; fold ring and little fingers. Hold the V sign steadily.", "Pan gesture verified"),
    ("zoom", "Thumb + middle zoom", "Bring thumb to middle fingertip while keeping index away from thumb.", "Zoom gesture verified"),
    ("menu", "Three-finger tools menu", "Raise index, middle and ring fingers; fold the little finger. This opens the tools menu.", "Tools-menu gesture verified"),
)
MENU_GESTURE_WINDOW_SECONDS = 12.0
MENU_GESTURE_MIN_DURATION = 0.12
MENU_GESTURE_MAX_DURATION = 10.5
# Z-gesture recognition is deliberately structured to reject ordinary curves.
# These values describe board-canvas pixels; the existing spot detector is untouched.
MENU_GESTURE_MIN_WIDTH = 80.0
MENU_GESTURE_MIN_DEPTH = 45.0
MENU_GESTURE_TRACKING_GAP_SECONDS = 1.40
# Smart Write turns a completed word into editable text after the user pauses.
# These settings affect only the board's writing workflow, not laser detection.
SMART_WRITE_PAUSE_SECONDS = 1.25
SMART_WRITE_MIN_CONFIDENCE = 25.0
SMART_WRITE_FONT_POINTS = 15
# Glide typing is a separate keyboard-input mode inside Smart Write. It uses
# the already mapped cursor path and does not inspect camera frames.
SWIPE_TYPE_RELEASE_GRACE_SECONDS = 0.28
# Avoid accidental words from rapid crossings: letters/actions require a stable dwell.
SWIPE_TYPE_KEY_DWELL_SECONDS = 1.0
SWIPE_TYPE_SUGGESTION_DWELL_SECONDS = 0.70
SWIPE_TYPE_PREDICT_INTERVAL_SECONDS = 0.24
SWIPE_TYPE_MIN_PATH_LENGTH_FACTOR = 0.42
SWIPE_TYPE_MAX_PATH_POINTS = 180
SWIPE_TYPE_MAX_PREDICTION_POINTS = 36
SWIPE_TYPE_WORD_MIN_SCORE = 0.92
SWIPE_TYPE_FONT_POINTS = 20
SWIPE_TYPE_BUILTIN_WORDS = tuple(dict.fromkeys(r"""
    "a an and are as at be been but by can could did do does down each even every for from get give go good got had has have he her here him his how i if in into is it its just know like little look make many me more most my name near need never new no not now of off on once one only or other our out over people project right said same see she so some such take than that the their them then there these they thing think this through time to too two up us use very want was way we well were what when where which who why will with word work world would write writer writing year yes you your hello hi hey good morning evening thanks thank please sorry welcome test text type swipe keyboard smart smartwrite board air laser pen color colour cursor menu click draw drawing page new undo redo school college class teacher student lesson note notes book phone home laptop computer touchless accessibility easy faster quick accurate accuracy robust stable smooth word words sentence paragraph line clean convert conversion design designer shape circle square triangle rectangle arrow erase eraser pen pencil edit save export file open close back done clear space enter delete shift abc more settings option options help home about start stop go left right up down first last next previous select choose choosing select this that those these we can do it i am you are we are this is my project i need help please make it work works working good morning good night see you thank you very much how are you what is this how does it work air writer smart board vishnu abhi vardhan posampally hyderabad telangana india python code coding app application school college engineer engineering student teacher write writing handwriting hand write handwrite smart designer smart write smart writing swipe typing glide typing type word type text hello world hello there good job nice great better best okay ok fine yes no maybe today tomorrow yesterday now later again try test testing open menu more color red green blue white black yellow purple orange pink gray grey small large big line lines smooth curve curves fast slow quick brown fox jumps over lazy dog the quick brown fox jumps over the lazy dog can you help me i want to write a word when the word is ready show it on the board drag letters to make a word select the right word mobile phone keyboard swipe type smart keyboard prediction predict word prediction words can be corrected another option spell check spell spelling sentence text box add space add line new page previous page next page clear page undo action redo action
""".split()))


def _swipe_point_segment_distance(point, start, end):
    """Euclidean distance from one 2D point to a line segment."""
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def predict_swipe_words(path_points, key_centers, key_width, words=None, max_results=3):
    """Rank offline word suggestions against the geometry of a keyboard-glide path.

    This is a lightweight local baseline rather than a neural keyboard language
    model. It scores candidate letter-key paths in order and supports a user word
    list supplied as swipe_words.txt. The laser detector is not involved here.
    """
    if words is None:
        words = SWIPE_TYPE_BUILTIN_WORDS
    if path_points is None or len(path_points) < 3 or not key_centers:
        return []
    kw = max(24.0, float(key_width))
    points = [(float(p[0]), float(p[1])) for p in path_points if len(p) >= 2]
    if len(points) > SWIPE_TYPE_MAX_PREDICTION_POINTS:
        stride = (len(points) - 1) / (SWIPE_TYPE_MAX_PREDICTION_POINTS - 1)
        points = [points[int(round(i * stride))] for i in range(SWIPE_TYPE_MAX_PREDICTION_POINTS)]
    if len(points) < 3:
        return []
    path_length = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))
    if path_length < kw * SWIPE_TYPE_MIN_PATH_LENGTH_FACTOR:
        return []

    valid_words = []
    seen = set()
    for word in words:
        word = str(word).strip().lower()
        if not (2 <= len(word) <= 16) or not word.isalpha() or word in seen:
            continue
        if all(ch in key_centers for ch in word):
            valid_words.append(word)
            seen.add(word)
    if not valid_words:
        return []

    estimated_letters = max(2.0, min(15.0, 1.0 + path_length / (kw * 0.85)))
    ranked = []
    for rank_index, word in enumerate(valid_words):
        route = [key_centers[ch] for ch in word]
        if len(route) < 2:
            continue
        # How closely the sampled swipe follows the polyline through candidate keys.
        route_costs = []
        for point in points:
            route_costs.append(min(_swipe_point_segment_distance(point, a, b)
                                   for a, b in zip(route, route[1:])))
        route_cost = (sum(route_costs) / len(route_costs)) / kw

        # Ordered alignment: each candidate letter must be visited in word order.
        # A repeated letter can be matched to nearby consecutive samples on the same key.
        first = route[0]
        previous = [math.hypot(p[0]-first[0], p[1]-first[1]) / kw for p in points]
        for letter_center in route[1:]:
            current = []
            best_prefix = float('inf')
            for i, point in enumerate(points):
                # Keeping the path index non-decreasing is what distinguishes e.g.
                # "ab" from "ba" even when their key centres are nearby.
                if i > 0:
                    best_prefix = min(best_prefix, previous[i-1])
                dist = math.hypot(point[0]-letter_center[0], point[1]-letter_center[1]) / kw
                current.append(best_prefix + dist if best_prefix < float('inf') else float('inf'))
            previous = current
        order_cost = min(previous) / max(1, len(word))
        start_cost = math.hypot(points[0][0]-route[0][0], points[0][1]-route[0][1]) / kw
        end_cost = math.hypot(points[-1][0]-route[-1][0], points[-1][1]-route[-1][1]) / kw
        endpoint_cost = (start_cost + end_cost) * 0.5
        # A candidate must begin and end reasonably close to the actual trace
        # endpoints. This rejects a generic straight sweep across a row that
        # happens to pass near a different word's middle keys.
        if start_cost > 0.58 or end_cost > 0.58:
            continue
        length_cost = abs(len(word) - estimated_letters) / max(4.0, estimated_letters)
        # Small rank prior only breaks near-ties in favour of common words.
        score = (0.48 * route_cost + 0.34 * order_cost + 0.16 * endpoint_cost
                 + 0.045 * length_cost + (rank_index / max(1, len(valid_words))) * 0.006)
        ranked.append((score, word))
    ranked.sort(key=lambda row: row[0])
    return [(word, float(score)) for score, word in ranked[:max_results]
            if score <= SWIPE_TYPE_WORD_MIN_SCORE]
# Written by the standalone OpenCV camera test after the user presses P while
# the preview visibly shows the phone. The user's confirmed result is MSMF/0.
CAMERA_PROFILE_FILE = ROOT / "droidcam_camera.txt"
HAND_CAMERA_PROFILE_FILE = ROOT / "hand_camera_profile.txt"
CAMERA_RETRY_INTERVAL_MS = 1800
CAMERA_READ_FAILURE_LIMIT = 8
CAMERA_FRAME_STALE_SECONDS = 1.0
DRAWINGS_DIR = ROOT / "drawings"
DRAWINGS_DIR.mkdir(exist_ok=True)


def load_camera_profile():
    """Read the camera backend/index confirmed by the test preview."""
    profile = None
    try:
        values = {}
        for line in CAMERA_PROFILE_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip()
        backend = values.get("backend", "").upper()
        index = int(values.get("index", "-1"))
        if backend in {"MSMF", "DSHOW"} and index >= 0:
            profile = {"backend": backend, "index": index}
    except (OSError, ValueError):
        pass
    return profile

def load_hand_camera_profile():
    """Load the hand-mode camera selection without touching the laser profile."""
    profile = None
    try:
        values = {}
        for line in HAND_CAMERA_PROFILE_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip()
        backend = values.get("backend", "").upper()
        index = int(values.get("index", "-1"))
        if backend in {"MSMF", "DSHOW"} and index >= 0:
            profile = {"backend": backend, "index": index}
    except (OSError, ValueError):
        pass
    return profile

def save_hand_camera_profile(index, backend):
    """Persist only the Hand Studio camera profile."""
    try:
        HAND_CAMERA_PROFILE_FILE.write_text(
            f"backend={str(backend).upper()}\nindex={int(index)}\n",
            encoding="utf-8",
        )
    except OSError:
        pass

# Stored camera calibration coordinates from the supplied project.
DEFAULT_CAL = np.array([
    [91,101,80,80], [309,77,960,80], [530,77,1840,80],
    [94,226,80,540], [321,251,960,540], [532,238,1840,540],
    [270,390,80,1000], [316,399,960,1000], [534,398,1840,1000]
], dtype=np.float32)


def enumerate_cameras():
    """Return (index, name) pairs from Windows DirectShow device names."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        return [(i, str(name)) for i, name in enumerate(names)]
    except Exception:
        return []


def open_camera(index, preferred_backend="MSMF"):
    """Open the exact OpenCV index using only the visually tested backend.

    Do not retry a different backend here: Windows can enumerate cameras in a
    different order per backend, so index 0 on another backend might be the
    laptop webcam. The in-app scanner lets the user select and save a new pair.
    """
    backend_map = {
        "MSMF": ("Media Foundation (MSMF)", getattr(cv2, "CAP_MSMF", None)),
        "DSHOW": ("DirectShow (DSHOW)", getattr(cv2, "CAP_DSHOW", None)),
    }
    key = str(preferred_backend).upper()
    entry = backend_map.get(key)
    if entry is None or entry[1] is None:
        raise RuntimeError(f"OpenCV backend {key!r} is unavailable. Click Scan OpenCV cameras to select a supported camera.")
    name, backend = entry
    cap = None
    errors = []
    try:
        cap = cv2.VideoCapture(int(index), backend)
        if not cap.isOpened():
            errors.append(f"{name}: device index {index} did not open")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # DroidCam may report isOpened() before its virtual-camera stream is ready.
            # Keep trying the exact saved backend/index while waiting for a real frame.
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    return cap, frame, name
                time.sleep(.10)
            errors.append(f"{name}: index {index} opened but did not deliver frames")
    except Exception as exc:
        errors.append(f"{name}: {exc}")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    raise RuntimeError(
        "Could not read video from the saved DroidCam profile "
        f"(backend={key}, index={index}).\n" + "\n".join(errors) +
        "\n\nClick 'Scan OpenCV cameras' and choose the preview that visibly shows your phone."
    )


def find_laser(frame, background_gray, yolo_model=None, laser_class_id=None, previous_point=None):
    """Return the most plausible bright red laser point plus confidence and mask.

    The project's strict contrast/shape detector remains first choice. A second
    peak-based candidate pass helps when DroidCam makes the dot only 1-2 pixels
    wide; it still requires a bright change from the saved laser-off background
    and red/white-core evidence, rather than following arbitrary red objects.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if background_gray is None or background_gray.shape != gray.shape:
        return None, 0.0, np.zeros_like(gray)

    diff = cv2.absdiff(gray, background_gray)
    _, dm = cv2.threshold(diff, int(getattr(detector, "DIFF_THRESHOLD", 35)), 255, cv2.THRESH_BINARY)
    _, bm = cv2.threshold(gray, int(getattr(detector, "MIN_BRIGHTNESS", 190)), 255, cv2.THRESH_BINARY)
    cm = detector.build_color_mask(hsv)
    strict_mask = cv2.bitwise_and(cv2.bitwise_and(dm, bm), cm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kernel)
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    candidates = []
    try:
        candidates = detector.find_candidates(strict_mask, gray, diff, frame)
    except Exception:
        candidates = []
    mask_for_preview = strict_mask

    if not candidates:
        red_a = cv2.inRange(hsv, np.array([0, 65, 155]), np.array([13, 255, 255]))
        red_b = cv2.inRange(hsv, np.array([167, 65, 155]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(red_a, red_b)
        # A clipped laser core can be almost white; only use it if it is both
        # brighter than the background and very bright in the current frame.
        white_core = cv2.inRange(hsv, np.array([0, 0, 238]), np.array([180, 95, 255]))
        # Red light can have low grayscale luminance even when the red
        # channel is saturated (pure BGR red is only ~76 in grayscale).
        # Detect changed saturated-red pixels separately from clipped-white cores.
        red_hot = cv2.bitwise_and(red, (diff >= 20).astype(np.uint8) * 255)
        white_hot = cv2.bitwise_and(white_core, cv2.bitwise_and(
            (diff >= 28).astype(np.uint8) * 255,
            (gray >= 232).astype(np.uint8) * 255))
        fallback = cv2.bitwise_or(red_hot, white_hot)
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel)
        mask_for_preview = fallback
        count, labels, stats, centers = cv2.connectedComponentsWithStats(fallback, 8)
        h, w = gray.shape[:2]
        for label in range(1, count):
            x, y, ww, hh, area = map(int, stats[label])
            if area < 1 or area > 180:
                continue
            aspect = max(ww, hh) / max(1, min(ww, hh))
            if aspect > 3.2:
                continue
            cx, cy = map(float, centers[label])
            ix, iy = int(round(cx)), int(round(cy))
            r = 7
            x0, x1 = max(0, ix-r), min(w, ix+r+1)
            y0, y1 = max(0, iy-r), min(h, iy+r+1)
            local_gray = gray[y0:y1, x0:x1]
            local_diff = diff[y0:y1, x0:x1]
            local_red = red[y0:y1, x0:x1]
            if local_gray.size == 0:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            rr = np.sqrt((xx-ix)**2 + (yy-iy)**2)
            core = rr <= 2.5
            ring = (rr >= 4.0) & (rr <= 7.0)
            if core.sum() == 0 or ring.sum() < 4:
                continue
            contrast = float(local_gray[core].mean() - local_gray[ring].mean())
            mean_diff = float(local_diff[core].mean())
            red_count = int(np.count_nonzero(local_red))
            core_brightness = float(local_gray[core].mean())
            if contrast < 8 or mean_diff < 24:
                continue
            if red_count == 0 and core_brightness < 248:
                continue
            compactness = min(1.0, 1.0 / max(1.0, aspect))
            score = (0.30 * min(core_brightness/255.0, 1.0)
                     + 0.28 * min(mean_diff/100.0, 1.0)
                     + 0.24 * min(contrast/70.0, 1.0)
                     + 0.13 * min(red_count/8.0, 1.0)
                     + 0.05 * compactness)
            candidates.append({"center": (cx, cy), "score": float(score)})

    if not candidates:
        return None, 0.0, mask_for_preview

    # Bright text/crosshairs on the calibration screen can also be compact and
    # high-contrast. Require red evidence near the candidate so the cursor does
    # not follow arbitrary white UI pixels. A real clipped-white laser core
    # normally keeps a red halo around it.
    red_evidence = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 45, 125]), np.array([15, 255, 255])),
        cv2.inRange(hsv, np.array([165, 45, 125]), np.array([180, 255, 255])),
    )
    filtered = []
    h_img, w_img = gray.shape[:2]
    for candidate in candidates:
        cx, cy = map(int, candidate["center"])
        x0, x1 = max(0,cx-9), min(w_img,cx+10)
        y0, y1 = max(0,cy-9), min(h_img,cy+10)
        if cv2.countNonZero(red_evidence[y0:y1,x0:x1]) > 0:
            filtered.append(candidate)
    candidates = filtered
    if not candidates:
        return None, 0.0, mask_for_preview

    # Temporal association: while a laser lock exists, prefer verified candidates
    # near that lock. Only fall back to distant candidates when none are nearby;
    # the caller then requires repeated frames before accepting a large jump.
    if previous_point is not None:
        try:
            px, py = float(previous_point[0]), float(previous_point[1])
            nearby = [c for c in candidates
                      if math.hypot(float(c["center"][0])-px,
                                    float(c["center"][1])-py) <= 85.0]
            if nearby:
                candidates = nearby
        except (TypeError, ValueError, IndexError):
            pass
    candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

    if yolo_model is not None and laser_class_id is not None:
        # Use the classifier to rank real OpenCV red/brightness candidates, not
        # as an absolute veto. Tiny laser cores can be heavily downsampled by a
        # webcam, so a single low YOLO score must not make the visible spot vanish.
        h, w = frame.shape[:2]
        ranked = []
        for candidate in candidates[:6]:
            cx, cy = map(int, candidate["center"])
            half = 32
            x0, y0, x1, y1 = cx-half, cy-half, cx+half, cy+half
            crop = frame[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]
            if crop.size == 0:
                continue
            pad_t, pad_l = max(0,-y0), max(0,-x0)
            pad_b, pad_r = max(0,y1-h), max(0,x1-w)
            if pad_t or pad_b or pad_l or pad_r:
                crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            crop = cv2.resize(crop, (64, 64))
            try:
                result = yolo_model(crop, verbose=False)[0]
                if result.probs is None:
                    continue
                prob = float(result.probs.data[int(laser_class_id)].item())
                cv_score = float(candidate.get("score", 0.0))
                combined = 0.55 * prob + 0.45 * cv_score
                ranked.append((combined, prob, cv_score, candidate))
            except Exception:
                continue
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            combined, prob, cv_score, best_candidate = ranked[0]
            # Normal acceptance needs both independent cues; a very strong,
            # compact background-changing red point may still pass if the model
            # is uncertain due to crop/exposure differences.
            if (prob >= 0.40 and cv_score >= 0.30) or cv_score >= 0.72:
                confidence = max(0.36, min(0.99, combined))
                return tuple(map(float, best_candidate["center"])), confidence, mask_for_preview
            return None, 0.0, mask_for_preview
        # Keep the OpenCV candidate pass as a fallback if the classifier itself
        # failed to evaluate all crops, rather than turning a runtime model issue
        # into a dead cursor.
        best = candidates[0]
        score = float(best.get("score", 0.0))
        if score >= 0.72:
            return tuple(map(float, best["center"])), min(0.80, score), mask_for_preview
        return None, 0.0, mask_for_preview
    best = candidates[0]
    return tuple(map(float, best["center"])), float(best.get("score", 0.0)), mask_for_preview

class AirWriterApp:
    def __init__(self, root, mode="laser"):
        self.root = root
        self.hand_mode = str(mode or "laser").strip().lower() == "hand"
        self.launch_mode = "hand" if self.hand_mode else "laser"
        self.root.title(APP_NAME)
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.root.configure(bg=DARK)
        self.cap = None
        self.camera_reader = None
        self._last_processed_camera_seq = -1
        self.camera_read_failures = 0
        self.camera_reconnect_pending = False
        self.camera_last_frame_time = 0.0
        self.camera_last_error = ""
        self.camera_index = None
        self.camera_name = "Not connected"
        self.camera_backend = ""
        self.camera_confirmed = False
        self.frame = None
        self.background_gray = None
        self.laser_point = None
        self.laser_score = 0.0
        self.laser_mask = None
        self.last_point_time = 0.0
        self.smoothed = None
        self.camera_devices = []
        self.camera_profile = load_hand_camera_profile() if self.hand_mode else load_camera_profile()
        self.profile_combo_labels = {}
        self.calibration_source_saved = CAL_FILE.is_file()
        self.calibration = self.load_calibration() if not self.hand_mode else None
        self.H = self.make_homography(self.calibration) if not self.hand_mode else None
        # Keep the last/default map only as a visual calibration guide. A fresh
        # five-point fit is required before cursor control is enabled.
        self.calibration_guide_H = None if self.H is None else self.H.copy()
        self.yolo_model = None
        self.yolo_class_id = None
        self.yolo_path = self._find_yolo_weights() if not self.hand_mode else None
        if self.yolo_path is not None:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(self.yolo_path))
                names = self.yolo_model.names
                entries = names.items() if isinstance(names, dict) else enumerate(names)
                for cid, cname in entries:
                    if str(cname).strip().lower() == "laser": self.yolo_class_id = int(cid); break
                if self.yolo_class_id is None: self.yolo_model = None
            except Exception:
                self.yolo_model = None
        # The Hand button enters this real MediaPipe pipeline. These states are
        # completely separate from the laser detector and its calibration data.
        self.hand_engine = None
        self.hand_tracker = None
        self.hand_pose = None
        self.hand_gesture_text = "Waiting for hand"
        self.hand_previous_center = None
        self.hand_last_seen = 0.0
        self.hand_pointer_filter = None
        self.hand_consensus = None
        self.hand_pinch_stabilizer = None
        self._hand_pinch_active = False
        self._hand_zoom_active = False
        self._hand_last_pointer_canvas = None
        self._hand_last_menu_trigger = 0.0
        self._hand_zoom_scale = 1.0
        self._hand_zoom_last_y = None
        self._hand_last_camera_error = ""
        self._hand_pointer_norm = None
        self._hand_handedness_hint = "Unknown"
        self.hand_gesture_profile = self._load_hand_gesture_profile() if self.hand_mode else {}
        try:
            hand_profile_version = int(self.hand_gesture_profile.get("version", 0) or 0)
        except (TypeError, ValueError):
            hand_profile_version = 0
        self.hand_academy_required = bool(
            self.hand_mode and not (
                hand_profile_version >= HAND_GESTURE_ACADEMY_VERSION
                and self.hand_gesture_profile.get("gesture_lesson_completed") is True
            )
        )
        self.hand_academy_window = None
        self.hand_academy_preview = None
        self.hand_academy_title = None
        self.hand_academy_instruction = None
        self.hand_academy_status = None
        self.hand_academy_progress = None
        self.hand_academy_step_index = 0
        self.hand_academy_hold_started = None
        self.hand_academy_advancing = False
        self.hand_academy_step_verified = [False] * len(HAND_ACADEMY_STEPS)
        self.hand_academy_completed_this_run = False
        self.hand_board_preview = None
        self.hand_board_preview_photo = None
        self.hand_board_status = None
        self.hand_board_gesture_chip = None
        if self.hand_mode:
            self._init_hand_engine()

        self.running_board = False
        self.board_win = None
        self.board_canvas = None
        self.board_toolbar = None
        self.board_w, self.board_h = 1280, 720
        self.board_mode = "CURSOR"
        self.pen_color = "#f4f7fb"
        self.pen_size = 5
        self.eraser_size = 32
        self.last_board_point = None
        # Drawing-only adaptive filter state. Cursor tracking and Z-menu gesture
        # recognition continue to use the original calibrated laser position.
        self._stroke_filter_time = 0.0
        self.pages = [[]]
        self.page_index = 0
        self.strokes = self.pages[0]
        self.redo_strokes = []
        self.current_stroke = None
        # In SMART_WRITE mode, collect completed pen strokes into one word
        # until the user pauses. Keep references to the original strokes so a
        # failed OCR attempt never loses ink and undo can restore the sketch.
        self._smart_write_pending_strokes = []
        self._smart_write_last_stroke_time = 0.0
        self._smart_write_processing = False
        # Swipe-to-Type uses a virtual QWERTY keyboard and the same mapped laser
        # cursor. It never changes the spot detector and never injects OS clicks.
        self.swipe_keyboard_open = False
        self.swipe_keyboard_keys = []
        self.swipe_keyboard_bounds = None
        self.swipe_suggestion_targets = []
        self.swipe_type_active = False
        self.swipe_type_path = []
        self.swipe_type_path_length = 0.0
        self.swipe_type_start_time = 0.0
        self.swipe_type_last_seen = 0.0
        self.swipe_type_last_path_point = None
        self.swipe_type_text = ""
        self.swipe_type_suggestions = []
        # The composed sentence is mirrored into the normal board document model
        # while typing, so each confirmed word is immediately visible and savable.
        self._swipe_type_live_text_object = None
        self.swipe_type_last_word_span = None
        self.swipe_type_last_word = ""
        self.swipe_type_last_predict_time = 0.0
        self.swipe_type_hover_target = None
        self.swipe_type_dwell_target = None
        self.swipe_type_dwell_since = 0.0
        self.swipe_type_dwell_fired_target = None
        self.swipe_type_key_width = 70.0
        self._swipe_type_word_cache = None
        # Track which stroke lists are freehand so only handwriting is spline-smoothed.
        self._freehand_stroke_ids = set()
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.shape_type = "Rectangle"
        self.shape_start = None
        self.shape_end = None
        self.swipe_start = None
        self.swipe_last = None
        self.scroll_last_point = None
        self._last_scroll_update = 0.0
        self._suppress_next_laser_click = False
        self.menu_page = "main"
        self.menu_hover_label = None
        self.page_label = None
        self.menu_items = ["Cursor", "Drag", "Pen", "Eraser", "Shapes", "Scroll", "Swipe", "Undo", "Redo", "More"]
        self.menu_ids = []
        self.menu_center = None
        self.menu_hover_since = None
        self.menu_gesture_points = []
        self._last_menu_gesture_center = None
        self.last_menu_trigger = 0
        self._last_menu_gesture_eval = 0.0
        # Menu selection uses a deliberate laser ON -> OFF gesture. After a
        # radial menu opens, require a stable OFF interval first so the OFF
        # that finished drawing the Z cannot accidentally select an item.
        self._menu_waiting_for_release = False
        self._menu_off_since = None
        self._last_board_canvas_point = None
        self.dragging = False
        self.click_laser_on = False
        self.last_mouse_click = 0
        self._laser_on_since = None
        self._laser_was_active = False
        self._laser_active_samples = 0
        self._last_mapped_point = None
        self._last_mapped_was_toolbar = False
        self._drag_down = False
        self._laser_off_since = None
        self._drag_off_since = None
        self._laser_dwell_gate = LaserDwellGate(
            LASER_CLICK_DWELL_SECONDS, LASER_DWELL_STABILITY_PX,
            LASER_DWELL_MAX_GAP_SECONDS, LASER_MIN_ACTIVE_SAMPLES)
        # The camera frame timestamp is captured before detector inference. On
        # slower laptops, valid laser detections may already be >0.22s old when
        # _handle_mouse_signal runs. Track processing time separately so a real
        # current detection is not rejected merely because inference took time.
        self._laser_point_processed_at = 0.0
        self._laser_last_counted_frame_stamp = 0.0
        # Loaded only when a non-Tk desktop click is actually needed. The main app
        # keeps exclusive ownership of the DroidCam reader and laser detector.
        self._click_operations_module = None
        self._click_operations_mouse = None
        # Desktop OS cursor can be suspended after a PyAutoGUI safety stop without
        # disabling canvas/menu interactions. Re-enable explicitly from the UI.
        self._desktop_mouse_suspended = False
        self._pending_desktop_click = None
        self._pending_desktop_click_after_id = None
        self._last_menu_action_time = 0.0
        self._wait_for_laser_off = False
        self._wait_for_laser_off_since = None
        self._pending_jump_point = None
        self._pending_jump_count = 0
        self.auto_phase = "idle"
        self.auto_phase_deadline = 0.0
        self.auto_background_samples = []
        self.auto_background_last_sample = 0.0
        self.auto_connect_attempts = 0
        self.auto_connect_started = False
        self.calibration_state = None
        self.calibration_retry_count = 0
        self.mode_label = None
        self._build_ui()
        self.refresh_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self._tick)
        if self.hand_mode:
            # Hand mode prefers the built-in webcam and never reads or changes
            # the saved DroidCam/laser camera profile.
            self.root.after(700, self._auto_connect_hand_camera)
        else:
            # The visually tested DroidCam profile is connected automatically.
            self.root.after(900, self._auto_connect_saved_camera)

    def _build_hand_ui(self):
        """Simple webcam dashboard; the laser calibration controls are not shown."""
        top = tk.Frame(self.root, bg=DARK, padx=24, pady=18)
        top.pack(fill="x")
        tk.Label(top, text="HAND GESTURE STUDIO", font=("Segoe UI", 23, "bold"),
                 bg=DARK, fg=TEXT).pack(anchor="w")
        tk.Label(top, text="MediaPipe hand landmarks · index pointer · pinch interaction · no spatial calibration",
                 font=("Segoe UI", 10), bg=DARK, fg=MUTED).pack(anchor="w", pady=(2, 0))
        body = tk.Frame(self.root, bg=DARK, padx=22, pady=4)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=PANEL, padx=18, pady=18, width=320)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)
        right = tk.Frame(body, bg=PANEL, padx=14, pady=14)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="01  LAPTOP / USB CAMERA", font=("Segoe UI", 11, "bold"),
                 bg=PANEL, fg=ACCENT).pack(anchor="w")
        self.device_var = tk.StringVar(value="")
        self.device_combo = ttk.Combobox(left, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill="x", pady=(12, 8))
        row = tk.Frame(left, bg=PANEL); row.pack(fill="x")
        self._button(row, "Refresh", self.refresh_devices, outline=True).pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._button(row, "Connect", self.connect_camera).pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.status = tk.Label(left, text="Starting hand engine…", bg=PANEL, fg="#ffbd69",
                               font=("Segoe UI", 9), wraplength=278, justify="left")
        self.status.pack(anchor="w", pady=(14, 8))
        self.hand_academy_button = self._button(left, "Start Gesture Academy", self._start_hand_academy, outline=True)
        self.hand_academy_button.pack(fill="x", pady=(6, 4))
        self._button(left, "Open Smart Board", self.open_board, outline=True).pack(fill="x", pady=4)
        self._button(left, "Close Smart Board", self.close_board, outline=True).pack(fill="x", pady=4)
        tk.Label(left, text="GESTURES", font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w", pady=(18, 7))
        guide = (
            "INDEX FINGER  ·  move pointer\n"
            "THUMB + INDEX PINCH  ·  draw / click / select\n"
            "V SIGN  ·  pan canvas\n"
            "THUMB + MIDDLE  ·  zoom gesture state\n"
            "THREE FINGERS  ·  open tools menu\n\n"
            "First launch: complete the six-step Gesture Academy. After that, the Smart Board keeps the canvas on the left and the live mirrored hand preview + gesture guide on the right.\n\n"
            "Hand video stays on this computer. No spatial hand calibration is used."
        )
        tk.Label(left, text=guide, bg=PANEL, fg=MUTED, wraplength=278, justify="left",
                 font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(right, text="LIVE HAND TRACKING PREVIEW", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))
        self.preview = tk.Label(right, bg="#090b10", fg=MUTED,
                                text="Connecting to the built-in webcam…", font=("Segoe UI", 11))
        self.preview.pack(fill="both", expand=True)
        self.preview_info = tk.Label(right, text="Waiting for camera frames and MediaPipe hand landmarks",
                                     bg=PANEL, fg=MUTED, anchor="w", font=("Consolas", 9))
        self.preview_info.pack(fill="x", pady=(10, 0))
        tk.Label(self.root, text="AIRWRITE HAND GESTURE STUDIO  •  Local MediaPipe processing",
                 bg=DARK, fg="#748095", font=("Segoe UI", 8), pady=8).pack(fill="x")

    def _init_hand_engine(self):
        """Load this project's hand engine and MediaPipe once for Hand mode."""
        engine_path = (ROOT / "hand_gesture_engine.py").resolve()
        if not engine_path.is_file():
            self._hand_last_camera_error = f"Hand engine file is missing: {engine_path}"
            return
        try:
            spec = importlib.util.spec_from_file_location("_airwrite_hand_gesture_engine", str(engine_path))
            if spec is None or spec.loader is None:
                raise ImportError("Could not create a loader for hand_gesture_engine.py")
            engine = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = engine
            spec.loader.exec_module(engine)
            import mediapipe as mp
            self.hand_engine = engine
            self._hand_mp = mp
            self.hand_tracker = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.55, min_tracking_confidence=0.55)
            self.hand_pointer_filter = engine.OneEuroPointFilter(
                min_cutoff=1.6, beta=0.035, minimum_jump_px=34.0, deadband_px=0.9)
            self.hand_consensus = engine.GestureConsensus(window=5, min_votes=3)
            self.hand_pinch_stabilizer = engine.PinchStabilizer(engage_frames=2, release_frames=4)
        except Exception as exc:
            self.hand_engine = None
            self.hand_tracker = None
            self._hand_last_camera_error = f"Could not start hand tracking: {type(exc).__name__}: {exc}"

    def _load_hand_gesture_profile(self):
        """Read local hand-setup metadata only; this file never contains frames."""
        try:
            value = json.loads(HAND_GESTURE_PROFILE_FILE.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_hand_gesture_profile(self):
        """Persist camera/handedness/lesson metadata; never save image or landmarks."""
        profile = {
            "version": HAND_GESTURE_ACADEMY_VERSION,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "camera_name": str(self.camera_name or "Unknown camera"),
            "camera_kind": str(self.hand_gesture_profile.get("camera_kind", "Webcam")),
            "handedness_hint": str(self._hand_handedness_hint or "Unknown"),
            "gesture_lesson_completed": True,
            "lesson_completed_at": datetime.now().isoformat(timespec="seconds"),
            "lessons": [step[0] for step in HAND_ACADEMY_STEPS],
            "storage_note": "Only setup metadata is stored. No images, video, or landmark frames are stored.",
        }
        temp_path = HAND_GESTURE_PROFILE_FILE.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        temp_path.replace(HAND_GESTURE_PROFILE_FILE)
        self.hand_gesture_profile = profile

    def _continue_hand_startup(self):
        """Run Academy before the Board, or skip it after this profile version passed."""
        if not self.hand_mode or self.cap is None or not self.camera_confirmed:
            return
        if self.hand_academy_required:
            self._start_hand_academy()
        else:
            self.open_board()

    def _start_hand_academy(self):
        """Present every hand-control lesson before allowing the first Board launch."""
        if not self.hand_mode:
            return
        if self.cap is None or not self.camera_confirmed or self.hand_tracker is None:
            self.status.config(text="Connect the Hand webcam first. Gesture Academy needs a live camera and MediaPipe.", fg="#ffbd69")
            return
        if self.hand_academy_window is not None:
            try:
                self.hand_academy_window.deiconify()
                self.hand_academy_window.lift()
                self.hand_academy_window.focus_force()
                return
            except Exception:
                self.hand_academy_window = None
        self.hand_academy_required = True
        self.hand_academy_completed_this_run = False
        self.hand_academy_step_index = 0
        self.hand_academy_hold_started = None
        self.hand_academy_advancing = False
        self.hand_academy_step_verified = [False] * len(HAND_ACADEMY_STEPS)

        win = tk.Toplevel(self.root)
        self.hand_academy_window = win
        win.title("AIRWRITE · Hand Gesture Academy")
        win.configure(bg=DARK)
        win.geometry("1120x690")
        win.minsize(960, 600)
        try:
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"1120x690+{max(0, (sw-1120)//2)}+{max(0, (sh-690)//2)}")
        except Exception:
            pass

        header = tk.Frame(win, bg=DARK, padx=24, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="HAND GESTURE ACADEMY", bg=DARK, fg=TEXT,
                 font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(header, text="Verify your hand, then learn each AIRWRITE gesture in order. The Board opens after all lessons are verified.",
                 bg=DARK, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(win, bg=DARK, padx=22, pady=8)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=PANEL, width=385, padx=20, pady=18)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)
        right = tk.Frame(body, bg=PANEL, padx=14, pady=14)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="LESSON", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.hand_academy_title = tk.Label(left, text="1 / 6 · Verify one hand", bg=PANEL, fg=TEXT,
                                           font=("Segoe UI", 17, "bold"), wraplength=335, justify="left")
        self.hand_academy_title.pack(anchor="w", pady=(10, 8))
        self.hand_academy_instruction = tk.Label(left, text=HAND_ACADEMY_STEPS[0][2], bg=PANEL, fg=MUTED,
                                                 font=("Segoe UI", 10), wraplength=335, justify="left")
        self.hand_academy_instruction.pack(anchor="w", pady=(0, 14))
        self.hand_academy_progress = ttk.Progressbar(left, mode="determinate", maximum=100, value=0)
        self.hand_academy_progress.pack(fill="x", pady=(0, 10))
        self.hand_academy_status = tk.Label(left, text="Show one hand in the camera view.", bg=PANEL, fg="#ffbd69",
                                            font=("Segoe UI", 10, "bold"), wraplength=335, justify="left")
        self.hand_academy_status.pack(anchor="w", pady=(2, 16))
        tk.Label(left, text="LEARNING ORDER", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 8))
        self.hand_academy_lesson_labels = []
        for idx, (_key, title, _instruction, _verified) in enumerate(HAND_ACADEMY_STEPS):
            label = tk.Label(left, text=f"○  {idx+1}. {title}", bg=PANEL, fg=MUTED,
                             font=("Segoe UI", 9), anchor="w")
            label.pack(fill="x", pady=3)
            self.hand_academy_lesson_labels.append(label)
        tk.Label(left, text="Privacy: camera processing stays local. Only camera name, handedness hint and lesson completion are saved; no photo, video or landmark data is stored.",
                 bg=PANEL, fg="#8492a6", font=("Segoe UI", 8), wraplength=335, justify="left").pack(anchor="w", side="bottom", pady=(18, 4))

        tk.Label(right, text="LIVE MIRRORED CAMERA · LANDMARK VERIFICATION", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))
        self.hand_academy_preview = tk.Label(right, bg="#090b10", fg=MUTED,
                                             text="Waiting for a live webcam frame…", font=("Segoe UI", 11))
        self.hand_academy_preview.pack(fill="both", expand=True)
        footer = tk.Frame(win, bg=DARK, padx=22, pady=12)
        footer.pack(fill="x")
        tk.Label(footer, text="Hold each recognised pose steadily until the progress bar completes.", bg=DARK,
                 fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
        self._button(footer, "Cancel Academy", self._close_hand_academy, outline=True).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", self._close_hand_academy)
        win.lift()
        win.focus_force()
        self._refresh_hand_academy_lesson()
        self.status.config(text="Gesture Academy started. Show one hand naturally to begin.", fg=ACCENT)

    def _close_hand_academy(self):
        """Close the lesson window without marking incomplete training as complete."""
        win = self.hand_academy_window
        self.hand_academy_window = None
        self.hand_academy_preview = None
        self.hand_academy_title = None
        self.hand_academy_instruction = None
        self.hand_academy_status = None
        self.hand_academy_progress = None
        self.hand_academy_hold_started = None
        self.hand_academy_advancing = False
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        if self.hand_mode:
            self.status.config(text="Gesture Academy is incomplete. Start it again to unlock the Hand Smart Board.", fg="#ffbd69")

    def _refresh_hand_academy_lesson(self):
        if self.hand_academy_window is None:
            return
        idx = min(self.hand_academy_step_index, len(HAND_ACADEMY_STEPS)-1)
        _key, title, instruction, _verified = HAND_ACADEMY_STEPS[idx]
        try:
            self.hand_academy_title.config(text=f"{idx+1} / {len(HAND_ACADEMY_STEPS)} · {title}")
            self.hand_academy_instruction.config(text=instruction)
            self.hand_academy_progress.configure(value=0)
            for i, label in enumerate(self.hand_academy_lesson_labels):
                if self.hand_academy_step_verified[i]:
                    label.config(text=f"✓  {i+1}. {HAND_ACADEMY_STEPS[i][1]}", fg=ACCENT)
                elif i == idx:
                    label.config(text=f"●  {i+1}. {HAND_ACADEMY_STEPS[i][1]}", fg=TEXT)
                else:
                    label.config(text=f"○  {i+1}. {HAND_ACADEMY_STEPS[i][1]}", fg=MUTED)
            prompts = (
                "Show one hand in the camera view.",
                "Extend only your index finger.",
                "Bring thumb and index fingertip together.",
                "Hold the V sign: index + middle raised.",
                "Touch thumb to middle; keep index away.",
                "Raise index + middle + ring; fold little finger.",
            )
            self.hand_academy_status.config(text=prompts[idx], fg="#ffbd69")
        except Exception:
            pass

    def _hand_academy_pose_matches(self, step_key, hand_present, pose):
        if not hand_present:
            return False
        if step_key == "verify":
            return True
        if pose is None:
            return False
        if step_key == "index":
            fingers = getattr(pose, "fingers", {}) or {}
            return bool(
                getattr(pose, "index_extended", False)
                and not getattr(pose, "pinch_active", False)
                and not getattr(pose, "zoom_active", False)
                and not getattr(pose, "pan_active", False)
                and not getattr(pose, "menu_active", False)
                and not fingers.get("middle", False)
                and not fingers.get("ring", False)
                and not fingers.get("pinky", False)
            )
        if step_key == "pinch":
            return bool(pose.pinch_active and not pose.zoom_active)
        if step_key == "pan":
            return bool(pose.pan_active)
        if step_key == "zoom":
            return bool(pose.zoom_active)
        if step_key == "menu":
            return bool(pose.menu_active)
        return False

    def _observe_hand_academy(self, hand_present, pose, handedness, timestamp):
        """Advance only when the current camera frame confirms the current gesture."""
        if self.hand_academy_window is None or self.hand_academy_advancing:
            return
        if handedness:
            self._hand_handedness_hint = str(handedness)
        idx = self.hand_academy_step_index
        step_key = HAND_ACADEMY_STEPS[idx][0]
        matched = self._hand_academy_pose_matches(step_key, hand_present, pose)
        if not matched:
            self.hand_academy_hold_started = None
            if self.hand_academy_progress is not None:
                try:
                    self.hand_academy_progress.configure(value=0)
                except Exception:
                    pass
            if self.hand_academy_status is not None:
                text = "No hand detected. Keep one hand fully inside the camera view." if not hand_present else "Pose not verified yet. Follow the instruction and hold it steady."
                try:
                    self.hand_academy_status.config(text=text, fg="#ffbd69")
                except Exception:
                    pass
            return
        if self.hand_academy_hold_started is None:
            self.hand_academy_hold_started = float(timestamp)
        elapsed = max(0.0, float(timestamp) - self.hand_academy_hold_started)
        progress = min(100.0, elapsed / HAND_ACADEMY_HOLD_SECONDS * 100.0)
        if self.hand_academy_progress is not None:
            try:
                self.hand_academy_progress.configure(value=progress)
            except Exception:
                pass
        if self.hand_academy_status is not None:
            try:
                self.hand_academy_status.config(
                    text=f"Pose recognised · hold steady {max(0.0, HAND_ACADEMY_HOLD_SECONDS-elapsed):.1f}s", fg=ACCENT)
            except Exception:
                pass
        if elapsed >= HAND_ACADEMY_HOLD_SECONDS:
            self.hand_academy_step_verified[idx] = True
            self.hand_academy_advancing = True
            try:
                self.hand_academy_status.config(text=HAND_ACADEMY_STEPS[idx][3], fg=ACCENT)
                self.hand_academy_lesson_labels[idx].config(
                    text=f"✓  {idx+1}. {HAND_ACADEMY_STEPS[idx][1]}", fg=ACCENT)
            except Exception:
                pass
            self.root.after(450, self._advance_hand_academy)

    def _advance_hand_academy(self):
        if self.hand_academy_window is None:
            return
        self.hand_academy_advancing = False
        self.hand_academy_hold_started = None
        self.hand_academy_step_index += 1
        if self.hand_academy_step_index >= len(HAND_ACADEMY_STEPS):
            self._complete_hand_academy()
            return
        self._refresh_hand_academy_lesson()

    def _complete_hand_academy(self):
        try:
            self._save_hand_gesture_profile()
        except Exception as exc:
            self._log_runtime_event(f"Could not save Hand Gesture Academy metadata: {type(exc).__name__}: {exc}")
            messagebox.showwarning(
                "Gesture Academy metadata not saved",
                f"All gestures were verified, but the local completion profile could not be saved.\n\n{exc}\n\nThe Board will still open for this session.",
                parent=self.hand_academy_window,
            )
        self.hand_academy_completed_this_run = True
        self.hand_academy_required = False
        win = self.hand_academy_window
        self.hand_academy_window = None
        self.hand_academy_preview = None
        self.hand_academy_title = None
        self.hand_academy_instruction = None
        self.hand_academy_status = None
        self.hand_academy_progress = None
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        self.status.config(text="Gesture Academy complete · all six controls verified. Opening Hand Smart Board…", fg=ACCENT)
        self.root.after(450, self.open_board)

    def _show_hand_academy_preview(self, frame):
        label = self.hand_academy_preview
        if label is None:
            return
        try:
            if not label.winfo_exists():
                return
            w = max(480, label.winfo_width())
            h = max(300, label.winfo_height())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ImageOps.pad(Image.fromarray(rgb), (w, h), method=Image.Resampling.LANCZOS,
                                 color=(9, 11, 16), centering=(0.5, 0.5))
            photo = ImageTk.PhotoImage(image)
            self._hand_academy_photo = photo
            label.configure(image=photo, text="")
        except Exception:
            pass

    def _show_hand_board_preview(self, frame, gesture_text):
        label = self.hand_board_preview
        if label is None:
            return
        try:
            if not label.winfo_exists():
                return
            w = max(280, label.winfo_width())
            h = max(180, label.winfo_height())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ImageOps.pad(Image.fromarray(rgb), (w, h), method=Image.Resampling.LANCZOS,
                                 color=(9, 11, 16), centering=(0.5, 0.5))
            photo = ImageTk.PhotoImage(image)
            self.hand_board_preview_photo = photo
            label.configure(image=photo, text="")
            if self.hand_board_status is not None:
                self.hand_board_status.config(text=f"LIVE GESTURE: {gesture_text}", fg=ACCENT if gesture_text != "NO HAND DETECTED" else "#ffbd69")
            if self.hand_board_gesture_chip is not None:
                self.hand_board_gesture_chip.config(text=f"  {gesture_text}  ")
        except Exception:
            pass

    def _auto_connect_hand_camera(self):
        """Prefer the integrated webcam, with a separately saved hand profile."""
        if not self.hand_mode or self.cap is not None:
            return
        if self.hand_tracker is None:
            message = self._hand_last_camera_error or "MediaPipe hand engine is unavailable."
            self.status.config(text=message, fg="#ff7474")
            self.preview_info.config(text=message)
            return
        self.refresh_devices()
        values = list(self.device_combo["values"])
        selected = self.device_var.get().strip()
        if selected and "droidcam" not in selected.lower():
            self.status.config(text=f"Hand engine ready. Selected {selected.split(':', 1)[1].strip()}; opening webcam…", fg=ACCENT)
            self.connect_camera()
        else:
            self.device_var.set("")
            self.status.config(text="No built-in/USB webcam was enumerated. Connect a webcam, press Refresh, and choose it. DroidCam is not auto-selected in Hand mode.", fg="#ffbd69")
            self.preview_info.config(text="No webcam device name was returned by Windows DirectShow enumeration.")

    def _connect_hand_camera(self):
        if self.hand_tracker is None:
            message = self._hand_last_camera_error or "MediaPipe hand engine is unavailable."
            self.status.config(text=message, fg="#ff7474")
            self.preview_info.config(text=message)
            return
        selected = self.device_var.get().strip()
        if not selected:
            self._auto_connect_hand_camera()
            return
        try:
            idx = int(selected.split(":", 1)[0].strip())
            selected_name = selected.split(":", 1)[1].strip()
            profile = load_hand_camera_profile()
            backend = profile["backend"] if profile and int(profile["index"]) == idx else "DSHOW"
            self.status.config(text=f"Opening hand camera: {selected_name} (index {idx}, {backend})…", fg="#ffbd69")
            self.preview_info.config(text="Opening the selected webcam. Laser calibration is not used in Hand mode.")
            self.root.update_idletasks()
            cap, frame, backend_name = open_camera(idx, backend)
            self._stop_camera_reader()
            self.cap, self.camera_index = cap, idx
            self.camera_name = selected_name
            self.camera_backend = backend_name
            self.frame = frame
            self._start_camera_reader(cap, frame)
            self.camera_confirmed = True
            self.camera_reconnect_pending = False
            self.camera_last_frame_time = time.time()
            self.background_gray = None
            self.laser_point = None
            self.last_point_time = 0.0
            save_hand_camera_profile(idx, "DSHOW" if "DirectShow" in backend_name else backend)
            self.status.config(text=f"HAND CAMERA CONNECTED · {selected_name}\nBackend: {backend_name}", fg=ACCENT)
            self.preview_info.config(text="MediaPipe is tracking one primary hand. No laser background capture or calibration will run.")
            self.root.after(350, self._continue_hand_startup)
        except Exception as exc:
            self._stop_camera_reader()
            self.camera_confirmed = False
            self.status.config(text="Hand camera could not be opened", fg="#ff7474")
            self.preview_info.config(text=f"{type(exc).__name__}: {exc}")
            self._hand_last_camera_error = str(exc)

    def _hand_local_board_point(self):
        """Convert the current hand pointer to board-canvas coordinates."""
        if not self.running_board or self.board_canvas is None:
            return None
        point = self._screen_point()
        if point is None:
            return None
        tk_point = self._desktop_point_to_tk(point)
        x = float(tk_point[0]) - self.board_canvas.winfo_rootx()
        y = float(tk_point[1]) - self.board_canvas.winfo_rooty()
        if 0 <= x < self.board_canvas.winfo_width() and 0 <= y < self.board_canvas.winfo_height():
            return x, y
        return None

    def _finish_hand_stroke(self):
        if self.current_stroke is not None and self.board_canvas is not None:
            now = time.time()
            if self.board_mode == "SMART_WRITE":
                self._queue_smart_write_stroke(self.current_stroke, now)
            self.board_canvas.delete("active-stroke-preview")
            self._redraw_board()
        self.current_stroke = None
        self.last_board_point = None
        self._stroke_filter_time = 0.0

    def _handle_hand_release(self):
        """Pinch release selects a target once; in drawing tools it ends the stroke."""
        if not self.running_board:
            return
        local = self._hand_local_board_point()
        if self.menu_ids and local is not None:
            self._radial_hover(local[0], local[1])
            self._select_radial_item_at_canvas_point(local)
            return
        # Toolbar controls must remain usable regardless of the active drawing tool.
        screen_point = self._screen_point()
        if screen_point is not None and self._laser_invoke_ui_button(screen_point):
            self._finish_hand_stroke()
            return
        if self.board_mode in ("PENCIL", "SMART_WRITE", "HIGHLIGHTER", "ERASER", "SHAPES", "SWIPE"):
            self._finish_hand_stroke()
            if self.board_mode == "SHAPES" and self.shape_start is not None and self.shape_end is not None:
                segments = self._shape_segments(self.shape_start, self.shape_end, self.shape_type)
                if segments:
                    self.strokes.append(segments)
                    self.redo_strokes = []
                    self._redraw_board()
                self.board_canvas.delete("shape-preview")
                self.shape_start = self.shape_end = None
            if self.board_mode == "SWIPE" and self.swipe_start is not None and self.swipe_last is not None:
                self._finish_swipe_gesture()
            return
        if screen_point is not None:
            x, y = screen_point
            self._perform_desktop_click(x, y, button="left")

    def _process_hand_frame(self, frame, timestamp):
        """Run the existing hand_gesture_engine on each NEW webcam frame."""
        if self.hand_tracker is None or self.hand_engine is None:
            return
        mirrored = cv2.flip(frame, 1)
        height, width = mirrored.shape[:2]
        rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
        results = self.hand_tracker.process(rgb)
        hand, center = self.hand_engine.select_primary_hand(
            results.multi_hand_landmarks or [], width, height, self.hand_previous_center)
        was_pinched = self._hand_pinch_active
        was_zooming = self._hand_zoom_active
        if hand is None:
            if timestamp - self.hand_last_seen > 0.18:
                self.hand_pose = None
                self.hand_previous_center = None
                self.hand_consensus.reset()
                self._hand_pinch_active = self.hand_pinch_stabilizer.update(False, force_release=True)
                self._hand_zoom_active = False
                if self.hand_pointer_filter is not None:
                    self.hand_pointer_filter.reset()
                self._hand_last_pointer_canvas = None
                self._hand_pointer_norm = None
                self.laser_point = None
                self.last_point_time = 0.0
                if was_pinched:
                    self._handle_hand_release()
            self._observe_hand_academy(False, None, None, timestamp)
            shown = mirrored
            cv2.putText(shown, "NO HAND DETECTED", (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 80, 255), 2)
            self.hand_gesture_text = "NO HAND DETECTED"
            self._show_preview(shown)
            self._show_hand_academy_preview(shown)
            self._show_hand_board_preview(shown, "NO HAND DETECTED")
            self.preview_info.config(text=f"{self.camera_name} | {width}x{height} | No hand landmarks")
            return

        self.hand_last_seen = timestamp
        self.hand_previous_center = center
        pose = self.hand_engine.classify_hand_pose(
            hand.landmark, width, height, pinch_was_active=was_pinched, zoom_was_active=was_zooming)
        self.hand_pose = pose
        handedness = "Unknown"
        try:
            handedness_items = getattr(results, "multi_handedness", None) or []
            if handedness_items and handedness_items[0].classification:
                handedness = str(handedness_items[0].classification[0].label)
        except Exception:
            pass
        self._observe_hand_academy(True, pose, handedness, timestamp)
        self._hand_pinch_active = self.hand_pinch_stabilizer.update(pose.pinch_active)
        self._hand_zoom_active = bool(pose.zoom_active)
        pointer = self.hand_pointer_filter.update(pose.index_point, timestamp) if pose.pointer_active else None
        if pointer is not None:
            self.laser_point = (float(pointer[0]), float(pointer[1]))
            self._hand_pointer_norm = (
                max(0.0, min(1.0, float(pointer[0]) / max(1, width))),
                max(0.0, min(1.0, float(pointer[1]) / max(1, height))),
            )
            self.last_point_time = timestamp
            self.H = np.eye(3, dtype=np.float32)
        else:
            self.laser_point = None
            self._hand_pointer_norm = None
            self.last_point_time = 0.0

        pan_active, menu_active = self.hand_consensus.update(pose.pan_active, pose.menu_active)
        if self.running_board and self._hand_zoom_active:
            if self._hand_zoom_last_y is not None:
                vertical_delta = self._hand_zoom_last_y - float(pose.wrist_center_y)
                zoom_factor = 1.0 + max(-0.04, min(0.04, vertical_delta * 3.0))
                new_scale = max(0.65, min(2.5, self._hand_zoom_scale * zoom_factor))
                if abs(new_scale - self._hand_zoom_scale) > 0.002:
                    self._hand_zoom_scale = new_scale
                    self._redraw_board()
            self._hand_zoom_last_y = float(pose.wrist_center_y)
        else:
            self._hand_zoom_last_y = None
        local = self._hand_local_board_point() if self.running_board else None
        if self.running_board and local is not None:
            if pan_active:
                if self._hand_last_pointer_canvas is not None:
                    dx = local[0] - self._hand_last_pointer_canvas[0]
                    dy = local[1] - self._hand_last_pointer_canvas[1]
                    if abs(dx) + abs(dy) < 180:
                        self.pan_x += dx
                        self.pan_y += dy
                        self._redraw_board()
                self.last_board_point = None
                self._finish_hand_stroke() if self._hand_pinch_active is False else None
            elif menu_active and not self.menu_ids and timestamp - self._hand_last_menu_trigger > 1.25:
                self._hand_last_menu_trigger = timestamp
                self._finish_hand_stroke()
                self.show_radial_menu(at=local, page="main")
            elif self.menu_ids:
                self._radial_hover(local[0], local[1])
            self._hand_last_pointer_canvas = local

        if was_pinched and not self._hand_pinch_active:
            self._handle_hand_release()

        # Draw landmarks into the dashboard preview so the user can verify that
        # the hand engine is genuinely running (not merely showing camera video).
        try:
            self._hand_mp.solutions.drawing_utils.draw_landmarks(
                mirrored, hand, self._hand_mp.solutions.hands.HAND_CONNECTIONS)
        except Exception:
            pass
        cv2.circle(mirrored, (int(pointer[0]), int(pointer[1])) if pointer else (int(pose.index_point[0]), int(pose.index_point[1])),
                   8, (40, 235, 120), 2)
        self.hand_gesture_text = pose.gesture
        cv2.putText(mirrored, pose.gesture, (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (40, 235, 120), 2)
        self._show_preview(mirrored)
        self._show_hand_academy_preview(mirrored)
        self._show_hand_board_preview(mirrored, pose.gesture)
        self.preview_info.config(text=(
            f"{self.camera_name} | {width}x{height} | {pose.gesture} | "
            f"Pinch: {'ACTIVE' if self._hand_pinch_active else 'released'}"))

    def _tick_hand(self):
        """Hand-mode frame loop; deliberately bypasses all laser detection/calibration."""
        if self.cap is None:
            return
        reader = getattr(self, "camera_reader", None)
        if reader is None:
            return
        frame, frame_seq, frame_stamp, read_failures, read_error = reader.snapshot()
        now = time.time()
        frame_age = float("inf") if not frame_stamp else max(0.0, now - frame_stamp)
        if frame is None or frame_age > CAMERA_FRAME_STALE_SECONDS:
            self.camera_read_failures = max(1, int(read_failures))
            self.camera_last_error = read_error or "No fresh webcam frame has arrived"
            if self.preview is not None:
                self.preview_info.config(text=f"Hand camera waiting for frames: {self.camera_last_error}")
            return
        if frame_seq == self._last_processed_camera_seq:
            return
        self._last_processed_camera_seq = frame_seq
        self.camera_last_frame_time = frame_stamp
        self.camera_read_failures = 0
        self.camera_last_error = ""
        self.frame = frame
        self._process_hand_frame(frame, frame_stamp or now)
        if self.running_board:
            self._board_tick()

    def _find_yolo_weights(self):
        candidates = [
            ROOT / "runs" / "classify" / "runs_airwriter" / "laser_verifier" / "weights" / "best.pt",
            ROOT / "runs_airwriter" / "laser_verifier" / "weights" / "best.pt",
            ROOT / "laser_verifier" / "weights" / "best.pt",
        ]
        return next((p for p in candidates if p.is_file()), None)

    def load_calibration(self):
        try:
            rows = np.loadtxt(CAL_FILE, delimiter=",")
            rows = np.atleast_2d(rows)
            # Support both legacy nine-point profiles and the new five-point
            # quick calibration format. Four non-collinear points are required.
            if rows.ndim == 2 and rows.shape[1] == 4 and rows.shape[0] >= 4:
                return rows.astype(np.float32)
        except Exception:
            pass
        return DEFAULT_CAL.copy()

    @staticmethod
    def make_homography(cal):
        try:
            h, _ = cv2.findHomography(cal[:, :2], cal[:, 2:4], method=0)
            return h if h is not None and np.isfinite(h).all() else None
        except Exception: return None

    def _build_ui(self):
        if self.hand_mode:
            self._build_hand_ui()
            return
        top = tk.Frame(self.root, bg=DARK, padx=24, pady=18); top.pack(fill="x")
        tk.Label(top, text="AIR WRITER", font=("Segoe UI", 24, "bold"), bg=DARK, fg=TEXT).pack(anchor="w")
        tk.Label(top, text="Laser-based touchless mouse · screen mapping · smart board", font=("Segoe UI", 10), bg=DARK, fg=MUTED).pack(anchor="w", pady=(2,0))
        body = tk.Frame(self.root, bg=DARK, padx=22, pady=4); body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=PANEL, padx=18, pady=18, width=330); left.pack(side="left", fill="y", padx=(0,16)); left.pack_propagate(False)
        right = tk.Frame(body, bg=PANEL, padx=14, pady=14); right.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="01  CAMERA CONNECTION", font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w")
        self.device_var = tk.StringVar(value="")
        self.device_combo = ttk.Combobox(left, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill="x", pady=(12,8))
        row = tk.Frame(left, bg=PANEL); row.pack(fill="x")
        self._button(row, "Refresh devices", self.refresh_devices, outline=True).pack(side="left", fill="x", expand=True, padx=(0,5))
        self._button(row, "Connect", self.connect_camera).pack(side="left", fill="x", expand=True, padx=(5,0))
        self._button(left, "Scan OpenCV cameras", self.scan_opencv_cameras, outline=True).pack(fill="x", pady=(7,0))
        self.status = tk.Label(left, text="●  DroidCam not connected", bg=PANEL, fg="#ffbd69", font=("Segoe UI", 9), wraplength=280, justify="left")
        self.status.pack(anchor="w", pady=(12,8))
        self.confirm_btn = self._button(left, "Confirm preview is my phone", self.confirm_phone_preview, outline=True)
        self.confirm_btn.pack(fill="x", pady=(0,16))
        self.confirm_btn.configure(state="disabled")
        tk.Label(left, text="02  LIVE CALIBRATION", font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w")
        tk.Label(left, text="After DroidCam connects, the app captures the background with the laser OFF, then advances through 9 targets when the laser spot stays steady. No keyboard is needed.", bg=PANEL, fg=MUTED, wraplength=285, justify="left", font=("Segoe UI", 9)).pack(anchor="w", pady=(8,10))
        self._button(left, "Restart automatic setup", self.capture_background).pack(fill="x", pady=4)
        self._button(left, "Retry 5-point calibration", self.start_calibration, outline=True).pack(fill="x", pady=4)
        self.cal_status = tk.Label(left, text="Calibration: waiting for DroidCam", bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=280, justify="left")
        self.cal_status.pack(anchor="w", pady=(5,18))
        tk.Label(left, text="03  AIR WRITER BOARD", font=("Segoe UI", 11, "bold"), bg=PANEL, fg=ACCENT).pack(anchor="w")
        self.open_board_btn = self._button(left, "Open Smart Board · Laser Click", self.open_board)
        self.open_board_btn.pack(fill="x", pady=(10,4))
        self._button(left, "Pause / resume laser cursor", self.toggle_mouse, outline=True).pack(fill="x", pady=4)
        self.mouse_enabled = False
        tk.Label(left, text=f"Touchless workflow: after calibration, the laser moves the cursor. In Smart Board, draw a capital Z: top line left-to-right, diagonal down-left, then bottom line left-to-right to open the radial menu. Release the laser and wait for MENU READY, then aim at an item and switch OFF to select it. Aim at a toolbar button and switch the laser OFF to activate it. Pen/Eraser/Shapes then perform real board actions while the cursor reticle stays visible. No keyboard is needed for normal use.", bg=PANEL, fg=MUTED, wraplength=285, justify="left", font=("Segoe UI", 8)).pack(anchor="w", pady=(14,0))
        tk.Label(right, text="LIVE DROIDCAM PREVIEW", bg=PANEL, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0,10))
        self.preview = tk.Label(right, bg="#090b10", fg=MUTED, text="Connect DroidCam to start the live preview", font=("Segoe UI", 11))
        self.preview.pack(fill="both", expand=True)
        yolo_text = f"YOLO verifier: {self.yolo_path.name}" if self.yolo_model is not None and self.yolo_path else "YOLO verifier: not loaded (OpenCV candidate scoring only)"
        self.preview_info = tk.Label(right, text=f"Waiting for a verified DroidCam frame  |  {yolo_text}", bg=PANEL, fg=MUTED, anchor="w", font=("Consolas", 9))
        self.preview_info.pack(fill="x", pady=(10,0))
        footer = tk.Label(self.root, text="AIR WRITER SMART BOARD  •  OpenCV + NumPy  •  Local processing", bg=DARK, fg="#748095", font=("Segoe UI", 8), pady=8)
        footer.pack(fill="x")

    def _button(self, parent, text, command, outline=False):
        return tk.Button(parent, text=text, command=command, bg=(PANEL if outline else ACCENT), fg=(TEXT if outline else "#071716"), activebackground="#87eadf", activeforeground="#071716", relief="flat", bd=0, padx=10, pady=9, cursor="hand2", font=("Segoe UI", 9, "bold"), highlightthickness=1 if outline else 0, highlightbackground="#3a4557")

    def refresh_devices(self):
        self.camera_devices = enumerate_cameras()
        if self.hand_mode:
            values = [f"{i}: {name}" for i, name in self.camera_devices]
            self.device_combo["values"] = values
            profile = load_hand_camera_profile()
            pick = None
            if profile is not None:
                pick = next((pos for pos, (idx, name) in enumerate(self.camera_devices)
                             if idx == profile["index"] and "droidcam" not in name.lower()), None)
            if pick is None:
                preferred_words = ("integrated", "built-in", "built in", "laptop", "webcam", "hd camera", "user facing")
                pick = next((pos for pos, (_idx, name) in enumerate(self.camera_devices)
                             if any(word in name.lower() for word in preferred_words)
                             and "droidcam" not in name.lower()), None)
            if pick is None:
                pick = next((pos for pos, (_idx, name) in enumerate(self.camera_devices)
                             if "droidcam" not in name.lower()), None)
            # The supplied ZIP includes a confirmed hand-camera profile. If
            # DirectShow name enumeration temporarily fails, still offer that
            # exact saved index/backend rather than silently choosing another
            # camera or returning an empty selector.
            if pick is None and profile is not None:
                label = f"{profile['index']}: Saved Hand Camera ({profile['backend']})"
                values.append(label)
                self.device_combo["values"] = values
                self.device_combo.current(len(values) - 1)
                self.status.config(
                    text=(f"Camera names were unavailable; using saved Hand camera profile "
                          f"index {profile['index']} / {profile['backend']}"),
                    fg="#ffbd69")
                return
            if pick is not None:
                self.device_combo.current(pick)
                self.status.config(text=f"Found {len(values)} camera(s). Hand mode selected {self.camera_devices[pick][1]}.", fg=ACCENT)
            else:
                self.device_var.set("")
                self.status.config(text="No built-in/USB webcam found by Windows. Connect a webcam, press Refresh, then select it. DroidCam is not auto-selected in Hand mode.", fg="#ffbd69")
            return
        # Prefer OS names when available. If DirectShow enumeration does not
        # expose the virtual device, offer the visually confirmed test profile.
        droid = [(i, n) for i, n in self.camera_devices if "droidcam" in n.lower()]
        values = [f"{i}: {n}" for i, n in droid]
        self.profile_combo_labels = {}
        profile = self.camera_profile or load_camera_profile()
        if profile is not None and not any(i == profile["index"] for i, _ in droid):
            label = f'{profile["index"]}: DroidCam (tested profile, {profile["backend"]})'
            values.append(label)
            self.profile_combo_labels[label] = profile
        self.device_combo["values"] = values
        if values:
            preferred_idx = profile["index"] if profile else None
            pick = next((pos for pos, value in enumerate(values)
                         if value.split(":", 1)[0].strip().isdigit()
                         and int(value.split(":", 1)[0].strip()) == preferred_idx), 0)
            self.device_combo.current(pick)
            details = f'Camera profile: {profile["backend"]} index {profile["index"]}. ' if profile else ""
            self.status.config(text=f"●  Found {len(values)} usable DroidCam option(s). {details}Select and connect.", fg=ACCENT)
        else:
            self.device_var.set("")
            self.status.config(text="●  DroidCam Video not found. Start the phone app and PC client, then click Refresh. The app will not select the laptop webcam.", fg="#ffbd69")

    def scan_opencv_cameras(self):
        """In-app version of test_droidcam_opencv.py.

        Walk through MSMF then DSHOW camera indices, show each live preview,
        and save the backend/index only when the user confirms the phone feed.
        """
        # Release the current stream first; virtual webcam drivers may report
        # busy when two VideoCapture objects target the same device.
        self._stop_camera_reader()
        self.camera_confirmed = False
        self.background_gray = None
        self.frame = None
        self.confirm_btn.configure(state="disabled")
        self.status.config(text="Scanning camera indices…", fg="#ffbd69")

        backends = []
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("MSMF", cv2.CAP_MSMF))
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DSHOW", cv2.CAP_DSHOW))
        candidates = [(label, backend, index) for label, backend in backends for index in range(5)]
        if not candidates:
            messagebox.showerror("Camera scan unavailable", "This OpenCV build has no MSMF or DSHOW camera backend.")
            return

        win = tk.Toplevel(self.root)
        win.title("Find and identify DroidCam")
        win.geometry("850x650")
        win.configure(bg=DARK)
        win.transient(self.root)
        tk.Label(
            win,
            text="Find your phone's live video. Click 'Use this phone camera' only when the preview is from your phone.",
            bg=DARK, fg=TEXT, wraplength=800, justify="left", font=("Segoe UI", 10), padx=16, pady=12
        ).pack(fill="x")
        info = tk.Label(win, text="Searching…", bg=PANEL, fg=ACCENT, font=("Consolas", 11), padx=10, pady=8)
        info.pack(fill="x", padx=12, pady=(0, 8))
        preview = tk.Label(win, text="Looking for a camera that returns frames…", bg="#090b10", fg=MUTED)
        preview.pack(fill="both", expand=True, padx=12, pady=4)
        buttons = tk.Frame(win, bg=DARK, padx=12, pady=12)
        buttons.pack(fill="x")
        state = {"position": -1, "cap": None, "after_id": None, "photo": None, "has_frame": False, "closing": False}

        use_btn = self._button(buttons, "Use this phone camera", lambda: use_candidate())
        use_btn.pack(side="right", padx=(6, 0))
        use_btn.configure(state="disabled")
        self._button(buttons, "Next camera", lambda: load_candidate(state["position"] + 1), outline=True).pack(side="right", padx=6)
        self._button(buttons, "Previous camera", lambda: load_candidate(max(0, state["position"] - 1)), outline=True).pack(side="right", padx=6)
        self._button(buttons, "Cancel", lambda: close_scanner(), outline=True).pack(side="left")

        def release_candidate():
            after_id = state.get("after_id")
            if after_id is not None:
                try:
                    win.after_cancel(after_id)
                except Exception:
                    pass
                state["after_id"] = None
            cap = state.get("cap")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                state["cap"] = None

        def close_scanner():
            if state["closing"]:
                return
            state["closing"] = True
            release_candidate()
            try:
                win.destroy()
            except Exception:
                pass
            self.status.config(text="Camera scan closed. Refresh or connect the selected DroidCam device.", fg="#ffbd69")

        def update_preview():
            if state["closing"] or not win.winfo_exists():
                return
            cap = state.get("cap")
            if cap is None:
                return
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                info.config(text="Video stopped. Choose Next camera or cancel.", fg="#ff7474")
                state["has_frame"] = False
                use_btn.configure(state="disabled")
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((800, 470), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                state["photo"] = photo
                preview.configure(image=photo, text="")
                state["has_frame"] = True
                use_btn.configure(state="normal")
            state["after_id"] = win.after(35, update_preview)

        def load_candidate(start_pos):
            release_candidate()
            state["has_frame"] = False
            use_btn.configure(state="disabled")
            if start_pos >= len(candidates):
                state["position"] = len(candidates)
                preview.configure(image="", text="No more camera feeds. Check the DroidCam Windows client and click Refresh devices.")
                state["photo"] = None
                info.config(text="Scan finished — no additional live previews.", fg="#ffbd69")
                return
            # Match the standalone test: try each backend/index, skip devices
            # that cannot open or deliver a frame, then preview the next feed.
            for pos in range(max(0, start_pos), len(candidates)):
                backend_name, backend_value, index = candidates[pos]
                info.config(text=f"Testing {backend_name}, camera index {index}…", fg=ACCENT)
                win.update_idletasks()
                cap = None
                try:
                    cap = cv2.VideoCapture(index, backend_value)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    frame = None
                    for _ in range(12):
                        ok, candidate_frame = cap.read()
                        if ok and candidate_frame is not None and candidate_frame.size:
                            frame = candidate_frame
                            break
                        time.sleep(0.08)
                    if frame is None:
                        cap.release()
                        continue
                    state["position"] = pos
                    state["cap"] = cap
                    info.config(text=f"Preview: backend={backend_name}, index={index}  |  If this is your phone, select Use this phone camera.", fg=ACCENT)
                    preview.configure(text="")
                    use_btn.configure(state="normal")
                    update_preview()
                    return
                except Exception as exc:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    info.config(text=f"Camera scan warning: {exc}", fg="#ff7474")
            state["position"] = len(candidates)
            preview.configure(image="", text="No camera preview was found. Reconnect DroidCam, then try again.")
            state["photo"] = None
            info.config(text="No live camera frames detected.", fg="#ff7474")

        def use_candidate():
            if not state["has_frame"] or state["position"] < 0 or state["position"] >= len(candidates):
                messagebox.showwarning("No live preview", "Wait until a live camera frame appears.", parent=win)
                return
            backend_name, _backend_value, index = candidates[state["position"]]
            try:
                CAMERA_PROFILE_FILE.write_text(f"backend={backend_name}\nindex={index}\n", encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Could not save camera profile", str(exc), parent=win)
                return
            self.camera_profile = {"backend": backend_name, "index": index}
            release_candidate()
            state["closing"] = True
            try:
                win.destroy()
            except Exception:
                pass
            self.refresh_devices()
            self.status.config(text=f"Saved camera profile: {backend_name}, index {index}. Reconnecting…", fg=ACCENT)
            self.root.after(150, self.connect_camera)

        win.protocol("WM_DELETE_WINDOW", close_scanner)
        load_candidate(0)

    def _stop_camera_reader(self):
        """Stop the sole frame reader and release its exact camera handle."""
        reader = getattr(self, "camera_reader", None)
        self.camera_reader = None
        cap = self.cap
        self.cap = None
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
        elif cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._last_processed_camera_seq = -1

    def _start_camera_reader(self, cap, first_frame=None):
        """Start one background reader after open_camera() has returned a real frame."""
        # connect_camera() has already stopped/released any previous stream. Do
        # not call _stop_camera_reader() here: self.cap already refers to *cap*.
        old_reader = getattr(self, "camera_reader", None)
        if old_reader is not None and getattr(old_reader, "capture", None) is not cap:
            try:
                old_reader.stop()
            except Exception:
                pass
        self.cap = cap
        self.camera_reader = LatestCameraFrameReader(cap, first_frame)
        self._last_processed_camera_seq = 0
        self.camera_last_frame_time = time.time()
        self.camera_read_failures = 0
        self.camera_last_error = ""

    def _show_raw_calibration_frame(self, state, frame):
        """Show the live phone image before detector/tracker code runs.

        Thus a detector exception cannot leave the calibration inset on its initial
        black 'Waiting for live DroidCam' placeholder.
        """
        if not state or state.get("closing") or frame is None:
            return
        label = state.get("camera_label")
        if label is None or not label.winfo_exists():
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            preview_size = tuple(map(int, state.get("camera_preview_size", (430, 320))))
            # Pad scales both up and down to the available inset while preserving
            # aspect ratio; this avoids a tiny 50x20 camera image in a large panel.
            image = ImageOps.pad(
                image, preview_size, method=Image.Resampling.LANCZOS,
                color=(5, 7, 11), centering=(0.5, 0.5)
            )
            photo = ImageTk.PhotoImage(image)
            state["camera_photo"] = photo
            label.configure(image=photo, text="")
            mask_label = state.get("mask_label")
            if mask_label is not None and mask_label.winfo_exists() and state.get("mask_photo") is None:
                mask_label.configure(text="Waiting for detector output")
        except Exception as exc:
            try:
                label.configure(image="", text=f"Camera preview error: {type(exc).__name__}: {exc}")
                state["camera_photo"] = None
            except Exception:
                pass

    def _mark_calibration_camera_stale(self, state, message):
        if not state:
            return
        try:
            camera_label = state.get("camera_label")
            if camera_label is not None and camera_label.winfo_exists():
                camera_label.configure(image="", text="Waiting for fresh DroidCam frames…")
                state["camera_photo"] = None
            mask_label = state.get("mask_label")
            if mask_label is not None and mask_label.winfo_exists():
                mask_label.configure(image="", text="No fresh detector frame")
                state["mask_photo"] = None
            status = state.get("monitor_status")
            if status is not None and status.winfo_exists():
                status.configure(text=message, fg="#ff8585")
        except Exception:
            pass

    def _auto_connect_saved_camera(self):
        """Keep trying the previously verified DroidCam profile until it is ready."""
        if self.auto_connect_started or self.cap is not None or self.camera_reconnect_pending:
            return
        self.auto_connect_attempts += 1
        self.camera_profile = load_camera_profile() or self.camera_profile
        if not self.camera_profile:
            self.status.config(text="●  No saved DroidCam profile. Use Scan OpenCV cameras once to identify your phone.", fg="#ffbd69")
            return

        # Retain the fail-closed identity check: never guess a camera index when
        # Windows does not currently enumerate a DroidCam device by name.
        enumerated = enumerate_cameras()
        if not any("droidcam" in name.lower() for _, name in enumerated):
            self.status.config(
                text="Waiting for Windows DroidCam Video… Start the phone stream and the Windows DroidCam client. The laptop webcam will not be selected.",
                fg="#ffbd69")
            self.preview_info.config(text="Waiting for the saved DroidCam profile to become available; automatic retry is active.")
            self.root.after(CAMERA_RETRY_INTERVAL_MS, self._auto_connect_saved_camera)
            return

        self.refresh_devices()
        values = list(self.device_combo["values"])
        if not values:
            self.status.config(text="Waiting for the DroidCam virtual camera… automatic retry is active.", fg="#ffbd69")
            self.root.after(CAMERA_RETRY_INTERVAL_MS, self._auto_connect_saved_camera)
            return
        wanted = int(self.camera_profile["index"])
        pick = next((i for i, value in enumerate(values)
                     if value.split(":", 1)[0].strip().isdigit()
                     and int(value.split(":", 1)[0].strip()) == wanted), 0)
        self.device_combo.current(pick)
        self.auto_connect_started = True
        self.connect_camera()

    def connect_camera(self):
        if self.hand_mode:
            self._connect_hand_camera()
            return
        self._stop_camera_reader()
        selected = self.device_var.get().strip()
        if not selected:
            self.refresh_devices()
            self.status.config(text="●  DroidCam not available. Start the phone and Windows client, then refresh devices.", fg="#ff7474")
            return
        try:
            idx = int(selected.split(":", 1)[0].strip())
            selected_name = selected.split(":", 1)[1].strip()
            profile = self.profile_combo_labels.get(selected)
            saved = profile or self.camera_profile or load_camera_profile()
            if profile is not None:
                idx = int(profile["index"])
                preferred_backend = str(profile["backend"]).upper()
                name = f"DroidCam (tested profile: {preferred_backend}, index {idx})"
            else:
                name = selected_name
                if "droidcam" not in name.lower():
                    raise RuntimeError("Selected device is not named DroidCam. The laptop webcam will not be used.")
                if saved is not None:
                    idx = int(saved["index"])
                    preferred_backend = str(saved["backend"]).upper()
                    name = f"{selected_name} (tested profile: {preferred_backend}, index {idx})"
                else:
                    preferred_backend = "MSMF"
            self.status.config(text=f"Connecting to DroidCam using {preferred_backend}, index {idx}…", fg="#ffbd69")
            self.preview_info.config(text="Opening the saved DroidCam device; no laptop-camera fallback is allowed.")
            self.root.update_idletasks()
            cap, frame, backend = open_camera(idx, preferred_backend)
            self.cap, self.camera_index, self.camera_name, self.camera_backend = cap, idx, name, backend
            self.frame = frame
            self._start_camera_reader(cap, frame)
            self.camera_read_failures = 0
            self.camera_reconnect_pending = False
            self.camera_last_frame_time = time.time()
            self.camera_last_error = ""
            # A saved profile was visually confirmed by the camera scanner, so no manual confirmation is needed.
            self.camera_confirmed = bool(saved is not None and ("droidcam" in name.lower() or profile is not None))
            self.confirm_btn.configure(state="disabled" if self.camera_confirmed else "normal")
            self.status.config(text=f"●  DROIDCAM CONNECTED — {name}\nBackend: {backend} | index: {idx}", fg=ACCENT)
            self.preview_info.config(text="Automatic setup starts shortly. Keep the laser OFF during background capture.")
            self.background_gray = None
            self.laser_point = None
            self.laser_mask = None
            self.laser_score = 0.0
            self.smoothed = None
            self._pending_jump_point = None
            self._pending_jump_count = 0
            if self.camera_confirmed:
                self.root.after(900, self._begin_automatic_setup)
            else:
                self.status.config(text="●  Preview opened. Verify it is your phone, then press Confirm once to trust this profile.", fg="#ffbd69")
        except Exception as exc:
            self.cap = None
            self.camera_confirmed = False
            self.status.config(text="●  DroidCam connection failed — no fallback used", fg="#ff7474")
            self.preview_info.config(text=str(exc))
            self.camera_last_frame_time = 0.0
            self.camera_last_error = str(exc)
            # Keep retrying this verified DroidCam profile; never switch to another camera.
            if self.camera_profile is not None:
                self.auto_connect_started = False
                self.root.after(CAMERA_RETRY_INTERVAL_MS, self._auto_connect_saved_camera)

    def _schedule_camera_reconnect(self, reason="DroidCam frame stream stopped"):
        """Release a dead stream and reconnect only the saved, verified phone profile."""
        if self.camera_reconnect_pending:
            return
        self.camera_reconnect_pending = True
        self.camera_confirmed = False
        self.camera_last_frame_time = 0.0
        self.camera_last_error = str(reason or "DroidCam frame stream stopped")
        self.auto_connect_started = False
        self.background_gray = None
        self.auto_background_samples = []
        self.auto_phase = "idle"
        self.laser_point = None
        self.laser_score = 0.0
        self.laser_mask = None
        self.smoothed = None
        self._pending_jump_point = None
        self._pending_jump_count = 0

        self._stop_camera_reader()

        message = "DROIDCAM FRAME LOST — reconnecting to the saved phone camera. Calibration is paused."
        self.status.config(text="●  DroidCam stream interrupted — reconnecting automatically", fg="#ff7474")
        self.preview_info.config(text=f"Camera reconnect pending: {self.camera_last_error}")
        state = self.calibration_state
        if state is not None:
            try:
                if state.get("camera_label") is not None and state["camera_label"].winfo_exists():
                    state["camera_label"].configure(image="", text="DroidCam disconnected — reconnecting…")
                    state["camera_photo"] = None
                if state.get("mask_label") is not None and state["mask_label"].winfo_exists():
                    state["mask_label"].configure(image="", text="Waiting for camera frames")
                    state["mask_photo"] = None
                if state.get("monitor_status") is not None and state["monitor_status"].winfo_exists():
                    state["monitor_status"].configure(text=message, fg="#ff8585")
            except Exception:
                pass
        self.root.after(CAMERA_RETRY_INTERVAL_MS, self._attempt_camera_reconnect)

    def _attempt_camera_reconnect(self):
        """Resume the saved DroidCam connection attempt after a stream failure."""
        self.camera_reconnect_pending = False
        if self.cap is not None:
            return
        self.auto_connect_started = False
        self.camera_profile = load_camera_profile() or self.camera_profile
        if not self.camera_profile:
            self.status.config(text="No saved DroidCam profile is available. Scan cameras and select your phone again.", fg="#ff7474")
            return
        self._auto_connect_saved_camera()

    def confirm_phone_preview(self):
        if (self.cap is None or self.frame is None or not self.cap.isOpened()
                or not self.camera_last_frame_time
                or time.time() - self.camera_last_frame_time > CAMERA_FRAME_STALE_SECONDS):
            messagebox.showwarning("No live preview", "DroidCam has not delivered a fresh frame yet. Wait for the live phone preview before confirming."); return
        if "droidcam" not in self.camera_name.lower():
            messagebox.showerror("Wrong camera", "The selected Windows device name is not DroidCam. Disconnect and select DroidCam Video."); return
        self.camera_confirmed=True
        self.confirm_btn.configure(state="disabled")
        self.status.config(text=f"●  PHONE PREVIEW CONFIRMED — {self.camera_name}\nBackend: {self.camera_backend}",fg=ACCENT)
        self.preview_info.config(text="Phone preview confirmed. Automatic background capture and calibration will start now.")
        self.root.after(350, self._begin_automatic_setup)

    def _begin_automatic_setup(self):
        if self.cap is None or not self.cap.isOpened() or not self.camera_confirmed:
            return
        if (not self.camera_last_frame_time
                or time.time() - self.camera_last_frame_time > CAMERA_FRAME_STALE_SECONDS):
            self.cal_status.config(text="Waiting for a fresh DroidCam frame. Calibration will begin only after live video returns.", fg="#ffbd69")
            self._schedule_camera_reconnect("No fresh DroidCam frame before automatic setup")
            return
        if self.calibration_state is not None:
            try:
                self.calibration_state["wizard"].destroy()
            except Exception:
                pass
            self.calibration_state = None
        if self.H is not None:
            self.calibration_guide_H = self.H.copy()
        self.H = None  # never activate mouse control with an old/stale mapping
        self.auto_phase = "warmup"
        self.auto_phase_deadline = time.time() + 4.0
        self.auto_background_samples = []
        self.auto_background_last_sample = 0.0
        self.background_gray = None
        self.laser_point = None
        self.laser_mask = None
        self.laser_score = 0.0
        self.smoothed = None
        self._pending_jump_point = None
        self._pending_jump_count = 0
        self.mouse_enabled = False
        self.running_board = False
        self.cal_status.config(text="Automatic setup: keep the laser OFF. Background capture begins after a 4-second countdown.", fg="#ffbd69")
        self.status.config(text="●  DroidCam connected — preparing automatic background capture", fg=ACCENT)
        self.preview_info.config(text="LASER OFF for 4 seconds, then keep the scene still while 30 background frames are captured.")

    def _advance_automatic_setup(self, frame, now):
        """Non-blocking 4-second warmup and 30-frame background capture."""
        if self.auto_phase == "warmup":
            remaining = max(0, int(math.ceil(self.auto_phase_deadline - now)))
            self.status.config(text=f"●  Background capture starts in {remaining}s — laser OFF", fg="#ffbd69")
            self.preview_info.config(text=f"Laser OFF · automatic background capture starts in {remaining} seconds.")
            if now >= self.auto_phase_deadline:
                self.auto_phase = "background"
                self.auto_background_samples = []
                self.auto_background_last_sample = 0.0
            return
        if self.auto_phase != "background":
            return
        # Sample at 20 Hz; this avoids 30 near-identical frames from a fast tick loop.
        if now - self.auto_background_last_sample >= 0.05:
            self.auto_background_samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy())
            self.auto_background_last_sample = now
        count = len(self.auto_background_samples)
        self.status.config(text=f"●  Capturing background {count}/30 — laser OFF", fg="#ffbd69")
        self.preview_info.config(text=f"Background capture {count}/30. Keep the laser OFF and camera still.")
        if count >= 30:
            stack = np.stack(self.auto_background_samples[:30], axis=0)
            self.background_gray = np.median(stack, axis=0).astype(np.uint8)
            self.auto_background_samples = []
            self.auto_phase = "calibrating"
            self.cal_status.config(text="Background ready. Starting quick 5-point laser calibration automatically.", fg=ACCENT)
            self.preview_info.config(text="Background ready — starting laser-only 5-point calibration.")
            self.root.after(400, self.start_calibration)

    def capture_background(self):
        """Recovery button: restart the automatic no-dialog setup."""
        self._begin_automatic_setup()

    def _paint_calibration_target(self, state, progress=0.0, message=None, captured=False):
        """Draw clear calibration targets over an always-live DroidCam backdrop."""
        wizard, canvas = state["wizard"], state["canvas"]
        if not wizard.winfo_exists():
            return
        w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if w < 300 or h < 200:
            w, h = max(1, wizard.winfo_screenwidth()), max(1, wizard.winfo_screenheight())
        canvas.delete("cal-ui")

        # Small corner HUDs only: do not put labels across top/bottom targets.
        canvas.create_rectangle(0, 0, w, 48, fill="#142235", outline="", tags="cal-ui")
        canvas.create_rectangle(0, h-40, w, h, fill="#142235", outline="", tags="cal-ui")
        canvas.create_text(20, 23, anchor="w", text="AIR WRITER  /  LIVE DROIDCAM",
                           fill=ACCENT, font=("Segoe UI", 14, "bold"), tags="cal-ui")
        idx = int(state.get("idx", 0))
        total_targets = len(state.get("target_fracs", ()))
        canvas.create_text(w-20, 23, anchor="e", text=("CALIBRATION VERIFIED" if idx >= total_targets else f"TARGET {idx+1} / {total_targets}"),
                           fill="#4cf0a0" if idx >= total_targets else TEXT,
                           font=("Segoe UI", 13, "bold"), tags="cal-ui")
        if idx >= total_targets:
            canvas.create_rectangle(w*0.22, h*0.43, w*0.78, h*0.57,
                                    fill="#142235", outline=ACCENT, width=2, tags="cal-ui")
            canvas.create_text(w/2, h/2-12, text="CALIBRATION VERIFIED", fill=ACCENT,
                               font=("Segoe UI", 25, "bold"), tags="cal-ui")
            canvas.create_text(w/2, h/2+24, text="Cursor mode is starting automatically", fill=TEXT,
                               font=("Segoe UI", 13), tags="cal-ui")
            return

        fx, fy = state["target_fracs"][idx]
        x, y = int(w*fx), int(h*fy)
        state["target_screen"] = (x, y)
        r = 25
        # Thick target with a halo remains easy to see over both light and dark video.
        canvas.create_oval(x-r-20, y-r-20, x+r+20, y+r+20, fill="#ffffff", outline="#142235", width=4, tags="cal-ui")
        canvas.create_oval(x-r-9, y-r-9, x+r+9, y+r+9,
                           fill="#fff2bf", outline=ACCENT if progress > 0 else "#e6a400", width=3, tags="cal-ui")
        if progress > 0:
            canvas.create_arc(x-r-20, y-r-20, x+r+20, y+r+20, start=90,
                              extent=-360*min(1.0, progress), style="arc",
                              outline="#4cf0a0", width=7, tags="cal-ui")
        color = "#4cf0a0" if captured else "#ffd166"
        canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, width=3, tags="cal-ui")
        canvas.create_line(x-38, y, x+38, y, fill="#142235", width=2, tags="cal-ui")
        canvas.create_line(x, y-38, x, y+38, fill="#142235", width=2, tags="cal-ui")

        # The old map is a guide only: a green predicted cursor should be aligned
        # with this yellow target before this target's sample is accepted.
        point = state.get("last_detection_point") if state.get("fresh_spot") else None
        predicted, error_px = self._calibration_guide_position(point, (w, h)) if point is not None else (None, None)
        state["alignment_error_px"] = error_px
        if predicted is not None:
            gx, gy = int(round(predicted[0])), int(round(predicted[1]))
            # Outline plus crosshair makes the guide visible even over bright video.
            canvas.create_oval(gx-16, gy-16, gx+16, gy+16, outline="#06140e", width=7, tags="cal-ui")
            canvas.create_oval(gx-16, gy-16, gx+16, gy+16, outline="#36ff92", width=3, dash=(5, 3), tags="cal-ui")
            canvas.create_line(gx-25, gy, gx+25, gy, fill="#06140e", width=5, tags="cal-ui")
            canvas.create_line(gx, gy-25, gx, gy+25, fill="#06140e", width=5, tags="cal-ui")
            canvas.create_line(gx-25, gy, gx+25, gy, fill="#36ff92", width=2, tags="cal-ui")
            canvas.create_line(gx, gy-25, gx, gy+25, fill="#36ff92", width=2, tags="cal-ui")
            canvas.create_text(gx+28, gy-22, anchor="w", text="ESTIMATED CURSOR", fill="#36ff92",
                               font=("Segoe UI", 9, "bold"), tags="cal-ui")
            if error_px is not None and error_px < 700:
                canvas.create_line(x, y, gx, gy, fill="#36ff92" if error_px <= state.get("alignment_tolerance_px", 320) else "#ffd166",
                                   width=2, dash=(4, 5), tags="cal-ui")

        fresh = bool(state.get("fresh_spot", False))
        score = float(state.get("last_detection_score", 0.0))
        if not fresh:
            footer = "LIVE DROIDCAM PANEL  •  NO VERIFIED RED LASER SPOT  •  target capture is paused"
            footer_color = "#ffb4b4"
        else:
            tol = float(state.get("alignment_tolerance_px", 320.0))
            if error_px is None:
                footer = f"LASER LOCKED  •  score {score:.2f}  •  Aim the physical red dot at the yellow target, then hold steady"
                footer_color = "#d8ffe9"
            elif error_px <= tol:
                footer = f"SPOT VERIFIED  •  old-map guide is {error_px:.0f}px from target  •  aim the physical dot on yellow and hold steady"
                footer_color = "#d8ffe9"
            else:
                footer = f"LASER LOCKED  •  old-map guide offset {error_px:.0f}px (reference only)  •  aim the physical dot at yellow target"
                footer_color = "#ffe3a8"
        canvas.create_text(20, h-20, anchor="w", text=footer,
                           fill=footer_color, font=("Segoe UI", 10, "bold"), tags="cal-ui")
        self._place_calibration_monitor(state)

    def _calibration_guide_position(self, camera_point, canvas_size=None):
        """Project current camera dot through the last known map for visual alignment guidance."""
        guide = getattr(self, "calibration_guide_H", None)
        if camera_point is None or guide is None:
            return None, None
        try:
            src = np.asarray([[[float(camera_point[0]), float(camera_point[1])]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, guide)[0, 0]
            if not np.isfinite(dst).all():
                return None, None
            if canvas_size is None:
                if self.calibration_state is None:
                    return None, None
                canvas = self.calibration_state["canvas"]
                canvas_size = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
            w, h = canvas_size
            point = (float(dst[0]) * w / 1920.0, float(dst[1]) * h / 1080.0)
            state = self.calibration_state
            target = state.get("target_screen") if state else None
            error = math.hypot(point[0] - target[0], point[1] - target[1]) if target else None
            return point, error
        except Exception:
            return None, None

    def _place_calibration_monitor(self, state):
        """Place a large live-camera diagnostic panel in the corner farthest from the target."""
        panel = state.get("monitor")
        if panel is None or not panel.winfo_exists():
            return
        wizard = state["wizard"]
        sw = max(1, wizard.winfo_screenwidth())
        sh = max(1, wizard.winfo_screenheight())
        # Keep the diagnostic panel below half of either screen dimension so a
        # corner remains free for every calibration target, including the centre.
        panel_w = min(720, max(440, int(sw * 0.48)))
        panel_h = min(400, max(300, int(sh * 0.48)))
        idx = max(0, min(8, int(state.get("idx", 0))))
        fx, fy = state["target_fracs"][idx]
        tx, ty = fx * sw, fy * sh
        margin = 12
        candidates = [
            (margin, margin),
            (max(margin, sw-panel_w-margin), margin),
            (margin, max(margin, sh-panel_h-44)),
            (max(margin, sw-panel_w-margin), max(margin, sh-panel_h-44)),
        ]
        scored = []
        for order, (px, py) in enumerate(candidates):
            dx = max(px-tx, 0.0, tx-(px+panel_w))
            dy = max(py-ty, 0.0, ty-(py+panel_h))
            distance = math.hypot(dx, dy)
            # Deterministic variety when all four corners are equally good.
            tie = (order + idx) % 4
            scored.append((distance, tie, px, py))
        _, _, px, py = max(scored)
        cam_w = min(430, max(250, panel_w - 290))
        cam_h = min(320, max(180, panel_h - 92))
        mask_w = max(180, panel_w - cam_w - 28)
        state["camera_preview_size"] = (int(cam_w), int(cam_h))
        state["mask_preview_size"] = (int(mask_w), int(cam_h))
        panel.place(x=int(px), y=int(py), width=int(panel_w), height=int(panel_h))
        panel.lift()

    def _create_calibration_monitor(self, wizard, state):
        """Show the camera image and exact detector mask during calibration.

        This is deliberately visible during every target, so a black / undetected
        frame cannot masquerade as successful calibration progress.
        """
        panel = tk.Frame(wizard, bg="#111722", highlightbackground=ACCENT, highlightthickness=2)
        state["monitor"] = panel
        tk.Label(panel, text="LIVE DROIDCAM  |  YELLOW = TARGET GUIDE  ·  GREEN/CYAN = VERIFIED LASER",
                 bg="#111722", fg=ACCENT, font=("Segoe UI", 9, "bold")).pack(fill="x", padx=7, pady=(5, 2))
        row = tk.Frame(panel, bg="#111722")
        row.pack(fill="x", padx=4)
        camera_label = tk.Label(row, bg="#05070b", fg=TEXT, text="Waiting for live DroidCam…", width=48, height=18)
        camera_label.pack(side="left", padx=(0, 5))
        mask_label = tk.Label(row, bg="#05070b", fg=TEXT, text="No detector mask", width=31, height=18)
        mask_label.pack(side="left")
        status = tk.Label(panel, text="Waiting for a verified red laser spot — no samples will be captured yet.",
                          bg="#111722", fg="#ff8585", font=("Segoe UI", 8, "bold"),
                          wraplength=620, justify="left")
        status.pack(fill="x", padx=6, pady=(2, 3))
        state["camera_label"] = camera_label
        state["mask_label"] = mask_label
        state["monitor_status"] = status
        state["camera_photo"] = None
        state["mask_photo"] = None
        self._place_calibration_monitor(state)

    def _refresh_calibration_backdrop(self, state, frame=None, point=None, score=0.0):
        """Draw a bright, stable target surface. Live DroidCam stays in the large inset panel.

        Showing the camera as a fullscreen background creates a recursive image of the
        same monitor. That can look dark and can feed the preview's own pixels back into
        laser detection. A stable surface plus a large live panel avoids both problems.
        """
        if not state or state.get("closing") or not state.get("wizard") or not state["wizard"].winfo_exists():
            return
        canvas = state["canvas"]
        w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if w < 300 or h < 200:
            return
        size = (w, h)
        if state.get("surface_size") == size and state.get("surface_drawn"):
            return
        try:
            canvas.delete("cal-bg")
            canvas.configure(bg="#f4f7fb")
            canvas.create_rectangle(0, 0, w, h, fill="#f4f7fb", outline="", tags=("cal-bg",))
            # Pale, subtle grid offers visual orientation but does not resemble a red laser.
            spacing = max(32, int(min(w, h) / 28))
            for gx in range(0, w + 1, spacing):
                canvas.create_line(gx, 0, gx, h, fill="#e5ebf2", width=1, tags=("cal-bg",))
            for gy in range(0, h + 1, spacing):
                canvas.create_line(0, gy, w, gy, fill="#e5ebf2", width=1, tags=("cal-bg",))
            canvas.create_rectangle(8, 8, w-8, h-8, outline="#d3dce7", width=2, tags=("cal-bg",))
            canvas.tag_lower("cal-bg")
            state["surface_size"] = size
            state["surface_drawn"] = True
        except Exception as exc:
            state["backdrop_error"] = str(exc)

    def _calibration_target_camera_point(self, state, frame_shape):
        """Project the active screen target back into camera pixels using the previous map as a visual guide."""
        guide = getattr(self, "calibration_guide_H", None)
        target = state.get("target_screen") if state else None
        if guide is None or target is None:
            return None
        try:
            canvas = state["canvas"]
            cw = max(1, canvas.winfo_width())
            ch = max(1, canvas.winfo_height())
            screen_ref = np.asarray([[[float(target[0])*1920.0/cw,
                                      float(target[1])*1080.0/ch]]], dtype=np.float32)
            inverse = np.linalg.inv(np.asarray(guide, dtype=np.float64))
            camera_pt = cv2.perspectiveTransform(screen_ref, inverse)[0, 0]
            h, w = frame_shape[:2]
            if not np.isfinite(camera_pt).all():
                return None
            if camera_pt[0] < 0 or camera_pt[0] >= w or camera_pt[1] < 0 or camera_pt[1] >= h:
                return None
            return float(camera_pt[0]), float(camera_pt[1])
        except Exception:
            return None

    def _update_calibration_monitor(self, state, frame, detector_mask, point, score, now):
        """Refresh the in-wizard visual proof that a current red spot is detected."""
        if not state or state.get("closing"):
            return
        self._refresh_calibration_backdrop(state, frame, point, score)
        panel = state.get("monitor")
        if panel is None or not panel.winfo_exists():
            fresh = point is not None and score >= CALIBRATION_MIN_CONFIDENCE
            state["fresh_spot"] = bool(fresh)
            state["last_detection_point"] = tuple(map(float, point)) if fresh else None
            state["last_detection_score"] = float(score) if fresh else 0.0
            return
        fresh = point is not None and score >= CALIBRATION_MIN_CONFIDENCE
        state["fresh_spot"] = bool(fresh)
        state["last_detection_point"] = tuple(map(float, point)) if fresh else None
        state["last_detection_score"] = float(score) if fresh else 0.0
        view = frame.copy()
        # Yellow marker = where the active screen target is expected to appear in
        # camera coordinates according to the previous/default map. This is a
        # visual guide only; it never counts as a detected laser or auto-accepts a sample.
        target_camera_point = self._calibration_target_camera_point(state, frame.shape)
        if target_camera_point is not None:
            tx, ty = int(round(target_camera_point[0])), int(round(target_camera_point[1]))
            cv2.circle(view, (tx, ty), 18, (0, 220, 255), 2)
            cv2.circle(view, (tx, ty), 5, (0, 220, 255), 1)
            cv2.line(view, (tx-24, ty), (tx+24, ty), (0, 220, 255), 1)
            cv2.line(view, (tx, ty-24), (tx, ty+24), (0, 220, 255), 1)
            cv2.putText(view, "TARGET GUIDE", (min(max(4, tx+22), max(4, frame.shape[1]-140)), max(16, ty-22)),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 220, 255), 1, cv2.LINE_AA)
        visual_red_point = None
        if not fresh:
            # Diagnostic only (never used for calibration): mark small red-ish
            # pixel blobs so the user can tell whether the spot is visible to the
            # camera but being rejected by the stricter laser verifier.
            try:
                hsv_diag = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                red1 = cv2.inRange(hsv_diag, np.array([0, 65, 85]), np.array([15, 255, 255]))
                red2 = cv2.inRange(hsv_diag, np.array([165, 65, 85]), np.array([180, 255, 255]))
                diag_mask = cv2.bitwise_or(red1, red2)
                n, labels, stats, centers = cv2.connectedComponentsWithStats(diag_mask, 8)
                h_img, w_img = diag_mask.shape[:2]
                options = []
                for label in range(1, n):
                    bx, by, bw, bh, area = map(int, stats[label])
                    if area < 1 or area > 120 or max(bw, bh) / max(1, min(bw, bh)) > 3.0:
                        continue
                    cx, cy = map(float, centers[label])
                    x0, x1 = max(0, int(cx)-3), min(w_img, int(cx)+4)
                    y0, y1 = max(0, int(cy)-3), min(h_img, int(cy)+4)
                    patch = frame[y0:y1, x0:x1]
                    if patch.size == 0:
                        continue
                    b_mean, g_mean, r_mean = [float(v) for v in cv2.mean(patch)[:3]]
                    redness = r_mean - max(g_mean, b_mean)
                    options.append((redness + min(area, 12) * 0.35, (cx, cy)))
                if options:
                    options.sort(key=lambda value: value[0], reverse=True)
                    visual_red_point = options[0][1]
            except Exception:
                visual_red_point = None
        if fresh:
            cx, cy = int(round(point[0])), int(round(point[1]))
            # Large overlay makes a small 1–3 px real spot visible in the inset.
            cv2.circle(view, (cx, cy), 16, (0, 255, 80), 2)
            # Never paint a synthetic red dot into the live preview: the camera can
            # see the preview on the monitor and mistake a red overlay for a real laser.
            cv2.circle(view, (cx, cy), 5, (255, 255, 0), -1)
            cv2.line(view, (cx-22, cy), (cx+22, cy), (0, 255, 80), 1)
            cv2.line(view, (cx, cy-22), (cx, cy+22), (0, 255, 80), 1)
            cv2.putText(view, f"VERIFIED {score:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 80), 2)
        else:
            if visual_red_point is not None:
                vx, vy = int(round(visual_red_point[0])), int(round(visual_red_point[1]))
                cv2.circle(view, (vx, vy), 16, (0, 215, 255), 2)
                cv2.circle(view, (vx, vy), 3, (0, 215, 255), -1)
                cv2.putText(view, "RED PIXELS - NOT VERIFIED", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 215, 255), 2)
            else:
                cv2.putText(view, "NO RED PIXELS VISIBLE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 255), 2)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        cam_im = Image.fromarray(rgb)
        cam_im = ImageOps.pad(
            cam_im, tuple(map(int, state.get("camera_preview_size", (430, 320)))),
            method=Image.Resampling.LANCZOS, color=(5, 7, 11), centering=(0.5, 0.5)
        )
        state["camera_photo"] = ImageTk.PhotoImage(cam_im)
        state["camera_label"].configure(image=state["camera_photo"], text="")

        if detector_mask is None or detector_mask.shape[:2] != frame.shape[:2]:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            mask = detector_mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        if target_camera_point is not None:
            tx, ty = int(round(target_camera_point[0])), int(round(target_camera_point[1]))
            cv2.circle(mask_rgb, (tx, ty), 18, (255, 220, 0), 1)
            cv2.line(mask_rgb, (tx-22, ty), (tx+22, ty), (255, 220, 0), 1)
            cv2.line(mask_rgb, (tx, ty-22), (tx, ty+22), (255, 220, 0), 1)
        if fresh:
            cx, cy = int(round(point[0])), int(round(point[1]))
            cv2.circle(mask_rgb, (cx, cy), 16, (0, 255, 80), 2)
        elif visual_red_point is not None:
            vx, vy = int(round(visual_red_point[0])), int(round(visual_red_point[1]))
            cv2.circle(mask_rgb, (vx, vy), 18, (255, 210, 0), 2)
            cv2.circle(mask_rgb, (vx, vy), 3, (255, 210, 0), -1)
            cv2.putText(mask_rgb, "UNVERIFIED", (8, max(18, vy-22)), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 210, 0), 1)
        else:
            cv2.putText(mask_rgb, "NO CANDIDATE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 80, 80), 2)
        mask_im = Image.fromarray(mask_rgb)
        mask_im = ImageOps.pad(
            mask_im, tuple(map(int, state.get("mask_preview_size", (250, 300)))),
            method=Image.Resampling.NEAREST, color=(5, 7, 11), centering=(0.5, 0.5)
        )
        state["mask_photo"] = ImageTk.PhotoImage(mask_im)
        state["mask_label"].configure(image=state["mask_photo"], text="")
        if fresh:
            state["monitor_status"].configure(
                text=(f"RED SPOT LOCKED · score {score:.2f} · ({point[0]:.0f}, {point[1]:.0f}) · "
                      + (f"guide offset {math.hypot(point[0]-target_camera_point[0], point[1]-target_camera_point[1]):.0f}px · align with yellow marker, hold steady"
                         if target_camera_point is not None else "hold steady on the physical target to capture")),
                fg=ACCENT)
        elif visual_red_point is not None:
            state["monitor_status"].configure(
                text="Red-looking pixels found, but laser verification failed. Keep the spot on the target; this target cannot be captured yet.",
                fg="#ffd166")
        else:
            state["monitor_status"].configure(
                text="NO RED PIXELS VISIBLE — calibration capture is BLOCKED. Check the camera view and point the laser at the target.",
                fg="#ff8585")

    def _mask_calibration_preview_feedback(self, frame):
        """Mask the on-screen camera preview from detection using the previous screen map.

        Since the phone is watching the laptop display, a displayed copy of the video
        contains recursive pixels. The inset is kept away from the active target and
        projected back into camera space so its own pixels are not treated as a laser.
        """
        state = self.calibration_state
        if not state or state.get("closing"):
            return frame
        panel = state.get("monitor")
        guide = getattr(self, "calibration_guide_H", None)
        bg = self.background_gray
        if panel is None or guide is None or bg is None or bg.shape != frame.shape[:2]:
            return frame
        try:
            if not panel.winfo_exists() or panel.winfo_width() < 350 or panel.winfo_height() < 220:
                return frame
            x0, y0 = panel.winfo_rootx(), panel.winfo_rooty()
            x1, y1 = x0 + panel.winfo_width(), y0 + panel.winfo_height()
            wizard = state["wizard"]
            sw, sh = max(1, wizard.winfo_screenwidth()), max(1, wizard.winfo_screenheight())
            corners = np.asarray([[[x0*1920.0/sw, y0*1080.0/sh],
                                   [x1*1920.0/sw, y0*1080.0/sh],
                                   [x1*1920.0/sw, y1*1080.0/sh],
                                   [x0*1920.0/sw, y1*1080.0/sh]]], dtype=np.float32)
            inv = np.linalg.inv(np.asarray(guide, dtype=np.float64))
            poly = cv2.perspectiveTransform(corners, inv)[0]
            if not np.isfinite(poly).all():
                return frame
            fh, fw = frame.shape[:2]
            area = abs(float(cv2.contourArea(poly.astype(np.float32))))
            frame_area = float(fw * fh)
            if area < frame_area * 0.003 or area > frame_area * 0.72:
                return frame
            # Reject unreasonable perspective projections rather than masking most of the frame.
            if np.ptp(poly[:, 0]) > fw * 1.7 or np.ptp(poly[:, 1]) > fh * 1.7:
                return frame
            exclusion = np.zeros((fh, fw), dtype=np.uint8)
            cv2.fillConvexPoly(exclusion, np.round(poly).astype(np.int32), 255)
            exclusion = cv2.dilate(exclusion, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            if cv2.countNonZero(exclusion) > frame_area * 0.75:
                return frame
            safe = frame.copy()
            gray_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            safe[exclusion > 0] = gray_bgr[exclusion > 0]
            state["preview_feedback_mask_pixels"] = cv2.countNonZero(exclusion)
            return safe
        except Exception:
            # Masking is an anti-feedback improvement, never a reason to lose the stream.
            return frame

    def start_calibration(self):
        """Run a quick five-target calibration; each point is captured automatically."""
        self.calibration_retry_count = 0
        if (self.cap is None or not self.cap.isOpened() or self.background_gray is None
                or not self.camera_last_frame_time
                or time.time() - self.camera_last_frame_time > CAMERA_FRAME_STALE_SECONDS):
            self.cal_status.config(text="Calibration is paused until live DroidCam frames and background capture are ready.", fg="#ffbd69")
            if self.cap is not None and (not self.camera_last_frame_time or time.time() - self.camera_last_frame_time > CAMERA_FRAME_STALE_SECONDS):
                self._schedule_camera_reconnect("Calibration requested without a fresh camera frame")
            return
        if self.calibration_state is not None:
            try:
                self.calibration_state["wizard"].destroy()
            except Exception:
                pass
            self.calibration_state = None
        self.H = None
        wizard = tk.Toplevel(self.root)
        wizard.title("Air Writer automatic calibration")
        wizard.configure(bg="#080b12")
        wizard.attributes("-fullscreen", True)
        wizard.attributes("-topmost", True)
        # Explicitly size/update the fullscreen window before computing target
        # coordinates. Otherwise Tk may briefly report the default 200x200 size.
        sw, sh = wizard.winfo_screenwidth(), wizard.winfo_screenheight()
        wizard.geometry(f"{sw}x{sh}+0+0")
        wizard.focus_force()
        canvas = tk.Canvas(wizard, bg="#080b12", highlightthickness=0, cursor="none")
        canvas.pack(fill="both", expand=True)
        wizard.update_idletasks()
        wizard.update()
        # Install the current real DroidCam frame immediately. The next frame-loop
        # update replaces it continuously; the calibration window never starts as a blank canvas.
        backdrop_item = None
        # Four widely separated corners plus the centre: enough for a homography
        # while reducing the user's calibration work from nine targets to five.
        fracs = [(0.06,0.10),(0.94,0.10),(0.50,0.50),
                 (0.06,0.90),(0.94,0.90)]
        state = {"wizard":wizard, "canvas":canvas, "idx":0, "points":[], "history":[],
                 "target_fracs":fracs, "target_screen":None, "last_paint":0.0,
                 "next_target_at":0.0, "finish_scheduled":False, "closing":False,
                 "last_monitor_update":0.0, "fresh_spot":False,
                 "last_detection_point":None, "last_detection_score":0.0,
                 "backdrop_item":backdrop_item, "backdrop_photo":None,
                 "last_backdrop_update":0.0, "backdrop_error":None,
                 "surface_drawn":False, "surface_size":None,
                 "alignment_error_px":None,
                 "alignment_tolerance_px":max(260.0, 0.14*math.hypot(wizard.winfo_screenwidth(), wizard.winfo_screenheight()))}
        self.calibration_state = state
        if self.frame is not None:
            self._refresh_calibration_backdrop(state, self.frame, None, 0.0)
        self._create_calibration_monitor(wizard, state)
        self.auto_phase = "calibrating"
        self.cal_status.config(text="Quick calibration: aim at each of 5 targets and hold the laser steady until the ring completes.", fg=ACCENT)
        self._paint_calibration_target(state)
        def escape_only(event):
            if event.keysym == "Escape":
                state["closing"] = True
                self.calibration_state = None
                self.auto_phase = "idle"
                try: wizard.destroy()
                except Exception: pass
                self.status.config(text="Calibration cancelled. Use Restart automatic setup to retry.", fg="#ffbd69")
        wizard.bind("<Key>", escape_only)
        wizard.protocol("WM_DELETE_WINDOW", lambda: escape_only(type("Event", (), {"keysym":"Escape"})()))

    def _feed_automatic_calibration(self, point, now, score=0.0):
        state = self.calibration_state
        if not state or state.get("closing") or not state["wizard"].winfo_exists():
            return
        fresh = point is not None and float(score) >= CALIBRATION_MIN_CONFIDENCE
        state["fresh_spot"] = bool(fresh)
        state["last_detection_point"] = tuple(map(float, point)) if fresh else None
        state["last_detection_score"] = float(score) if fresh else 0.0
        _guide_point, alignment_error = self._calibration_guide_position(point) if fresh else (None, None)
        state["alignment_error_px"] = alignment_error
        if now < state["next_target_at"]:
            return
        if state["idx"] >= len(state["target_fracs"]):
            if not state["finish_scheduled"]:
                state["finish_scheduled"] = True
                self.root.after(700, self._complete_automatic_calibration)
            return
        history = state["history"]
        # The previous homography is only a visual reference, never a gate.
        # A moved camera must still be able to recalibrate from physical targets.
        # Only a fresh verified spot that remains stable may be sampled. Do not
        # re-use a recently cached cursor position to advance calibration.
        if fresh:
            new_xy = np.asarray([float(point[0]), float(point[1])], dtype=np.float32)
            if history and float(np.linalg.norm(new_xy-np.asarray(history[-1][1:3], dtype=np.float32))) > 16.0:
                history.clear()
            history.append((now, float(point[0]), float(point[1])))
        else:
            if history and now-history[-1][0] > 0.22:
                history.clear()
        while history and now-history[0][0] > 1.35:
            history.pop(0)

        progress = 0.0
        stable = False
        median_point = None
        span = 0.0
        if len(history) >= 12:
            arr = np.asarray([[v[1],v[2]] for v in history], dtype=np.float32)
            span = history[-1][0] - history[0][0]
            std = np.std(arr, axis=0)
            ranges = np.ptp(arr, axis=0)
            stable = span >= 0.75 and float(np.max(std)) <= 5.5 and float(np.max(ranges)) <= 18.0
            if stable:
                progress = min(1.0, span/1.0)
                median_point = np.median(arr, axis=0)

        unique_from_previous = True
        if median_point is not None:
            for captured in state["points"]:
                if np.linalg.norm(median_point - np.asarray(captured[:2], dtype=np.float32)) < 18.0:
                    unique_from_previous = False
                    break

        msg = "Aim the red laser dot at this target and hold steady; it captures automatically."
        if not fresh:
            msg = "Laser not detected. Aim at the target and keep the laser ON."
        elif not stable:
            msg = "Laser detected — hold the dot very steady; the ring fills when stable."
        elif not unique_from_previous:
            msg = "Move the dot to this new target; this point matches the previous target."
            progress = 1.0

        if fresh and stable and unique_from_previous and median_point is not None and progress >= 0.98:
            w, h = state["canvas"].winfo_width(), state["canvas"].winfo_height()
            tx, ty = state["target_screen"]
            row = [float(median_point[0]), float(median_point[1]),
                   float(tx)*1920.0/max(1,w), float(ty)*1080.0/max(1,h)]
            state["points"].append(row)
            state["idx"] += 1
            state["history"].clear()
            state["next_target_at"] = now + 0.55
            if state["idx"] >= len(state["target_fracs"]):
                self._paint_calibration_target(state, 1.0, captured=True)
                self.cal_status.config(text="Five stable target samples collected. Checking mapping quality…", fg=ACCENT)
                if not state["finish_scheduled"]:
                    state["finish_scheduled"] = True
                    self.root.after(800, self._complete_automatic_calibration)
                return
            self._paint_calibration_target(state, 1.0, message="Target captured ✓ — moving to the next target…", captured=True)
            state["last_paint"] = now
            return

        if now-state["last_paint"] > 0.08:
            self._paint_calibration_target(state, progress, msg)
            state["last_paint"] = now

    def _retry_automatic_calibration(self, reason):
        state = self.calibration_state
        self.calibration_state = None
        if state:
            state["closing"] = True
            try: state["wizard"].destroy()
            except Exception: pass
        self.H = None
        self.auto_phase = "calibrating"
        self.calibration_retry_count += 1
        self.cal_status.config(text=f"Calibration not accepted: {reason}. Correct the laser position, then press Retry 5-point calibration.", fg="#ffbd69")
        self.status.config(text="Calibration stopped safely; no automatic retry loop. Press Retry 5-point calibration when ready.", fg="#ffbd69")
        # Do not loop through calibration indefinitely. Leave the camera preview
        # running and let the user explicitly retry after correcting the issue.
        self.auto_phase = "ready_for_calibration_retry"

    def _complete_automatic_calibration(self):
        state = self.calibration_state
        if not state or len(state["points"]) < len(state.get("target_fracs", ())) or len(state["points"]) < 5:
            return
        rows = np.asarray(state["points"], dtype=np.float32)
        src, dst = rows[:, :2], rows[:, 2:4]
        frame_w = max(1, int(self.frame.shape[1]))
        frame_h = max(1, int(self.frame.shape[0]))
        pairwise = []
        for i in range(len(src)):
            for j in range(i+1, len(src)):
                pairwise.append(float(np.linalg.norm(src[i]-src[j])))
        if not pairwise or min(pairwise) < 12.0:
            self._retry_automatic_calibration("several targets mapped to the same camera area")
            return
        if float(np.ptp(src[:,0])) < frame_w*0.18 or float(np.ptp(src[:,1])) < frame_h*0.18:
            self._retry_automatic_calibration("laser samples do not cover enough of the camera view")
            return
        try:
            H0, mask = cv2.findHomography(src, dst, cv2.RANSAC, 50.0)
            inliers = np.ones(len(src), dtype=bool) if mask is None else mask.ravel().astype(bool)
            if H0 is None or int(inliers.sum()) < 4:
                self._retry_automatic_calibration("fewer than four points agree on a screen mapping")
                return
            H, _ = cv2.findHomography(src[inliers], dst[inliers], 0)
            if H is None or not np.isfinite(H).all() or abs(float(H[2,2])) < 1e-10:
                self._retry_automatic_calibration("the perspective transform is invalid")
                return
            projected = cv2.perspectiveTransform(src.reshape(-1,1,2), H).reshape(-1,2)
            errors = np.linalg.norm(projected-dst, axis=1)
            # Validate every collected point, including any RANSAC outlier. This
            # prevents a bad fifth/centre sample from being silently ignored.
            accepted_errors = errors
            median_error = float(np.median(accepted_errors)) if len(accepted_errors) else 999.0
            p90_error = float(np.percentile(accepted_errors, 90)) if len(accepted_errors) else 999.0
            if median_error > 55.0 or p90_error > 110.0:
                self._retry_automatic_calibration(f"mapping error is too high (median {median_error:.0f}px, P90 {p90_error:.0f}px)")
                return
        except Exception as exc:
            self._retry_automatic_calibration(f"could not calculate homography ({exc})")
            return

        # Store destinations in the existing 1920x1080 reference format so the
        # current mapping file stays compatible with other project modules.
        self.calibration = rows
        self.H = H
        self.calibration_guide_H = H.copy()
        self.calibration_source_saved = True
        try:
            np.savetxt(CAL_FILE, rows, delimiter=",", fmt="%.3f")
        except OSError as exc:
            self.preview_info.config(text=f"Calibration valid in memory but could not save {CAL_FILE.name}: {exc}")
        self.calibration_state = None
        self.auto_phase = "ready"
        try:
            state["closing"] = True
            state["wizard"].destroy()
        except Exception:
            pass
        self.cal_status.config(text=f"Calibration complete: 5 points · median error {median_error:.1f}px · P90 {p90_error:.1f}px", fg=ACCENT)
        self.status.config(text="●  CALIBRATION COMPLETE — cursor mode active. Use the laser to move, click, draw, erase, undo and redo.", fg=ACCENT)
        self.preview_info.config(text="Calibration saved. Automatically opening the smart board in cursor mode.")
        self.mouse_enabled = True
        self.board_mode = "CURSOR"
        self._wait_for_laser_off = not self.hand_mode
        self._wait_for_laser_off_since = None
        self._laser_on_since = None
        self._laser_was_active = False
        self._laser_off_since = None
        self._last_mapped_point = None
        self.root.after(250, self._auto_open_board)

    def _auto_open_board(self):
        """Automatically open the Smart Board after a successful calibration."""
        if self.H is None or self.background_gray is None or not self.camera_confirmed:
            return
        self.board_mode = "CURSOR"
        self.mouse_enabled = True
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            if getattr(self, "open_board_btn", None) is not None:
                self.open_board_btn.configure(text="SMART BOARD OPEN")
        except Exception:
            pass
        self.status.config(text="● CALIBRATION COMPLETE — opening Smart Board…", fg=ACCENT)
        self.preview_info.config(text="Calibration passed. Smart Board will open automatically in cursor mode.")
        if self.mode_label is not None:
            self.mode_label.config(text=f"LASER MODE: laser OFF = click")
        # Use Tk's event queue so the calibration window is fully destroyed
        # before the full-screen board is created. This avoids focus/z-order races.
        def _open_board_after_calibration():
            try:
                self.open_board()
            except Exception as exc:
                self.running_board = False
                detail = f"Smart Board auto-open failed: {type(exc).__name__}: {exc}"
                self._log_runtime_event(detail)
                try:
                    self.status.config(text=detail, fg="#ff7474")
                    self.preview_info.config(text=detail)
                except Exception:
                    pass
                traceback.print_exc()

        try:
            self.root.after_idle(_open_board_after_calibration)
        except Exception:
            _open_board_after_calibration()

    def _tick(self):
        if self.hand_mode:
            try:
                self._tick_hand()
            except Exception as exc:
                detail = f"Hand tracking error: {type(exc).__name__}: {exc}"
                try:
                    self.preview_info.config(text=detail)
                    self._log_runtime_event(detail)
                except Exception:
                    pass
            self.root.after(25, self._tick)
            return
        now = time.time()
        if self.cap is not None:
            try:
                reader = getattr(self, "camera_reader", None)
                if reader is None:
                    raise RuntimeError("DroidCam frame reader is not running")
                frame, frame_seq, frame_stamp, read_failures, read_error = reader.snapshot()
                frame_age = float("inf") if not frame_stamp else max(0.0, now - frame_stamp)
                if frame is None or frame_age > CAMERA_FRAME_STALE_SECONDS:
                    self.camera_read_failures = max(1, int(read_failures))
                    self.camera_last_error = read_error or "No fresh DroidCam frame has arrived"
                    self.laser_point = None
                    self.laser_score = 0.0
                    self.laser_mask = None
                    self.last_point_time = 0.0
                    self._laser_point_processed_at = 0.0
                    self.smoothed = None
                    self._pending_jump_point = None
                    self._pending_jump_count = 0
                    self.status.config(text="●  DROIDCAM FRAME STALE — waiting for a fresh phone frame", fg="#ff7474")
                    if self.calibration_state is not None:
                        state = self.calibration_state
                        state["fresh_spot"] = False
                        state["last_detection_point"] = None
                        state["last_detection_score"] = 0.0
                        self._mark_calibration_camera_stale(
                            state, f"CAMERA FRAME STALE — {self.camera_last_error}. Calibration is paused.")
                    if frame_age > CAMERA_FRAME_STALE_SECONDS and not self.camera_reconnect_pending:
                        self._schedule_camera_reconnect(self.camera_last_error)
                elif frame_seq == self._last_processed_camera_seq:
                    # No new sensor sample yet. Do not reuse a cached point as a
                    # tracking sample or as part of the 0.20-second click dwell.
                    pass
                else:
                    self._last_processed_camera_seq = frame_seq
                    now = frame_stamp
                    self.camera_last_frame_time = frame_stamp
                    self.camera_read_failures = 0
                    self.camera_last_error = ""
                    self.frame = frame
                    if self.calibration_state is not None:
                        self._show_raw_calibration_frame(self.calibration_state, frame)
                    # These three values describe this exact frame only; cached
                    # cursor positions never count as fresh calibration samples.
                    detected_point = None
                    detected_score = 0.0
                    detected_mask = None
                    calibration_point = None
                    calibration_score = 0.0
                    if self.auto_phase in ("warmup", "background"):
                        self._advance_automatic_setup(frame, now)

                    if self.background_gray is not None and self.auto_phase not in ("warmup", "background"):
                        detection_frame = self._mask_calibration_preview_feedback(frame) if self.calibration_state is not None else frame
                        detected_point, detected_score, detected_mask = find_laser(
                            detection_frame, self.background_gray, self.yolo_model,
                            self.yolo_class_id, previous_point=self.laser_point)
                        self.laser_mask = detected_mask
                        if detected_point is not None:
                            cand = np.asarray(detected_point, dtype=np.float32)
                            current = None if self.laser_point is None else np.asarray(self.laser_point, dtype=np.float32)
                            if current is None or now-self.last_point_time > 0.22 or float(np.linalg.norm(cand-current)) <= 85.0:
                                if self.smoothed is None or current is None:
                                    self.smoothed = cand.copy()
                                else:
                                    dt = max(1.0/120.0, min(0.15, now-self.last_point_time))
                                    distance = float(np.linalg.norm(cand-current))
                                    speed = distance / dt
                                    # Slow/stationary movements receive stronger jitter filtering;
                                    # fast intentional movements get a larger alpha to reduce lag.
                                    if distance <= 1.25:
                                        alpha = 0.24
                                    else:
                                        alpha = 0.34 + 0.54 * min(1.0, speed / 700.0)
                                    self.smoothed = alpha*cand + (1.0-alpha)*self.smoothed
                                self.laser_point = tuple(map(float, self.smoothed))
                                self.laser_score = float(detected_score)
                                self.last_point_time = now
                                self._laser_point_processed_at = time.time()
                                self._pending_jump_point = None
                                self._pending_jump_count = 0
                                calibration_point = tuple(map(float, cand))
                                calibration_score = float(detected_score)
                            else:
                                # Accept a large jump only after three consistent
                                # current-frame detections at the new location.
                                if self._pending_jump_point is None or float(np.linalg.norm(cand-self._pending_jump_point)) > 28.0:
                                    self._pending_jump_point = cand
                                    self._pending_jump_count = 1
                                else:
                                    self._pending_jump_point = 0.55*cand + 0.45*self._pending_jump_point
                                    self._pending_jump_count += 1
                                if self._pending_jump_count >= 3:
                                    self.smoothed = self._pending_jump_point.copy()
                                    self.laser_point = tuple(map(float, self.smoothed))
                                    self.laser_score = float(detected_score)
                                    self.last_point_time = now
                                    self._laser_point_processed_at = time.time()
                                    self._pending_jump_point = None
                                    self._pending_jump_count = 0
                                    calibration_point = tuple(map(float, cand))
                                    calibration_score = float(detected_score)
                        elif now-self.last_point_time > LASER_DETECTION_RELEASE_GRACE_SECONDS:
                            self.laser_point = None
                            self._laser_point_processed_at = 0.0
                            self.smoothed = None
                            self._pending_jump_point = None
                            self._pending_jump_count = 0

                    if self.calibration_state is not None:
                        self._feed_automatic_calibration(calibration_point, now, calibration_score)
                        if now - self.calibration_state.get("last_monitor_update", 0.0) >= 0.06:
                            self._update_calibration_monitor(
                                self.calibration_state, frame, detected_mask,
                                calibration_point, calibration_score, now)
                            self.calibration_state["last_monitor_update"] = now

                    shown = frame.copy()
                    if self.laser_point is not None and now-self.last_point_time < 0.20:
                        p = tuple(map(int, self.laser_point))
                        cv2.circle(shown, p, 9, (60,240,130), 2)
                        cv2.circle(shown, p, 2, (60,240,130), -1)
                        cv2.putText(shown, f"LASER LOCK {self.laser_score:.2f}", (12,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (60,240,130), 2)
                    elif self.background_gray is None:
                        cv2.putText(shown, "AUTOMATIC SETUP: LASER OFF", (12,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,190,255), 2)
                    else:
                        cv2.putText(shown, "LASER NOT VERIFIED", (12,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,80,255), 2)
                    self._show_preview(shown)
                    if self.background_gray is not None:
                        self.preview_info.config(text=f"{self.camera_name} | {frame.shape[1]}×{frame.shape[0]} | Spot: {'LOCKED' if self.laser_point is not None and now-self.last_point_time<.20 else 'not detected'} | Mapping: {'ready' if self.H is not None else self.auto_phase}")
                    if self.mouse_enabled:
                        self._handle_mouse_signal()
                    if self.running_board:
                        self._board_tick()
            except Exception as exc:
                detail = f"Camera/tracker processing error: {type(exc).__name__}: {exc}"
                self.preview_info.config(text=detail)
                try:
                    self._log_runtime_event(detail)
                except Exception:
                    pass
                if self.calibration_state is not None:
                    state = self.calibration_state
                    try:
                        if state.get("monitor_status") is not None and state["monitor_status"].winfo_exists():
                            state["monitor_status"].configure(text=detail, fg="#ff8585")
                        if state.get("camera_label") is not None and state["camera_label"].winfo_exists():
                            # Keep the last real camera image visible; never replace it with
                            # the original black waiting placeholder on detector errors.
                            if state.get("camera_photo") is None:
                                state["camera_label"].configure(text="Camera frame received; detector error shown below")
                    except Exception:
                        pass
        self.root.after(25, self._tick)

    def _show_preview(self, frame):
        w=max(320,self.preview.winfo_width()); h=max(240,self.preview.winfo_height())
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); im=Image.fromarray(rgb)
        im=ImageOps.pad(im,(w,h),method=Image.Resampling.LANCZOS,color=(9,11,16),centering=(0.5,0.5))
        self._preview_photo=ImageTk.PhotoImage(im); self.preview.configure(image=self._preview_photo,text="")

    def toggle_mouse(self):
        self.mouse_enabled = not self.mouse_enabled
        if self.mouse_enabled:
            self._desktop_mouse_suspended = False
            self._reset_click_signal_state()
            self.last_mouse_click = 0.0
            self._wait_for_laser_off = True
            self._wait_for_laser_off_since = None
            self._log_runtime_event("Mouse control re-enabled; click state re-armed")
            message = "Mouse control enabled · release the laser briefly, then click with a fresh ON→OFF cycle."
            color = ACCENT
        else:
            self._cancel_pending_desktop_click()
            self._reset_click_signal_state()
            if self._drag_down:
                try:
                    import pyautogui
                    pyautogui.mouseUp(button="left")
                except Exception:
                    pass
                self._drag_down = False
            message = "Mouse control disabled · safe preview mode."
            color = "#ffbd69"
        self.status.config(text=message, fg=color)

    def open_board(self):
        if self.hand_mode:
            if self.hand_academy_required and not self.hand_academy_completed_this_run:
                self.status.config(text="Complete all six Gesture Academy lessons before opening the Hand Smart Board.", fg="#ffbd69")
                if self.cap is not None and self.camera_confirmed:
                    self._start_hand_academy()
                return
            if self.cap is None or not self.camera_confirmed or self.hand_tracker is None:
                self.status.config(text="Board locked until the Hand camera and MediaPipe tracker are ready.", fg="#ffbd69")
                return
        else:
            if self.cap is None or self.background_gray is None or not self.camera_confirmed:
                self.status.config(text="Board locked until DroidCam and background capture are ready.", fg="#ffbd69")
                return
            if self.H is None:
                self.status.config(text="Board locked until calibration succeeds. Use Retry 5-point calibration.", fg="#ffbd69")
                return
        if self.running_board and self.board_win:
            try:
                self.board_win.lift()
                self.board_win.attributes("-fullscreen", True)
            except Exception:
                pass
            return
        self.running_board = True
        if not self.pages:
            self.pages = [[]]
            self.page_index = 0
        self.page_index = max(0, min(self.page_index, len(self.pages)-1))
        self.strokes = self.pages[self.page_index]
        self.current_stroke = None
        self.last_board_point = None
        self.menu_gesture_points = []
        self._last_menu_gesture_center = None
        self.shape_start = self.shape_end = None
        self.swipe_start = self.swipe_last = None
        self.scroll_last_point = None
        self.board_win = tk.Toplevel(self.root)
        self.board_win.title("Air Writer Smart Board")
        self.board_win.configure(bg="#0b111b")
        self.board_win.attributes("-fullscreen", True)
        self.board_win.attributes("-topmost", True)
        self.board_win.lift()

        bar = tk.Frame(self.board_win, bg="#151e2b", padx=10, pady=7,
                       highlightbackground="#273548", highlightthickness=1)
        bar.pack(fill="x")
        self.board_toolbar = bar
        brand = tk.Label(bar, text="AIR WRITER", bg="#151e2b", fg=ACCENT,
                         font=("Segoe UI", 12, "bold"))
        brand.pack(side="left", padx=(0, 12))
        actions = [
            ("Cursor", lambda:self.set_mode("CURSOR")),
            ("Drag", lambda:self.set_mode("DRAG")),
            ("Pen", lambda:self.set_mode("PENCIL")),
            ("Smart Write", lambda:self.set_mode("SMART_WRITE")),
            ("Swipe Type", self.open_swipe_keyboard),
            ("Erase", lambda:self.set_mode("ERASER")),
            ("Shapes", lambda:self.show_radial_menu(page="shapes")),
            ("Scroll", lambda:self.set_mode("SCROLL")),
            ("Swipe / Pages", lambda:self.set_mode("SWIPE")),
            ("Undo", self.undo), ("Redo", self.redo),
            ("New page", self.new_page), ("Save PNG", self.save_board),
            ("Radial Menu", self.show_radial_menu), ("Close Board", self.close_board),
        ]
        for label, command in actions:
            self._button(bar, label, command, outline=True).pack(side="left", padx=2, pady=1)

        initial_mode_text = (
            "HAND MODE · index move · pinch draw/click · V pan · 3 fingers menu"
            if self.hand_mode else
            f"CURSOR MODE · aim, then laser OFF clicks"
        )
        self.mode_label = tk.Label(bar, text=initial_mode_text,
                                   bg="#151e2b", fg=ACCENT, font=("Segoe UI", 8, "bold"))
        self.mode_label.pack(side="right", padx=(12, 2))

        self.hand_board_preview = None
        self.hand_board_preview_photo = None
        self.hand_board_status = None
        self.hand_board_gesture_chip = None
        if self.hand_mode:
            # Keep the drawing canvas full-size. The gesture guide is anchored
            # at the TOP-RIGHT; the live camera preview is a separate panel at
            # the BOTTOM-RIGHT so both remain visible at the same time.
            work_area = tk.Frame(self.board_win, bg="#080e17")
            work_area.pack(fill="both", expand=True)
            self.board_canvas = tk.Canvas(work_area, bg="#080e17", highlightthickness=0, cursor="none")
            self.board_canvas.pack(fill="both", expand=True)

            # Persistent live gesture status at the bottom-left of the canvas.
            self.hand_board_gesture_chip = tk.Label(
                work_area, text="  HAND READY  ", bg="#152536", fg=ACCENT,
                font=("Segoe UI", 8, "bold"), padx=10, pady=6)
            self.hand_board_gesture_chip.place(relx=0.015, rely=0.985, anchor="sw")

            # Gesture instructions / "matter" stays at the TOP-RIGHT.
            hand_guide_panel = tk.Frame(
                work_area, bg="#151e2b", width=350, height=205, padx=12, pady=10,
                highlightbackground="#3a536b", highlightthickness=1)
            hand_guide_panel.place(relx=0.99, rely=0.015, anchor="ne",
                                   width=350, height=205)
            hand_guide_panel.pack_propagate(False)
            tk.Label(hand_guide_panel, text="HAND GESTURE GUIDE",
                     bg="#151e2b", fg=ACCENT, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 7))
            guide = (
                "INDEX FINGER  ·  move pointer\\n"
                "THUMB + INDEX PINCH  ·  draw / click / select\\n"
                "V SIGN  ·  pan canvas\\n"
                "THUMB + MIDDLE  ·  zoom\\n"
                "THREE FINGERS  ·  open tools menu; pinch-release to select"
            )
            tk.Label(hand_guide_panel, text=guide, bg="#151e2b", fg=TEXT,
                     font=("Segoe UI", 8), justify="left", anchor="w",
                     wraplength=320).pack(anchor="w", fill="x", pady=(0, 7))
            tk.Label(hand_guide_panel, text="Camera processing stays local.",
                     bg="#151e2b", fg=MUTED, font=("Segoe UI", 8),
                     justify="left").pack(anchor="w")

            # Live mirrored camera window stays independently visible at BOTTOM-RIGHT.
            hand_camera_panel = tk.Frame(
                work_area, bg="#151e2b", width=350, height=335, padx=12, pady=10,
                highlightbackground="#3a536b", highlightthickness=1)
            hand_camera_panel.place(relx=0.99, rely=0.985, anchor="se",
                                    width=350, height=335)
            hand_camera_panel.pack_propagate(False)
            tk.Label(hand_camera_panel, text="LIVE MIRRORED HAND CAMERA",
                     bg="#151e2b", fg=TEXT, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 6))
            self.hand_board_preview = tk.Label(
                hand_camera_panel, bg="#080e17", fg=MUTED,
                text="Waiting for camera…", font=("Segoe UI", 9), height=11)
            self.hand_board_preview.pack(fill="both", expand=True, pady=(0, 7))
            self.hand_board_status = tk.Label(
                hand_camera_panel, text="LIVE GESTURE: waiting for hand",
                bg="#151e2b", fg="#ffbd69", anchor="w",
                font=("Segoe UI", 8, "bold"), wraplength=320, justify="left")
            self.hand_board_status.pack(fill="x")
        else:
            self.board_canvas = tk.Canvas(self.board_win, bg="#080e17", highlightthickness=0, cursor="none")
            self.board_canvas.pack(fill="both", expand=True)
        self.board_canvas.bind("<Configure>", lambda e:self._redraw_board())
        self.board_win.bind("<Escape>", lambda e:self.close_board())
        self.board_win.protocol("WM_DELETE_WINDOW", self.close_board)
        self.board_mode = "CURSOR"
        self.mouse_enabled = True
        self.pan_x = self.pan_y = 0.0
        self._wait_for_laser_off = True
        self._wait_for_laser_off_since = None
        self._laser_on_since = None
        self._laser_was_active = False
        self._laser_off_since = None
        self._last_mapped_point = None
        self._last_mapped_was_toolbar = False
        self.menu_gesture_points = []
        self._last_menu_gesture_center = None
        self.last_board_point = None
        self.current_stroke = None
        # Allow the FIRST Z gesture immediately after the board opens. The
        # cooldown is started only after a menu actually opens, not at board launch.
        self.last_menu_trigger = time.time() - 2.0
        self._redraw_board()
        try:
            self.board_win.update_idletasks()
        except Exception:
            pass
        try:
            self.board_win.focus_force()
        except Exception:
            pass
        if self.hand_mode:
            self.status.config(text="Hand Smart Board open · index finger moves the pointer; pinch and release to draw or click; V-sign pans; three fingers opens tools.", fg=ACCENT)
            if self.mode_label is not None:
                self.mode_label.config(text="HAND MODE · index move · pinch draw/click · V pan · 3 fingers menu")
        else:
            self.status.config(text="Smart Board open · Z MENU: draw one continuous capital Z inside the canvas (top line → diagonal down-left → bottom line). The radial tools menu opens at the gesture centre. Release the laser, wait for MENU READY, then aim at an item and switch OFF. The Radial Menu toolbar button is a manual fallback.", fg=ACCENT)

    def close_board(self):
        self._cancel_pending_desktop_click()
        if getattr(self, "board_mode", None) == "SWIPE_TYPE" and getattr(self, "swipe_type_text", "").strip() and getattr(self, "board_canvas", None) is not None:
            try:
                self._sync_swipe_type_text_object(finalize=True, redraw=False)
                self.swipe_type_text = ""
                self.swipe_type_suggestions = []
                self.swipe_type_last_word_span = None
                self.swipe_type_last_word = ""
            except Exception as exc:
                try:
                    self._log_runtime_event(f"Could not finalize Swipe Type draft while closing board: {exc}")
                except Exception:
                    pass
        self.running_board = False
        self.last_board_point = None
        self.menu_gesture_points = []
        self._last_menu_gesture_center = None
        self.shape_start = self.shape_end = None
        self.swipe_start = self.swipe_last = None
        self.scroll_last_point = None
        if self._drag_down:
            try:
                import pyautogui
                pyautogui.mouseUp()
            except Exception:
                pass
            self._drag_down = False
        if self.board_win:
            try: self.board_win.destroy()
            except Exception: pass
        self.board_win = None
        self.board_canvas = None
        self.board_toolbar = None
        self.mode_label = None
        self.menu_ids = []
        self._menu_waiting_for_release = False
        self._menu_off_since = None
        self._last_board_canvas_point = None
        self._reset_click_signal_state()
        self._menu_off_since = None
        self.swipe_keyboard_open = False
        self._cancel_swipe_type_gesture()
        self.swipe_keyboard_keys = []
        self.swipe_suggestion_targets = []
        self.board_mode = "CURSOR" if self.H is not None else "PENCIL"
        if self.hand_mode and self.camera_confirmed:
            try:
                self.root.deiconify()
                self.root.lift()
            except Exception:
                pass
        elif self.H is not None and self.camera_confirmed:
            self.mouse_enabled = True
            try:
                self.root.deiconify()
                self.root.lift()
                if getattr(self, "open_board_btn", None) is not None:
                    self.open_board_btn.configure(text="OPEN SMART BOARD  ·  LASER CLICK")
            except Exception:
                pass
            self.status.config(text=f"Board closed · aim at Open Smart Board, then switch laser OFF.", fg=ACCENT)

    def set_mode(self, mode):
        mode = str(mode).upper()
        allowed = {"CURSOR", "DRAG", "PENCIL", "SMART_WRITE", "SWIPE_TYPE", "HIGHLIGHTER", "ERASER", "SHAPES", "SCROLL", "SWIPE"}
        if mode not in allowed:
            mode = "CURSOR"
        # If the user leaves Smart Write, finish the pending word now instead
        # of silently dropping the strokes waiting for the pause timer.
        if self.board_mode == "SWIPE_TYPE" and mode != "SWIPE_TYPE":
            self._cancel_swipe_type_gesture()
            # Finalize any composed text before leaving the keyboard. The Z
            # escape gesture therefore saves the sentence rather than trapping
            # the user in Swipe Type or silently losing the words.
            self._sync_swipe_type_text_object(finalize=True, redraw=False)
            self.swipe_type_text = ""
            self.swipe_type_suggestions = []
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
            self.swipe_keyboard_open = False
            if self.board_canvas is not None:
                self.board_canvas.delete("swipe-keyboard")
                self.board_canvas.delete("swipe-type-header")
                self.board_canvas.delete("swipe-type-trace")
            self._redraw_board()
        if self.board_mode == "SMART_WRITE" and mode != "SMART_WRITE":
            if self.current_stroke is not None:
                self._queue_smart_write_stroke(self.current_stroke, time.time())
                if self.board_canvas is not None:
                    self.board_canvas.delete("active-stroke-preview")
                    self._redraw_board()
                self.current_stroke = None
            if self._smart_write_pending_strokes:
                self._process_smart_write_pending(force=True)
        if self._drag_down and mode != "DRAG":
            try:
                import pyautogui
                pyautogui.mouseUp()
            except Exception:
                pass
            self._drag_down = False
        if self.current_stroke is not None and self.board_canvas is not None:
            self.board_canvas.delete("active-stroke-preview")
            self._redraw_board()
        self.board_mode = mode
        self.last_board_point = None
        self._stroke_filter_time = 0.0
        self.current_stroke = None
        self.menu_gesture_points = []
        self._last_menu_gesture_center = None
        self.shape_start = self.shape_end = None
        self.swipe_start = self.swipe_last = None
        self.scroll_last_point = None
        if self.mode_label is not None:
            descriptions = {
                "CURSOR": f"CURSOR · laser OFF = one left-click",
                "DRAG": "DRAG · hold laser ON to press · laser OFF releases",
                "PENCIL": "PEN · trace while laser is ON; switch OFF to lift",
                "SMART_WRITE": f"SMART WRITE · handwriting-to-text after {SMART_WRITE_PAUSE_SECONDS:.2f}s pause",
                "SWIPE_TYPE": f"SWIPE TYPE · hold each key {SWIPE_TYPE_KEY_DWELL_SECONDS:.1f}s · letters combine into words · Z opens tools",
                "HIGHLIGHTER": "HIGHLIGHTER · broad green strokes while laser is ON",
                "ERASER": "ERASER · trace over marks while laser is ON",
                "SHAPES": f"SHAPES · {self.shape_type} · start then end the laser stroke",
                "SCROLL": "SCROLL / PAN · move the laser up/down to move the page",
                "SWIPE": "SWIPE / PAGE · sweep horizontally, then switch laser OFF",
            }
            self.mode_label.config(text=descriptions[mode])
        self.status.config(text=f"Laser tool changed to {mode}", fg=ACCENT)

    def choose_color(self, event=None):
        palette = {"White":"#f4f7fb", "Red":"#ff5353", "Green":"#42e695", "Blue":"#55a7ff",
                   "Yellow":"#ffd45c", "Cyan":"#52e8e8", "Magenta":"#ef75ff", "Black":"#090b10"}
        self.pen_color = palette.get(self.color_var.get(), self.pen_color)
        self.set_mode("PENCIL")

    def change_size(self):
        try:
            self.pen_size = max(1, min(30, int(self.size_var.get())))
        except Exception:
            pass

    def new_page(self):
        """Create a new blank page every time the explicit New Page action is chosen."""
        if not self.pages:
            self.pages = [[]]
            self.page_index = 0
        self.pages.append([])
        self.page_index = len(self.pages) - 1
        self.strokes = self.pages[self.page_index]
        self.redo_strokes = []
        self.current_stroke = None
        self.last_board_point = None
        self._stroke_filter_time = 0.0
        self.pan_x = self.pan_y = 0.0
        self._last_clear_snapshot = None
        self._redraw_board()
        self.status.config(text=f"New page created · page {self.page_index+1} of {len(self.pages)}", fg=ACCENT)

    def _go_to_page(self, index, create=False):
        if create and index >= len(self.pages):
            while len(self.pages) <= index:
                self.pages.append([])
        if not self.pages:
            self.pages = [[]]
        index = max(0, min(len(self.pages)-1, int(index)))
        self.page_index = index
        self.strokes = self.pages[index]
        self.redo_strokes = []
        self.current_stroke = None
        self.last_board_point = None
        self.pan_x = self.pan_y = 0.0
        self._redraw_board()
        self.status.config(text=f"Page {self.page_index+1} of {len(self.pages)}", fg=ACCENT)

    def undo(self):
        # While the swipe keyboard is active, Undo removes one typed character
        # from the live draft first; normal document Undo is available after Done
        # or after opening the Z menu, which finalizes the draft.
        if self.board_mode == "SWIPE_TYPE" and str(self.swipe_type_text).strip():
            self.swipe_type_text = self.swipe_type_text[:-1]
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
            self.swipe_type_suggestions = []
            self._sync_swipe_type_text_object(finalize=False)
            self._draw_swipe_type_header()
            self.status.config(text="Swipe Type: last character removed.", fg=ACCENT)
            return
        # Clear Page is a document operation: retain one page-local snapshot so
        # an immediate Undo can restore it, and Redo can clear it again.
        clear_snapshot = getattr(self, "_last_clear_snapshot", None)
        if clear_snapshot and clear_snapshot[0] == self.page_index and not self.strokes:
            snapshot = list(clear_snapshot[1])
            self.strokes.extend(snapshot)
            self.redo_strokes.append({"type": "clear_page_redo", "page_index": self.page_index})
            self._last_clear_snapshot = None
            self._redraw_board()
            self.status.config(text="Undo · cleared page restored.", fg=ACCENT)
            self._log_runtime_event("Undo restored cleared page")
            return
        if self.strokes:
            item = self.strokes.pop()
            if isinstance(item, dict) and item.get("type") == "smart_write_text":
                originals = list(item.get("original_strokes", []))
                self.strokes.extend(originals)
                self.redo_strokes.append({
                    "type": "smart_write_redo",
                    "text_object": item,
                    "original_strokes": originals,
                })
            else:
                self.redo_strokes.append(item)
            self._redraw_board()
            self.status.config(text="Undo completed.", fg=ACCENT)
        else:
            self.status.config(text="Nothing to undo on this page.", fg="#ffbd69")

    def redo(self):
        if self.redo_strokes:
            item = self.redo_strokes.pop()
            if isinstance(item, dict) and item.get("type") == "smart_write_redo":
                originals = {id(stroke) for stroke in item.get("original_strokes", [])}
                self.strokes[:] = [stroke for stroke in self.strokes if id(stroke) not in originals]
                self.strokes.append(item["text_object"])
            elif isinstance(item, dict) and item.get("type") == "clear_page_redo":
                self._last_clear_snapshot = (self.page_index, list(self.strokes))
                self.strokes.clear()
                self.status.config(text="Redo · page cleared again.", fg=ACCENT)
            else:
                self.strokes.append(item)
            self._redraw_board()
            self.status.config(text="Redo completed.", fg=ACCENT)
        else:
            self.status.config(text="Nothing to redo on this page.", fg="#ffbd69")

    def _render_page_image(self, strokes):
        image = Image.new("RGB", (self.board_w, self.board_h), (8, 14, 23))
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            if isinstance(stroke, dict) and stroke.get("type") == "smart_write_text":
                try:
                    from PIL import ImageFont
                    point_size = max(8, int(stroke.get("font_points", SMART_WRITE_FONT_POINTS)))
                    pixel_size = max(12, round(point_size * 96 / 72))
                    try:
                        font = ImageFont.truetype("arial.ttf", pixel_size)
                    except OSError:
                        font = ImageFont.load_default()
                    text_value = str(stroke.get("text", ""))
                    max_width = max(60, int(float(stroke.get("width", self.board_w - 120.0))))
                    wrapped_lines = []
                    for paragraph in text_value.splitlines() or [""]:
                        words = paragraph.split()
                        if not words:
                            wrapped_lines.append("")
                            continue
                        line = ""
                        for word in words:
                            candidate = word if not line else f"{line} {word}"
                            try:
                                candidate_width = float(font.getlength(candidate))
                            except Exception:
                                candidate_width = float(len(candidate) * pixel_size * 0.58)
                            if line and candidate_width > max_width:
                                wrapped_lines.append(line)
                                line = word
                            else:
                                line = candidate
                        if line:
                            wrapped_lines.append(line)
                    draw.multiline_text(
                        (float(stroke["x"]), float(stroke["y"])),
                        "\n".join(wrapped_lines),
                        fill=stroke.get("color", "#f4f7fb"),
                        font=font,
                        spacing=max(3, round(pixel_size * 0.20)),
                    )
                except Exception:
                    continue
                continue
            for a, b, color, width in stroke:
                try:
                    rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0,2,4))
                    draw.line((a[0], a[1], b[0], b[1]), fill=rgb, width=max(1, int(width)))
                except Exception:
                    continue
        return image

    def save_board(self):
        path = DRAWINGS_DIR / f"AirWriter_{datetime.now():%Y%m%d_%H%M%S}_page{self.page_index+1}.png"
        try:
            self._render_page_image(self.strokes).save(path)
            self.status.config(text=f"Saved PNG: {path.name}", fg=ACCENT)
            if self.mode_label is not None:
                self.mode_label.config(text=f"SAVED · {path.name}")
        except Exception as exc:
            self.status.config(text=f"Save failed: {exc}", fg="#ff7474")

    def export_pdf(self):
        path = DRAWINGS_DIR / f"AirWriter_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        try:
            images = [self._render_page_image(page).convert("RGB") for page in self.pages]
            if not images:
                images = [self._render_page_image(self.strokes).convert("RGB")]
            images[0].save(path, save_all=True, append_images=images[1:], resolution=120.0)
            self.status.config(text=f"Exported {len(images)} page(s) to {path.name}", fg=ACCENT)
        except Exception as exc:
            self.status.config(text=f"PDF export failed: {exc}", fg="#ff7474")

    def _desktop_size(self):
        """Return the coordinate size used by PyAutoGUI for real desktop input."""
        try:
            import pyautogui
            width, height = pyautogui.size()
            if width > 0 and height > 0:
                return int(width), int(height)
        except Exception:
            pass
        return max(1, self.root.winfo_screenwidth()), max(1, self.root.winfo_screenheight())

    def _desktop_point_to_tk(self, point):
        """Convert PyAutoGUI physical desktop coordinates to Tk widget coordinates."""
        if point is None:
            return None
        try:
            import pyautogui
            desktop_w, desktop_h = pyautogui.size()
            tk_w, tk_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            return (float(point[0]) * tk_w / max(1, desktop_w),
                    float(point[1]) * tk_h / max(1, desktop_h))
        except Exception:
            return float(point[0]), float(point[1])

    def _screen_point(self):
        """Project laser pixels, or hand-normalized coordinates, into desktop pixels."""
        if self.hand_mode:
            if time.time()-self.last_point_time > 0.22:
                return None
            norm = self._hand_pointer_norm
            if norm is None:
                return None
            try:
                tk_w = max(1, int(self.root.winfo_screenwidth()))
                tk_h = max(1, int(self.root.winfo_screenheight()))
                if self.running_board and self.board_canvas is not None:
                    # Hand coordinates are mapped directly into the left drawing
                    # canvas. No laser homography or spatial hand calibration is used.
                    tk_x = float(self.board_canvas.winfo_rootx()) + float(norm[0]) * max(1, self.board_canvas.winfo_width())
                    tk_y = float(self.board_canvas.winfo_rooty()) + float(norm[1]) * max(1, self.board_canvas.winfo_height())
                else:
                    tk_x = float(norm[0]) * tk_w
                    tk_y = float(norm[1]) * tk_h
                sw, sh = self._desktop_size()
                return (tk_x * sw / tk_w, tk_y * sh / tk_h)
            except Exception:
                return None
        processed_at = float(getattr(self, "_laser_point_processed_at", 0.0) or 0.0)
        if (self.laser_point is None or self.H is None or processed_at <= 0.0
                or time.time() - processed_at > LASER_ACTIVE_MAX_AGE_SECONDS):
            return None
        src = np.array([[[self.laser_point[0], self.laser_point[1]]]], dtype=np.float32)
        try:
            dst = cv2.perspectiveTransform(src, self.H)[0, 0]
            sw, sh = self._desktop_size()
            x = float(dst[0]) * sw / 1920.0
            y = float(dst[1]) * sh / 1080.0
            if not np.isfinite([x, y]).all():
                return None
            margin = max(1, min(DESKTOP_POINTER_SAFE_MARGIN, (sw - 1) // 4, (sh - 1) // 4))
            return max(margin, min(sw - 1 - margin, x)), max(margin, min(sh - 1 - margin, y))
        except Exception:
            return None

    def _point_in_board_toolbar(self, desktop_point, margin=0):
        """True when the mapped laser cursor is over or close to the board toolbar."""
        if not self.board_toolbar or desktop_point is None:
            return False
        try:
            if not self.board_toolbar.winfo_exists():
                return False
            tx, ty = self._desktop_point_to_tk(desktop_point)
            x0 = self.board_toolbar.winfo_rootx()
            y0 = self.board_toolbar.winfo_rooty()
            x1 = x0 + self.board_toolbar.winfo_width()
            y1 = y0 + self.board_toolbar.winfo_height()
            m = float(max(0, margin))
            return x0 - m <= tx <= x1 + m and y0 - m <= ty <= y1 + m
        except Exception:
            return False

    def _widget_contains_laser_point(self, widget, desktop_point, margin=8):
        """Test a mapped desktop point against a Tk widget's actual screen bounds.

        Tk and PyAutoGUI may report different sizes under Windows display scaling.
        Convert the physical desktop point to Tk screen coordinates before the hit test.
        """
        if widget is None or desktop_point is None:
            return False
        try:
            if not widget.winfo_exists() or not widget.winfo_viewable():
                return False
            tx, ty = self._desktop_point_to_tk(desktop_point)
            x0, y0 = widget.winfo_rootx(), widget.winfo_rooty()
            x1, y1 = x0 + widget.winfo_width(), y0 + widget.winfo_height()
            return x0 - margin <= tx <= x1 + margin and y0 - margin <= ty <= y1 + margin
        except Exception:
            return False

    def _laser_button_candidates(self):
        """Return all visible, enabled Tk buttons belonging to the active window.

        The board toolbar is searched recursively, not only at its first level.
        Dashboard scanning excludes other Toplevels such as camera/calibration dialogs.
        """
        root = self.board_toolbar if self.running_board else self.root
        if root is None:
            return []
        expected = self.board_win if self.running_board else self.root
        expected_path = getattr(expected, "_w", None)
        found = []

        def walk(parent):
            try:
                children = list(parent.winfo_children())
            except Exception:
                return
            for child in children:
                try:
                    if not child.winfo_exists():
                        continue
                    top = child.winfo_toplevel()
                    if expected_path is not None and getattr(top, "_w", None) != expected_path:
                        continue
                    cls = child.winfo_class()
                    if cls in ("Button", "TButton"):
                        state = str(child.cget("state")).lower()
                        if state != "disabled" and child.winfo_viewable():
                            found.append(child)
                    # Recurse through frames and other containers so nested controls
                    # are discoverable without accidentally crossing into another window.
                    walk(child)
                except Exception:
                    continue

        walk(root)
        return found

    def _find_laser_ui_button(self, desktop_point, expected_target=None):
        """Resolve the closest real button under the calibrated laser point."""
        if desktop_point is None:
            return None
        candidates = self._laser_button_candidates()
        if expected_target and str(expected_target).startswith("tk:"):
            wanted = str(expected_target)[3:]
            candidates = [widget for widget in candidates if str(widget) == wanted]
            if not candidates:
                return None

        try:
            tx, ty = self._desktop_point_to_tk(desktop_point)
        except Exception:
            return None
        ranked = []
        tolerance = float(LASER_BUTTON_HIT_TOLERANCE_PX)
        for widget in candidates:
            try:
                x0 = float(widget.winfo_rootx())
                y0 = float(widget.winfo_rooty())
                x1 = x0 + float(widget.winfo_width())
                y1 = y0 + float(widget.winfo_height())
                dx = max(x0 - tx, 0.0, tx - x1)
                dy = max(y0 - ty, 0.0, ty - y1)
                distance = math.hypot(dx, dy)
                if distance > tolerance:
                    continue
                inside = (dx == 0.0 and dy == 0.0)
                area = max(1, int(widget.winfo_width()) * int(widget.winfo_height()))
                ranked.append((0 if inside else 1, distance, area, widget))
            except Exception:
                continue
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return ranked[0][3] if ranked else None

    def _laser_click_target_key(self, desktop_point):
        """Identify the button, menu item, or desktop coordinate under the laser."""
        if desktop_point is None:
            return None

        if self.running_board and self.menu_ids:
            local = self._screen_point_to_board_canvas(desktop_point)
            if local is None:
                return None
            x, y = local
            nearest, distance = self._nearest_radial_menu_item(x, y)
            return f"menu:{nearest[0]}" if nearest is not None else None

        button = self._find_laser_ui_button(desktop_point)
        if button is not None:
            return f"tk:{str(button)}"

        # Do not treat empty toolbar space as a generic OS desktop click.
        if self.running_board and self._point_in_board_toolbar(
                desktop_point, margin=LASER_BUTTON_HIT_TOLERANCE_PX):
            # The cursor is inside the toolbar, but Tk may expose some themed or
            # composite controls through a non-Button widget class. Preserve a
            # clickable target so the OFF edge can fall back to a real desktop
            # click at that exact toolbar coordinate.
            return "toolbar-fallback"
        return "desktop"

    def _laser_invoke_ui_button(self, desktop_point, expected_target=None):
        """Run the exact Tk callback that owns the button under the laser."""
        if desktop_point is None:
            return False
        widget = self._find_laser_ui_button(desktop_point, expected_target=expected_target)
        if widget is None:
            self._log_runtime_event(
                f"Tk button dispatch missed: target={expected_target!r}; point={desktop_point!r}")
            return False
        try:
            label = str(widget.cget("text"))
        except Exception:
            label = str(widget)
        try:
            widget.invoke()
            self._log_runtime_event(f"Tk button action invoked once: {label!r} ({widget})")
            return True
        except Exception as exc:
            self._log_runtime_event(
                f"Tk button invoke failed: {label!r}: {type(exc).__name__}: {exc}")
            return False

    def _log_runtime_event(self, message):
        """Write concise diagnostics for click/menu failures without blocking the UI."""
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with RUNTIME_LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] {message}\n")
        except OSError:
            pass

    def _reset_click_signal_state(self):
        """Re-arm click recognition without touching camera detection or menu state."""
        self._laser_on_since = None
        self._laser_was_active = False
        self._laser_active_samples = 0
        self._laser_off_since = None
        self._last_mapped_point = None
        self._laser_last_counted_frame_stamp = 0.0
        self._last_mapped_was_toolbar = False
        self._drag_off_since = None
        gate = getattr(self, "_laser_dwell_gate", None)
        if gate is not None:
            gate.reset()

    def _cancel_pending_desktop_click(self):
        """Cancel a queued single click when mouse control is disabled or app closes."""
        after_id = getattr(self, "_pending_desktop_click_after_id", None)
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._pending_desktop_click_after_id = None
        self._pending_desktop_click = None

    def _get_click_operations_mouse(self):
        """Load click_operations.py's one-shot Mouse executor on first use only."""
        if self._click_operations_mouse is not None:
            return self._click_operations_mouse
        module_path = (ROOT / "click_operations.py").resolve()
        if not module_path.is_file():
            raise FileNotFoundError(f"Required click executor is missing: {module_path}")
        module_name = "_airwrite_click_operations_" + hashlib.sha1(
            str(module_path).encode("utf-8")).hexdigest()[:12]
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load click executor: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        mouse_type = getattr(module, "Mouse", None)
        if not callable(mouse_type):
            raise ImportError("click_operations.py does not expose Mouse")
        self._click_operations_module = module
        self._click_operations_mouse = mouse_type(dry_run=False)
        self._log_runtime_event(f"Loaded one-shot click executor from {module_path.name}")
        return self._click_operations_mouse

    def _perform_desktop_click(self, x, y, button="left"):
        """Perform exactly one OS click, explicitly releasing the button on every path."""
        button = str(button).lower().strip()
        if button not in ("left", "right", "middle"):
            self._log_runtime_event(f"Rejected unsupported desktop click button: {button!r}")
            return False
        if not self.mouse_enabled or self._desktop_mouse_suspended:
            self._log_runtime_event(
                f"Skipped desktop {button}-click: mouse input is disabled or safety-suspended")
            self.status.config(
                text="Desktop click unavailable: enable mouse control and clear any safety pause.",
                fg="#ffbd69")
            return False

        import pyautogui
        width, height = pyautogui.size()
        margin = max(1, int(DESKTOP_POINTER_SAFE_MARGIN))
        sx = max(margin, min(int(width) - 1 - margin, int(round(x))))
        sy = max(margin, min(int(height) - 1 - margin, int(round(y))))
        try:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0
            # A real OS click is used only for non-Tk targets. Tk buttons use
            # widget.invoke() above, so app actions cannot be lost to window focus.
            click_sent = self._get_click_operations_mouse().click(button, sx, sy)
            if not click_sent:
                self._log_runtime_event(
                    f"click_operations suppressed {button}-click at ({sx},{sy})")
                return False
            self._log_runtime_event(f"Desktop {button}-click executed at ({sx},{sy}) via click_operations.py")
            self.status.config(
                text=("Right-click executed after a 0.95s stationary hold." if button == "right"
                      else "Left-click executed." if button == "left"
                      else "Middle-click executed."),
                fg=ACCENT)
            return True
        except Exception as exc:
            if type(exc).__name__ == "FailSafeException":
                self._desktop_mouse_suspended = True
                self._log_runtime_event("PyAutoGUI fail-safe during click; desktop input suspended")
                self.status.config(
                    text="Desktop mouse safety paused. Move the physical mouse away from the screen corner, then toggle mouse control off/on.",
                    fg="#ffbd69")
            else:
                self._log_runtime_event(
                    f"Desktop {button}-click failed: {type(exc).__name__}: {exc}")
                self.status.config(text=f"Desktop {button}-click failed: {exc}", fg="#ff7474")
            return False

    def _register_desktop_click_cycle(self, x, y, now, held_seconds=0.0):
        """Dispatch exactly one left-click on every confirmed laser OFF edge."""
        success = self._perform_desktop_click(x, y, button="left")
        return "left-click" if success else "left-click-failed"

    def _handle_mouse_signal(self):
        """Process one debounced ON→OFF action; board menus use internal dispatch."""
        if not self.mouse_enabled or self.H is None or self.background_gray is None or not self.camera_confirmed:
            if self._drag_down:
                try:
                    import pyautogui
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self._drag_down = False
            self._laser_on_since = None
            self._laser_was_active = False
            self._laser_active_samples = 0
            self._laser_off_since = None
            self._drag_off_since = None
            self._last_mapped_was_toolbar = False
            self._laser_dwell_gate.reset()
            return
        point = self._screen_point()
        now = time.time()
        active = point is not None and self.laser_point is not None
        # Resolve the target from each fresh camera sample before applying the
        # drawing-mode guard. Toolbar buttons must remain clickable from every tool.
        target_key = self._laser_click_target_key(point) if active else None
        target_is_button = bool(target_key and str(target_key).startswith("tk:"))
        toolbar_now = bool(
            self.running_board and not self.menu_ids and
            (self._point_in_board_toolbar(point, margin=LASER_BUTTON_HIT_TOLERANCE_PX) or target_is_button)
        )
        toolbar_target = bool(
            self.running_board and not self.menu_ids and
            (toolbar_now or (self._last_mapped_was_toolbar and not active))
        )
        # In drawing modes, laser OFF ends the stroke instead of clicking the
        # desktop. The radial menu and toolbar are exceptions so the user can
        # reach every option without ever touching a keyboard.
        if self.running_board and self.board_mode not in ("CURSOR", "DRAG", "SCROLL") and not self.menu_ids and not toolbar_target:
            self._laser_on_since = None
            self._laser_was_active = False
            self._laser_active_samples = 0
            self._laser_off_since = None
            self._last_mapped_point = None
            self._last_mapped_was_toolbar = False
            self._laser_dwell_gate.reset()
            if self._drag_down:
                try:
                    import pyautogui
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self._drag_down = False
            return
        if self._wait_for_laser_off:
            if active:
                self._wait_for_laser_off_since = None
            else:
                if self._wait_for_laser_off_since is None:
                    self._wait_for_laser_off_since = now
                elif now - self._wait_for_laser_off_since >= 0.45:
                    self._wait_for_laser_off = False
                    self._wait_for_laser_off_since = None
                    self._laser_on_since = None
                    self._laser_was_active = False
                    self._laser_active_samples = 0
                    self._laser_off_since = None
                    self._last_mapped_point = None
                    self._last_mapped_was_toolbar = False
                    self.status.config(text=f"Cursor armed · aim, then laser OFF = one left-click per cycle.", fg=ACCENT)
            # Never convert the final calibration target into an accidental click.
            self._laser_on_since = None
            self._laser_was_active = False
            self._laser_active_samples = 0
            self._laser_off_since = None
            self._last_mapped_point = None
            self._laser_dwell_gate.reset()
            return
        # A radial menu is opened while the laser is still tracing the Z gesture.
        # Ignore the Z gesture's release. The user must release the laser and
        # wait briefly, then point to an option and perform a fresh ON -> OFF.
        if self.running_board and self.menu_ids and self._menu_waiting_for_release:
            if active:
                self._menu_off_since = None
            else:
                if self._menu_off_since is None:
                    self._menu_off_since = now
                elif now - self._menu_off_since >= 0.32:
                    self._menu_waiting_for_release = False
                    self._menu_off_since = None
                    self._laser_on_since = None
                    self._laser_was_active = False
                    self._laser_active_samples = 0
                    self._laser_off_since = None
                    self._last_mapped_point = None
                    self._suppress_next_laser_click = False
                    self.status.config(
                        text=f"Radial menu ready · aim at one item, then switch the laser OFF to select.",
                        fg=ACCENT)
            # The menu-opening Z must never be treated as an item click.
            self._laser_on_since = None
            self._laser_was_active = False
            self._laser_active_samples = 0
            self._laser_off_since = None
            self._last_mapped_point = None
            self._laser_dwell_gate.reset()
            return
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0
            if self.running_board and self.board_mode == "DRAG" and not self.menu_ids and not toolbar_target:
                if active:
                    self._drag_off_since = None
                    pyautogui.moveTo(int(point[0]), int(point[1]), _pause=False)
                    if not self._drag_down:
                        pyautogui.mouseDown(button="left")
                        self._drag_down = True
                elif self._drag_down:
                    if self._drag_off_since is None:
                        self._drag_off_since = now
                    if now-self._drag_off_since >= 0.18:
                        pyautogui.mouseUp(button="left")
                        self._drag_down = False
                        self._drag_off_since = None
                self._laser_dwell_gate.reset()
                return
            if self.running_board and self.board_mode == "SCROLL" and not self.menu_ids and not toolbar_target:
                if active:
                    pyautogui.moveTo(int(point[0]), int(point[1]), _pause=False)
                # Scroll/pan is a movement gesture; switching the laser OFF must not click.
                self._laser_on_since = None
                self._laser_was_active = False
                self._laser_active_samples = 0
                self._laser_off_since = None
                self._last_mapped_point = None
                self._laser_dwell_gate.reset()
                return
            if active:
                self._laser_off_since = None
                # The Smart Board draws its own precise reticle. Moving the global
                # OS cursor is needed only for desktop control / DRAG, not for board
                # drawing or radial-menu selection. This prevents routine board use
                # from reaching PyAutoGUI's screen-corner fail-safe.
                desktop_pointer_needed = (
                    not self.running_board
                    or (self.running_board and not self.menu_ids and self.board_mode in ("CURSOR", "DRAG"))
                )
                if desktop_pointer_needed and not self._desktop_mouse_suspended:
                    pyautogui.moveTo(int(point[0]), int(point[1]), _pause=False)
                if self._laser_on_since is None:
                    self._laser_on_since = now
                    self._laser_active_samples = 0
                frame_stamp = float(self.last_point_time or 0.0)
                is_new_detection = frame_stamp > self._laser_last_counted_frame_stamp + 1e-6
                if is_new_detection:
                    self._laser_active_samples += 1
                    self._last_mapped_point = point
                    self._last_mapped_was_toolbar = bool(toolbar_target)
                    self._laser_dwell_gate.observe(frame_stamp, point, target_key)
                    self._laser_last_counted_frame_stamp = frame_stamp
                elif self._last_mapped_point is None:
                    # Keep a valid target coordinate available for the OFF edge,
                    # even if the current camera frame was a brief detector miss.
                    self._last_mapped_point = point
                    self._last_mapped_was_toolbar = bool(toolbar_target)
                self._laser_was_active = True
            elif self._laser_was_active:
                if self._laser_off_since is None:
                    self._laser_off_since = now
                # A fresh laser spot over a target followed by OFF is one click.
                # Keep the brief OFF confirmation and cycle cooldown for debouncing,
                # but do not require a dwell timer before dispatching the action.
                if now-self._laser_off_since >= LASER_OFF_CONFIRM_SECONDS:
                    held_seconds = self._laser_dwell_gate.held_seconds
                    click_point = self._laser_dwell_gate.stable_point or self._last_mapped_point
                    held_target = self._laser_dwell_gate.target
                    enough_on_evidence = self._laser_active_samples >= LASER_MIN_ACTIVE_SAMPLES
                    if enough_on_evidence and click_point is not None and now-self.last_mouse_click >= MOUSE_CLICK_COOLDOWN_SECONDS:
                        x, y = int(round(click_point[0])), int(round(click_point[1]))
                        action_performed = False
                        click_result = "internal-action"
                        held_target = self._laser_dwell_gate.target
                        if self.running_board and self.menu_ids and str(held_target).startswith("menu:"):
                            action_performed = bool(self._select_radial_item_at_screen_point((x, y)))
                            click_result = "radial-menu-selection" if action_performed else "radial-menu-selection-failed"
                        elif held_target and str(held_target).startswith("tk:"):
                            action_performed = self._laser_invoke_ui_button((x, y), expected_target=held_target)
                            if action_performed:
                                click_result = "tk-button-invoke"
                            else:
                                action_performed = self._perform_desktop_click(x, y, button="left")
                                click_result = "tk-button-os-click-fallback" if action_performed else "tk-button-invoke-failed"
                        elif held_target == "toolbar-fallback" and self.running_board and not self._desktop_mouse_suspended:
                            action_performed = self._perform_desktop_click(x, y, button="left")
                            click_result = "toolbar-os-click-fallback" if action_performed else "toolbar-os-click-fallback-failed"
                        elif held_target == "desktop" and not self._desktop_mouse_suspended and not (
                                self.running_board and self.board_mode in ("PENCIL", "HIGHLIGHTER", "ERASER", "SHAPES", "SWIPE", "SCROLL")):
                            click_result = self._register_desktop_click_cycle(x, y, now, held_seconds)
                            action_performed = click_result in ("left-click", "right-click")
                        elif self._desktop_mouse_suspended:
                            self.status.config(
                                text="Desktop cursor safety is paused. Board drawing and radial-menu actions remain available; move the physical mouse away from a screen corner, then toggle mouse control off/on to resume.",
                                fg="#ffbd69")
                        if action_performed:
                            self.last_mouse_click = now
                            self._log_runtime_event(
                                f"ON->OFF action accepted once at ({x},{y}); hold={held_seconds:.3f}s; detail={click_result}; board={self.running_board}; menu_open={bool(self.menu_ids)}")
                        elif self.running_board and self.menu_ids:
                            self._log_runtime_event(
                                f"ON->OFF menu selection rejected at ({x},{y}); target={self._laser_dwell_gate.target!r}; hold={held_seconds:.3f}s")
                    else:
                        self._log_runtime_event(
                            f"ON->OFF ignored: no fresh target point or click cycle is cooling down; target={self._laser_dwell_gate.target!r}; last_point={self._last_mapped_point!r}")
                    self._laser_on_since = None
                    self._laser_was_active = False
                    self._laser_active_samples = 0
                    self._laser_off_since = None
                    self._last_mapped_was_toolbar = False
                    self._laser_dwell_gate.reset()
        except Exception as exc:
            name = type(exc).__name__
            if name == "FailSafeException":
                # Do not disable the entire app: that made all radial-menu clicks
                # stop after a cursor reached a screen corner. Suspend OS mouse
                # injection only; internal board/menu actions remain operational.
                self._desktop_mouse_suspended = True
                self._reset_click_signal_state()
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self._log_runtime_event("PyAutoGUI fail-safe triggered; desktop cursor injection suspended; board interactions retained")
                self.status.config(
                    text="Desktop cursor safety paused OS mouse movement. Smart Board menu and drawing stay active. Move the physical mouse away from the screen corner, then toggle mouse control off/on to resume desktop control.",
                    fg="#ffbd69")
            elif name not in ("KeyboardInterrupt",):
                self._log_runtime_event(f"Mouse control exception: {type(exc).__name__}: {exc}")
                self.status.config(text=f"Mouse control warning: {exc}", fg="#ffbd69")

    def _draw_board_laser_cursor(self, cx, cy):
        """Visible reticle follows the laser in every mode, including Pen/Erase and menus."""
        if not self.board_canvas:
            return
        if self.board_mode == "ERASER":
            color = "#ff8585"
        elif self.board_mode in ("PENCIL", "SMART_WRITE"):
            color = self.pen_color if self.pen_color.lower() not in ("#090b10", "#000000") else "#55a7ff"
        elif self.board_mode == "HIGHLIGHTER":
            color = "#9fe870"
        elif self.board_mode == "SHAPES":
            color = "#ffd166"
        else:
            color = ACCENT
        self.board_canvas.delete("laser-cursor")
        radius = 11
        self.board_canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                                      outline=color, width=2, tags=("laser-cursor",))
        self.board_canvas.create_line(cx-17, cy, cx-5, cy, fill=color, width=1, tags=("laser-cursor",))
        self.board_canvas.create_line(cx+5, cy, cx+17, cy, fill=color, width=1, tags=("laser-cursor",))
        self.board_canvas.create_line(cx, cy-17, cx, cy-5, fill=color, width=1, tags=("laser-cursor",))
        self.board_canvas.create_line(cx, cy+5, cx, cy+17, fill=color, width=1, tags=("laser-cursor",))
        self.board_canvas.create_oval(cx-2, cy-2, cx+2, cy+2,
                                      fill="#ffffff", outline="", tags=("laser-cursor",))
        self.board_canvas.tag_raise("laser-cursor")

    def _smooth_board_stroke_point(self, x, y, now):
        """Adaptively smooth pen coordinates without slowing cursor/menu gestures.

        Slow movements get stronger filtering to suppress hand tremor; fast
        intentional movements get a higher response so corners and letters stay
        responsive. The raw calibrated point is still used for click/menu logic.
        """
        point = (float(x), float(y))
        previous = self.last_board_point
        dt = max(1.0 / 120.0, min(0.12, now - self._stroke_filter_time)) if self._stroke_filter_time else 1.0 / 30.0
        self._stroke_filter_time = now
        if previous is None:
            return point
        dx, dy = point[0] - previous[0], point[1] - previous[1]
        distance = math.hypot(dx, dy)
        # Ignore sub-pixel-equivalent jitter in the board's logical coordinate
        # space. This threshold is intentionally small to preserve fine writing.
        if distance < 0.65:
            return previous
        speed = distance / dt
        alpha = 0.22 + 0.56 * min(1.0, speed / 1000.0)
        return (previous[0] + alpha * dx, previous[1] + alpha * dy)

    def _open_radial_menu_from_z(self, fallback_point=None):
        """Open the laser radial menu once a complete Z path is recognized."""
        if self.hand_mode or not self.running_board or self.board_canvas is None or self.menu_ids:
            return False
        now = time.time()
        if now - float(getattr(self, "last_menu_trigger", 0.0)) <= 1.4:
            return False

        menu_at = self._last_menu_gesture_center or fallback_point or self._last_board_canvas_point
        if menu_at is None:
            return False
        try:
            menu_at = (float(menu_at[0]), float(menu_at[1]))
            if not all(math.isfinite(v) for v in menu_at):
                return False
        except Exception:
            return False

        self.last_menu_trigger = now
        old_mode = self.board_mode
        # A Z trace is a command, never artwork or typed content.
        if old_mode in ("PENCIL", "SMART_WRITE", "HIGHLIGHTER", "ERASER") and self.current_stroke is not None:
            if self.strokes and self.strokes[-1] is self.current_stroke:
                self.strokes.pop()
            self.current_stroke = None
            self._redraw_board()
        if old_mode == "SHAPES":
            self.board_canvas.delete("shape-preview")
            self.shape_start = self.shape_end = None
        if old_mode == "SWIPE":
            self.swipe_start = self.swipe_last = None
        self.scroll_last_point = None
        if self._drag_down:
            try:
                import pyautogui
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
            self._drag_down = False
        if old_mode == "SWIPE_TYPE":
            self.set_mode("CURSOR")

        # Show the menu immediately, centered near the Z, and quarantine the
        # Z gesture's own release so it cannot select a menu item accidentally.
        self.show_radial_menu(at=menu_at, page="main")
        self._log_runtime_event(
            f"Laser Z gesture accepted: center=({menu_at[0]:.1f},{menu_at[1]:.1f}), previous_mode={old_mode}")
        self._suppress_next_laser_click = False
        self.menu_gesture_points = []
        self._last_menu_gesture_center = None
        self.last_board_point = None
        try:
            if fallback_point is not None:
                self._draw_board_laser_cursor(float(fallback_point[0]), float(fallback_point[1]))
            self.board_canvas.tag_raise("laser-cursor")
        except Exception:
            pass
        self.status.config(
            text="Z detected · radial menu opened. Switch laser OFF, wait for MENU READY, then aim at an item and switch OFF to select.",
            fg=ACCENT)
        return True

    def _board_tick(self):
        if not self.board_canvas or not self.board_win:
            return
        # Switching from the dashboard to the fullscreen board must not reuse
        # the same laser ON movement as a drawing stroke or Z gesture. Still draw
        # the reticle while waiting: the user must be able to SEE that tracking works.
        waiting_for_release = bool(self._wait_for_laser_off and not self.hand_mode)
        if waiting_for_release:
            self.board_canvas.delete("active-stroke-preview")
            if self.current_stroke is not None:
                self._redraw_board()
            self.last_board_point = None
            self.current_stroke = None
            self._stroke_filter_time = 0.0
            self.menu_gesture_points = []
            self._last_menu_gesture_center = None
        p = self._screen_point()
        now = time.time()
        if p is None:
            # A tiny gap in verified cursor frames must not end a glide word
            # immediately; confirm the release before predicting the word.
            if self.board_mode == "SWIPE_TYPE" and self.swipe_keyboard_open and self.swipe_type_active:
                if now - self.swipe_type_last_seen >= SWIPE_TYPE_RELEASE_GRACE_SECONDS:
                    self._finish_swipe_type_gesture(now)
                else:
                    self.board_canvas.delete("laser-cursor")
                    return
            # Last-frame fallback: some cameras drop the final spot frame just
            # before the next 75 ms classifier tick. Check the just-finished path
            # on release so a valid Z is not lost at the endpoint.
            if (not self.hand_mode and not self.menu_ids and self.menu_gesture_points
                    and now - self.menu_gesture_points[-1][2] <= min(1.20, MENU_GESTURE_TRACKING_GAP_SECONDS)
                    and now - float(getattr(self, "last_menu_trigger", 0.0)) > 1.4):
                if self._looks_like_z_gesture():
                    last_xy = self.menu_gesture_points[-1][:2]
                    if self._open_radial_menu_from_z(last_xy):
                        self.board_canvas.delete("laser-cursor")
                        return
            # Hide the cursor reticle when the spot is no longer fresh; retain
            # the last local coordinate briefly for ON -> OFF menu selection.
            self.board_canvas.delete("laser-cursor")
            # Finish a pending shape exactly when the laser is lifted.
            if self.board_mode == "SHAPES" and self.shape_start is not None and self.shape_end is not None:
                segments = self._shape_segments(self.shape_start, self.shape_end, self.shape_type)
                if segments:
                    self.strokes.append(segments)
                    self.redo_strokes = []
                    self._redraw_board()
                self.board_canvas.delete("shape-preview")
                self.shape_start = self.shape_end = None
            if self.board_mode == "SWIPE" and self.swipe_start is not None and self.swipe_last is not None:
                self._finish_swipe_gesture()
            if self.current_stroke is not None:
                if self.board_mode == "SMART_WRITE":
                    self._queue_smart_write_stroke(self.current_stroke, now)
                self.board_canvas.delete("active-stroke-preview")
                self._redraw_board()
            self.last_board_point = None
            self.current_stroke = None
            self._stroke_filter_time = 0.0
            self.scroll_last_point = None
            if self.menu_gesture_points and now-self.menu_gesture_points[-1][2] > MENU_GESTURE_TRACKING_GAP_SECONDS:
                self.menu_gesture_points = []
            self._last_menu_gesture_center = None
            if (self.board_mode == "SMART_WRITE" and self.current_stroke is None
                    and self._smart_write_pending_strokes
                    and now - self._smart_write_last_stroke_time >= SMART_WRITE_PAUSE_SECONDS):
                self._process_smart_write_pending()
            return
        p_tk = self._desktop_point_to_tk(p)
        left = self.board_canvas.winfo_rootx()
        top = self.board_canvas.winfo_rooty()
        cw, ch = max(1,self.board_canvas.winfo_width()), max(1,self.board_canvas.winfo_height())
        cx, cy = p_tk[0]-left, p_tk[1]-top
        if not (0 <= cx < cw and 0 <= cy < ch):
            self.board_canvas.delete("laser-cursor")
            self._last_board_canvas_point = None
            if self.board_mode == "SWIPE" and self.swipe_start is not None:
                self._finish_swipe_gesture()
            if self.current_stroke is not None:
                if self.board_mode == "SMART_WRITE":
                    self._queue_smart_write_stroke(self.current_stroke, now)
                self.board_canvas.delete("active-stroke-preview")
                self._redraw_board()
            self.last_board_point = None
            self.current_stroke = None
            self._stroke_filter_time = 0.0
            self.menu_gesture_points = []
            self._last_menu_gesture_center = None
            return
        self._last_board_canvas_point = (float(cx), float(cy))
        self._draw_board_laser_cursor(cx, cy)
        if waiting_for_release:
            self.last_board_point = None
            self.current_stroke = None
            self._stroke_filter_time = 0.0
            self.menu_gesture_points = []
            self._last_menu_gesture_center = None
            return
        # Content coordinates are offset by the board viewport pan state.
        zoom = float(getattr(self, "_hand_zoom_scale", 1.0)) if self.hand_mode else 1.0
        origin_x = cw * (1.0 - zoom) * 0.5 + self.pan_x
        origin_y = ch * (1.0 - zoom) * 0.5 + self.pan_y
        x = max(0, min(self.board_w-1, (cx-origin_x)*self.board_w/(cw*zoom)))
        y = max(0, min(self.board_h-1, (cy-origin_y)*self.board_h/(ch*zoom)))

        # A deliberate Z-shaped gesture (top-left → top-right → bottom-left → bottom-right) is the universal escape gesture from EVERY laser tool mode. The recognizer keeps a 12-second path window and tolerates brief pauses between Z strokes. This is
        # essential for keyboard-free use: users must be able to leave Pen/Erase/
        # Shapes/Drag/Scroll/Swipe without becoming trapped in that mode.
        # In SHAPES mode ordinary geometry remains available; the Z gesture is large and
        # directional to reduce confusion with normal shape drawing. The toolbar
        # menu remains available as a visible fallback.
        if not self.menu_ids and not self.hand_mode:
            # Hand mode has its own verified three-finger tools gesture. The Z
            # recognizer remains laser-only so a hand-drawn letter Z is not stolen.
            # When Swipe Type is active, reserve the keyboard panel for typing.
            # The universal Z gesture remains available in the blank workspace
            # above it, so the user can always return to the full radial menu.
            gesture_zone_allowed = True
            if self.board_mode == "SWIPE_TYPE" and self.swipe_keyboard_open:
                keyboard_top = float(getattr(self, "swipe_type_key_y_top", ch * 0.68))
                gesture_zone_allowed = cy < keyboard_top - 10.0
                if not gesture_zone_allowed:
                    self.menu_gesture_points = []
                    self._last_menu_gesture_center = None
            if gesture_zone_allowed:
                self.menu_gesture_points.append((cx, cy, now))
                self.menu_gesture_points = [item for item in self.menu_gesture_points if now-item[2] <= MENU_GESTURE_WINDOW_SECONDS][-420:]
                # Run the relatively detailed geometry classifier at a bounded
                # rate; checking every camera frame can create latency over long
                # sessions, especially when the user is drawing slowly.
                gesture_eval_due = now - float(getattr(self, "_last_menu_gesture_eval", 0.0)) >= 0.075
                if now-self.last_menu_trigger > 1.4 and gesture_eval_due:
                    self._last_menu_gesture_eval = now
                    if not self._looks_like_z_gesture():
                        menu_match = False
                    else:
                        menu_match = True
                else:
                    menu_match = False
                if menu_match:
                    if self._open_radial_menu_from_z((cx, cy)):
                        return
        if self.menu_ids:
            self._radial_hover(cx, cy)
            self._draw_board_laser_cursor(cx, cy)
            self.last_board_point = None
            return

        if self.board_mode == "SWIPE_TYPE" and self.swipe_keyboard_open:
            self._update_swipe_type_point(cx, cy, now)
            self.last_board_point = None
            return

        # In Hand mode, pointing only moves the reticle. A confirmed pinch is
        # required before pen/eraser/shape strokes can alter the board.
        if (self.hand_mode and not self._hand_pinch_active
                and self.board_mode in ("PENCIL", "SMART_WRITE", "HIGHLIGHTER", "ERASER", "SHAPES", "SWIPE")):
            if self.current_stroke is not None:
                self._finish_hand_stroke()
            self.last_board_point = None
            self._draw_board_laser_cursor(cx, cy)
            return

        if self.board_mode == "SCROLL":
            if self.scroll_last_point is not None:
                dy = cy-self.scroll_last_point[1]
                if abs(dy) >= 2:
                    self.pan_y = max(-ch*1.5, min(ch*1.5, self.pan_y+dy))
                    self._redraw_board()
            self.scroll_last_point = (cx, cy)
            self._draw_board_laser_cursor(cx, cy)
            self.last_board_point = None
            return

        if self.board_mode == "SWIPE":
            if self.swipe_start is None:
                self.swipe_start = (cx, cy, now)
            self.swipe_last = (cx, cy, now)
            self._draw_board_laser_cursor(cx, cy)
            self.last_board_point = None
            return

        if self.board_mode == "SHAPES":
            if self.shape_start is None:
                self.shape_start = (x, y)
            self.shape_end = (x, y)
            self._draw_shape_preview()
            self._draw_board_laser_cursor(cx, cy)
            self.last_board_point = None
            return

        if self.board_mode in ("PENCIL", "SMART_WRITE", "HIGHLIGHTER", "ERASER"):
            if self.board_mode == "ERASER":
                color, width = "#080e17", self.eraser_size
            elif self.board_mode == "HIGHLIGHTER":
                color, width = "#9fe870", max(12, self.pen_size * 3)
            else:
                color, width = self.pen_color, self.pen_size
            if self.last_board_point is not None:
                ax, ay = self.last_board_point
                raw_distance = math.hypot(x-ax, y-ay)
                if raw_distance >= 110:
                    # Never bridge a large tracking jump with ink. Commit the
                    # existing segment chain, break the stroke, and re-anchor.
                    if self.current_stroke is not None:
                        self.board_canvas.delete("active-stroke-preview")
                        self._redraw_board()
                    self.current_stroke = None
                    self.last_board_point = (x, y)
                    self._stroke_filter_time = now
                    self._draw_board_laser_cursor(cx, cy)
                    return
            fx, fy = self._smooth_board_stroke_point(x, y, now)
            if self.last_board_point is not None:
                ax, ay = self.last_board_point
                if math.hypot(fx-ax, fy-ay) >= 0.35:
                    segment_width = width * 2 if self.board_mode == "ERASER" else width
                    segment = ((ax, ay), (fx, fy), color, segment_width)
                    if self.current_stroke is None:
                        self.current_stroke = []
                        self.strokes.append(self.current_stroke)
                        self._freehand_stroke_ids.add(id(self.current_stroke))
                        self.redo_strokes = []
                    self.current_stroke.append(segment)
                    self._draw_live_stroke_preview(cw, ch)
            self.last_board_point = (fx, fy)
            self._draw_board_laser_cursor(cx, cy)
            return

        self.last_board_point = None
        self.current_stroke = None
        if self.board_mode in ("CURSOR", "DRAG"):
            self._draw_board_laser_cursor(cx, cy)

    def _draw_live_stroke_preview(self, cw=None, ch=None):
        """Render the whole current freehand stroke as one smooth spline.

        Drawing each camera sample as an independent two-point line makes Tk's
        smooth=True ineffective and creates visibly jagged joins. Buffering the
        stroke and rendering a single rounded spline removes those joins while
        retaining the original points for undo/export.
        """
        if not self.board_canvas or not self.current_stroke:
            return
        cw = max(1, int(cw or self.board_canvas.winfo_width()))
        ch = max(1, int(ch or self.board_canvas.winfo_height()))
        first = self.current_stroke[0]
        points = [first[0]] + [segment[1] for segment in self.current_stroke]
        if len(points) < 2:
            return
        coords = []
        for px, py in points:
            coords.extend((px*cw/self.board_w+self.pan_x,
                           py*ch/self.board_h+self.pan_y))
        color = self.current_stroke[-1][2]
        width = self.current_stroke[-1][3]
        self.board_canvas.delete("active-stroke-preview")
        self.board_canvas.create_line(
            *coords, fill=color,
            width=max(1, width*max(cw/self.board_w, 0.5)),
            capstyle="round", joinstyle="round", smooth=True, splinesteps=16,
            tags=("active-stroke-preview",))
        self.board_canvas.tag_raise("active-stroke-preview")

    def _shape_segments(self, start, end, shape_type):
        x1, y1 = map(float, start); x2, y2 = map(float, end)
        color, width = self.pen_color, self.pen_size
        if math.hypot(x2-x1, y2-y1) < 5:
            return []
        if shape_type == "Line":
            return [((x1,y1),(x2,y2),color,width)]
        if shape_type == "Rectangle":
            pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
            return [(pts[i],pts[i+1],color,width) for i in range(4)]
        if shape_type == "Triangle":
            pts = [((x1+x2)/2.0,y1),(x2,y2),(x1,y2),((x1+x2)/2.0,y1)]
            return [(pts[i],pts[i+1],color,width) for i in range(len(pts)-1)]
        if shape_type == "Diamond":
            pts = [((x1+x2)/2.0,y1),(x2,(y1+y2)/2.0),((x1+x2)/2.0,y2),(x1,(y1+y2)/2.0),((x1+x2)/2.0,y1)]
            return [(pts[i],pts[i+1],color,width) for i in range(len(pts)-1)]
        if shape_type == "Star":
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            rx, ry = abs(x2-x1)/2.0, abs(y2-y1)/2.0
            pts=[]
            for i in range(10):
                radius = 1.0 if i % 2 == 0 else 0.42
                angle = -math.pi/2 + i*math.pi/5
                pts.append((cx+rx*radius*math.cos(angle), cy+ry*radius*math.sin(angle)))
            pts.append(pts[0])
            return [(pts[i],pts[i+1],color,width) for i in range(len(pts)-1)]
        if shape_type == "Arrow":
            length = max(1.0, math.hypot(x2-x1,y2-y1))
            angle = math.atan2(y2-y1,x2-x1)
            head = min(35.0, max(14.0, length*0.18))
            a1 = angle + math.pi*0.82
            a2 = angle - math.pi*0.82
            p1=(x2+head*math.cos(a1), y2+head*math.sin(a1))
            p2=(x2+head*math.cos(a2), y2+head*math.sin(a2))
            return [((x1,y1),(x2,y2),color,width),((x2,y2),p1,color,width),((x2,y2),p2,color,width)]
        # Ellipse represented as connected line segments so undo and exports stay consistent.
        cx, cy = (x1+x2)/2, (y1+y2)/2
        rx, ry = abs(x2-x1)/2, abs(y2-y1)/2
        pts = [(cx+rx*math.cos(t), cy+ry*math.sin(t)) for t in np.linspace(0, 2*math.pi, 41)]
        return [(pts[i],pts[i+1],color,width) for i in range(len(pts)-1)]

    def _draw_shape_preview(self):
        if not self.board_canvas or self.shape_start is None or self.shape_end is None:
            return
        self.board_canvas.delete("shape-preview")
        cw, ch = max(1,self.board_canvas.winfo_width()), max(1,self.board_canvas.winfo_height())
        for a,b,color,width in self._shape_segments(self.shape_start, self.shape_end, self.shape_type):
            self.board_canvas.create_line(a[0]*cw/self.board_w+self.pan_x, a[1]*ch/self.board_h+self.pan_y,
                                          b[0]*cw/self.board_w+self.pan_x, b[1]*ch/self.board_h+self.pan_y,
                                          fill=color, width=max(1,width*cw/self.board_w),
                                          capstyle="round", joinstyle="round", tags=("shape-preview",))

    def _finish_swipe_gesture(self):
        start, end = self.swipe_start, self.swipe_last
        self.swipe_start = self.swipe_last = None
        if not start or not end:
            return
        dx, dy = end[0]-start[0], end[1]-start[1]
        duration = end[2]-start[2]
        if abs(dx) >= 140 and abs(dy) <= 110 and 0.12 <= duration <= 2.5:
            if dx < 0:
                self._go_to_page(self.page_index+1, create=True)
            elif self.page_index > 0:
                self._go_to_page(self.page_index-1)
            self.status.config(text=("Swipe left: next page" if dx < 0 else "Swipe right: previous page"), fg=ACCENT)

    def _looks_like_z_gesture(self):
        """Recognize a natural, slightly imperfect Z drawn with the laser.

        This classifier is intentionally tolerant of hand tremor, rounded
        corners, tilted strokes and unequal arm lengths. It still requires the
        sequence left-to-right, down-left, left-to-right, so lines, Vs and Us
        do not open the menu.
        """
        self._last_menu_gesture_center = None
        raw = self.menu_gesture_points
        if len(raw) < 12:
            return False
        try:
            xy_all = np.asarray([[float(p[0]), float(p[1])] for p in raw], dtype=np.float32)
            ts_all = np.asarray([float(p[2]) for p in raw], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            return False
        if not np.isfinite(xy_all).all() or not np.isfinite(ts_all).all():
            return False

        # Remove only frame-to-frame jitter for classification. The actual
        # cursor, drawing and laser coordinates are not changed.
        smooth_all = xy_all.copy()
        if len(xy_all) >= 5:
            for i in range(1, len(xy_all) - 1):
                smooth_all[i] = np.median(xy_all[i-1:i+2], axis=0)

        try:
            canvas_w = max(1.0, float(self.board_canvas.winfo_width())) if self.board_canvas else 1280.0
            canvas_h = max(1.0, float(self.board_canvas.winfo_height())) if self.board_canvas else 720.0
        except Exception:
            canvas_w, canvas_h = 1280.0, 720.0

        min_width = max(65.0, min(MENU_GESTURE_MIN_WIDTH, canvas_w * 0.075))
        min_depth = max(42.0, min(MENU_GESTURE_MIN_DEPTH, canvas_h * 0.075))
        max_width = min(0.88 * canvas_w, 1500.0)
        n = len(smooth_all)
        first_start = max(0, n - 420)

        # Try recent suffixes so aiming the pointer at the Z start is not
        # mistaken for part of the gesture.
        for start_idx in range(n - 12, first_start - 1, -1):
            raw_sub = smooth_all[start_idx:]
            times = ts_all[start_idx:]
            if len(raw_sub) < 12:
                continue
            duration = float(times[-1] - times[0])
            if duration < MENU_GESTURE_MIN_DURATION or duration > MENU_GESTURE_MAX_DURATION:
                continue

            # Collapse near-identical samples, then resample by travelled
            # distance. Drawing speed therefore does not affect recognition.
            deltas = np.linalg.norm(np.diff(raw_sub, axis=0), axis=1)
            keep = np.concatenate(([True], deltas >= 0.8))
            sub_unique = raw_sub[keep]
            if len(sub_unique) < 8:
                continue
            step = np.linalg.norm(np.diff(sub_unique, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(step)))
            path_length = float(cumulative[-1])
            if path_length < min_width * 1.08:
                continue

            targets = np.linspace(0.0, path_length, 48)
            sub = np.column_stack((
                np.interp(targets, cumulative, sub_unique[:, 0]),
                np.interp(targets, cumulative, sub_unique[:, 1]),
            )).astype(np.float32)
            x, y = sub[:, 0], sub[:, 1]
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            x_span, depth_span = x_max - x_min, y_max - y_min
            if x_span < min_width or x_span > max_width or depth_span < min_depth:
                continue

            m = len(sub)
            # B is the upper/right corner; C is the later lower/left corner.
            b_end = max(10, min(m - 7, int(m * 0.72)))
            b_idx = int(np.argmax(x[:b_end]))
            if b_idx < 4 or not (0.07 * (m - 1) <= b_idx <= 0.67 * (m - 1)):
                continue
            c_start = min(m - 5, b_idx + 2)
            c_end = max(c_start + 1, min(m - 2, int(m * 0.95)))
            if c_end <= c_start:
                continue
            c_idx = c_start + int(np.argmin(x[c_start:c_end]))
            if not (0.30 * (m - 1) <= c_idx <= 0.94 * (m - 1)):
                continue
            if c_idx - b_idx < 3 or m - 1 - c_idx < 3:
                continue

            a, b, c, d = sub[0], sub[b_idx], sub[c_idx], sub[-1]
            top_dx, _top_dy = float(b[0] - a[0]), float(b[1] - a[1])
            diag_dx, diag_dy = float(c[0] - b[0]), float(c[1] - b[1])
            bottom_dx, _bottom_dy = float(d[0] - c[0]), float(d[1] - c[1])

            # Three broad directional arms. These limits allow a rough Z,
            # rather than requiring mathematically straight horizontal lines.
            if top_dx < 0.14 * x_span:
                continue
            if diag_dx > -0.12 * x_span or diag_dy < 0.17 * depth_span:
                continue
            if bottom_dx < 0.12 * x_span:
                continue

            # The trace must begin upper-left and finish lower-right. Endpoint
            # margins are broad enough for rounded corners and imperfect aim.
            if a[0] > x_min + 0.52 * x_span or a[1] > y_min + 0.60 * depth_span:
                continue
            if d[0] < x_min + 0.50 * x_span or d[1] < y_min + 0.42 * depth_span:
                continue

            top = sub[:b_idx + 1]
            diagonal = sub[b_idx:c_idx + 1]
            bottom = sub[c_idx:]
            # Permit reversals caused by jitter, but require the overall
            # direction of each of the three strokes to remain recognisable.
            top_progress = float(np.mean(np.diff(top[:, 0]) > -0.10 * x_span / max(1, len(top))))
            diagonal_progress = float(np.mean(
                (np.diff(diagonal[:, 0]) < 0.10 * x_span / max(1, len(diagonal)))
                & (np.diff(diagonal[:, 1]) > -0.10 * depth_span / max(1, len(diagonal)))
            ))
            bottom_progress = float(np.mean(np.diff(bottom[:, 0]) > -0.10 * x_span / max(1, len(bottom))))
            if top_progress < 0.42 or diagonal_progress < 0.36 or bottom_progress < 0.42:
                continue

            diagonal_slope = abs(diag_dy) / max(1.0, abs(diag_dx))
            if not (0.10 <= diagonal_slope <= 5.5):
                continue
            if path_length < 1.10 * x_span:
                continue

            self._last_menu_gesture_center = (
                float((a[0] + b[0] + c[0] + d[0]) / 4.0),
                float((a[1] + b[1] + c[1] + d[1]) / 4.0),
            )
            return True
        return False

    def show_radial_menu(self, at=None, page="main"):
        if not self.board_canvas:
            return
        self._clear_radial_menu()
        if at is None:
            at = (self.board_canvas.winfo_width()/2, self.board_canvas.winfo_height()/2)
        cx, cy = map(float, at)
        self.menu_page = page
        self._log_runtime_event(f"Radial menu opened: page={page!r}, cursor={at!r}")
        items_by_page = {
            # Keep the first page deliberately small: larger angular spacing makes
            # laser selection much less ambiguous than an 11-item crowded wheel.
            # Put Swipe Type on the first page as well as the Smart Write submenu,
            # so it is reachable immediately after the universal Z gesture.
            "main": ["Cursor", "Drag", "Pen", "Eraser", "Shapes", "Swipe Type", "Undo", "Redo", "More"],
            "more": ["Smart Write", "Highlight", "Scroll", "Swipe", "Color", "Size +", "Size -", "Pages", "Files", "Back"],
            "smartwrite": ["Handwriting", "Swipe Type", "Recognize Text", "Back"],
            "pages": ["Previous page", "Next page", "New page", "Back"],
            "files": ["Recognize Text", "Save PNG", "Export PDF", "Clear page", "Close Board", "Back"],
            "shapes": ["Line", "Rectangle", "Ellipse", "Arrow", "Triangle", "Diamond", "Star", "Back"],
            "color": ["White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Black", "Back"],
        }
        labels = items_by_page.get(page, items_by_page["main"])
        # Wider rings give each item a separate, easier-to-hit target.
        radius = 158 if len(labels) >= 9 else (126 if len(labels) >= 7 else 100)
        cw, ch = max(1,self.board_canvas.winfo_width()), max(1,self.board_canvas.winfo_height())
        cx = max(radius+44, min(cw-radius-44, cx))
        cy = max(radius+44, min(ch-radius-44, cy))
        self.menu_center = (cx, cy)
        # Halo and center icon make this a wheel-shaped menu rather than a list.
        self.board_canvas.create_oval(cx-radius-34, cy-radius-34, cx+radius+34, cy+radius+34,
                                      fill="#0c1724", outline="#2a4357", width=2, tags=("radial",))
        self.board_canvas.create_oval(cx-35, cy-35, cx+35, cy+35,
                                      fill=ACCENT, outline="#ecfffb", width=2, tags=("radial",))
        self.board_canvas.create_text(cx, cy, text={"main":"AIR\nTOOLS","more":"MORE\nTOOLS","smartwrite":"SMART\nWRITE","shapes":"SHAPES","color":"COLOR"}.get(page,"TOOLS"),
                                      fill="#061615", font=("Segoe UI", 8, "bold"), justify="center", tags=("radial",))
        for i, label in enumerate(labels):
            angle = -math.pi/2 + 2*math.pi*i/len(labels)
            x, y = cx+radius*math.cos(angle), cy+radius*math.sin(angle)
            if len(labels) <= 6:
                rw, rh = 50, 29
            elif len(labels) >= 9:
                rw, rh = 40, 24
            else:
                rw, rh = 44, 26
            rid = self.board_canvas.create_oval(x-rw, y-rh, x+rw, y+rh, fill="#182536",
                                                outline="#5f7c96", width=2, tags=("radial",))
            tid = self.board_canvas.create_text(x, y, text=label, fill=TEXT,
                                                font=("Segoe UI", 8, "bold"), tags=("radial",))
            self.menu_ids.append((label, rid, tid, x, y))
        self.menu_hover_since = None
        self.menu_hover_label = None
        self._menu_waiting_for_release = True
        self._menu_off_since = None
        self._last_board_canvas_point = None
        self._suppress_next_laser_click = False
        self._laser_dwell_gate.reset()
        self._laser_on_since = None
        self._laser_was_active = False
        self._laser_off_since = None
        self._last_mapped_point = None
        self.status.config(
            text="Radial menu open · release the laser first. When MENU READY appears, point at an item and switch OFF to select.",
            fg=ACCENT)

    def _clear_radial_menu(self):
        if self.board_canvas:
            self.board_canvas.delete("radial")
        self.menu_ids = []
        self.menu_hover_since = None
        self.menu_hover_label = None

    def _nearest_radial_menu_item(self, x, y):
        """Resolve a radial option with a forgiving radius and ambiguity guard."""
        ranked = sorted(
            (math.hypot(float(x) - item[3], float(y) - item[4]), item)
            for item in self.menu_ids
        )
        if not ranked or ranked[0][0] > RADIAL_MENU_HIT_RADIUS:
            return None, (ranked[0][0] if ranked else float("inf"))
        # The hit areas may overlap by a few pixels, but never choose a neighbour
        # when the spot is almost exactly between two actions.
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 8.0:
            return None, ranked[0][0]
        return ranked[0][1], ranked[0][0]

    def _radial_hover(self, x, y):
        nearest, dmin = self._nearest_radial_menu_item(x, y)
        label = nearest[0] if nearest is not None else None
        now = time.time()
        if label != self.menu_hover_label:
            for item in self.menu_ids:
                try:
                    self.board_canvas.itemconfigure(
                        item[1],
                        fill="#24504d" if item[0] == label else "#182536",
                        outline=ACCENT if item[0] == label else "#5f7c96",
                    )
                except Exception:
                    pass
            self.menu_hover_label = label
            # Start the dwell clock only when entering a new target. Do not
            # reset it on every frame while the pointer remains on that item.
            self.menu_hover_since = (now, label) if label else None
        elif label and self.menu_hover_since is None:
            self.menu_hover_since = (now, label)

    def _screen_point_to_board_canvas(self, point):
        """Convert desktop pixels to the board canvas local coordinate system."""
        if not self.board_canvas or point is None:
            return None
        try:
            point_tk = self._desktop_point_to_tk(point)
            x = float(point_tk[0]) - float(self.board_canvas.winfo_rootx())
            y = float(point_tk[1]) - float(self.board_canvas.winfo_rooty())
            cw = float(max(1, self.board_canvas.winfo_width()))
            ch = float(max(1, self.board_canvas.winfo_height()))
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < cw and 0 <= y < ch):
                return None
            return x, y
        except Exception:
            return None

    def _select_radial_item_at_screen_point(self, point):
        """Use the exact click coordinate and update the current hover target first."""
        local = self._screen_point_to_board_canvas(point)
        if local is None:
            self.status.config(text="Aim within the Smart Board menu area, then switch the laser OFF.", fg="#ffbd69")
            return False
        self._last_board_canvas_point = local
        # This is idempotent when the pointer stayed on the same item. If the final
        # click point is a different item, its dwell timer correctly starts anew.
        self._radial_hover(*local)
        return self._select_radial_item_at_canvas_point(local)

    def _select_radial_item_at_canvas_point(self, point):
        """Select a radial action using the same local coordinates used to draw it."""
        if not self.menu_ids or not self.board_canvas or point is None:
            return False
        x, y = map(float, point)
        nearest, dmin = self._nearest_radial_menu_item(x, y)
        # A wider hit radius helps laser aiming; the helper rejects ambiguous
        # points between adjacent options rather than firing a neighbouring action.
        if nearest is None:
            self.status.config(text="Aim at one menu option (not between two options), then switch the laser OFF.", fg="#ffbd69")
            return False
        label = nearest[0]
        # The laser OFF edge selects the menu item under the pointer.
        now = time.time()
        old_center = self.menu_center
        old_page = self.menu_page
        self._clear_radial_menu()
        self._menu_waiting_for_release = False
        self._menu_off_since = None
        self._last_menu_action_time = now
        self._log_runtime_event(f"Radial menu action requested: {label!r} on page {old_page!r}")
        try:
            # Internal commands and submenu navigation must never issue a Windows
            # mouse click. They dispatch directly to the selected action.
            self._menu_action(label)
        except Exception as exc:
            self._log_runtime_event(f"Radial menu action failed for {label!r}: {type(exc).__name__}: {exc}")
            self.status.config(text=f"{label} failed: {exc}. Menu restored; please retry.", fg="#ff7474")
            if self.board_canvas is not None and self.running_board:
                self._clear_radial_menu()
                self.show_radial_menu(at=old_center, page=old_page)
            return False
        self._log_runtime_event(f"Radial menu action completed: {label!r}")
        # For submenus, show_radial_menu has set its own waiting-for-release state.
        # For normal actions, set_mode/action has closed the menu and the state is idle.
        if label == "Color":
            self.status.config(text="Color palette opened · choose a colour; no desktop click will be sent.", fg=ACCENT)
        return True

    def _menu_action(self, label):
        if label == "Cursor": self.set_mode("CURSOR")
        elif label == "Drag": self.set_mode("DRAG")
        elif label == "Pen": self.set_mode("PENCIL")
        elif label == "Smart Write": self.show_radial_menu(at=self.menu_center, page="smartwrite")
        elif label == "Handwriting": self.set_mode("SMART_WRITE")
        elif label == "Swipe Type": self.open_swipe_keyboard()
        elif label == "Highlight": self.set_mode("HIGHLIGHTER")
        elif label == "Eraser": self.set_mode("ERASER")
        elif label == "Shapes":
            self.set_mode("SHAPES")
            self.show_radial_menu(at=self.menu_center, page="shapes")
        elif label == "Scroll": self.set_mode("SCROLL")
        elif label == "Swipe": self.set_mode("SWIPE")
        elif label == "Undo": self.undo()
        elif label == "Redo": self.redo()
        elif label == "More": self.show_radial_menu(at=self.menu_center, page="more")
        elif label == "Pages": self.show_radial_menu(at=self.menu_center, page="pages")
        elif label == "Files": self.show_radial_menu(at=self.menu_center, page="files")
        elif label == "Color":
            center = self.menu_center
            self.show_radial_menu(at=center, page="color")
            self.status.config(text="Color palette open · aim at a swatch, pause briefly, then switch OFF. This only changes pen colour; it does not click the desktop.", fg=ACCENT)
        elif label == "Recognize Text": self.recognize_board_text()
        elif label in ("Size +", "Size -"):
            self.pen_size = max(1, min(30, self.pen_size + (2 if label == "Size +" else -2)))
            self.set_mode("PENCIL")
        elif label == "Previous page": self._go_to_page(self.page_index-1)
        elif label == "Next page": self._go_to_page(self.page_index+1, create=True)
        elif label == "New page": self.new_page()
        elif label == "Save PNG": self.save_board()
        elif label == "Export PDF": self.export_pdf()       
        elif label == "Clear page":
            # Snapshot before clearing so the menu's Undo item can restore the page.
            self._last_clear_snapshot = (self.page_index, list(self.strokes))
            self.strokes.clear()
            self.redo_strokes.clear()
            self._swipe_type_live_text_object = None
            self.current_stroke = None
            self.last_board_point = None
            self.shape_start = self.shape_end = None
            if self.board_canvas is not None:
                self.board_canvas.delete("active-stroke-preview")
                self.board_canvas.delete("shape-preview")
                self._redraw_board()
            self._log_runtime_event(f"Cleared page index {self.page_index}; undo snapshot retained")
            self.status.config(text="Current page cleared · select Undo to restore it.", fg=ACCENT)
        elif label == "Close Board":
            self.close_board()
        elif label in ("Line", "Rectangle", "Ellipse", "Arrow", "Triangle", "Diamond", "Star"):
            self.shape_type = label
            self.set_mode("SHAPES")
        elif label == "Back":
            parent = {"more":"main", "smartwrite":"more", "shapes":"main", "color":"more", "pages":"more", "files":"more"}.get(self.menu_page, "main")
            self.show_radial_menu(at=self.menu_center, page=parent)
        elif label in {"White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Black"}:
            palette = {
                "White": "#f4f7fb", "Red": "#ff5353", "Green": "#42e695",
                "Blue": "#55a7ff", "Yellow": "#ffd45c", "Cyan": "#52e8e8",
                "Magenta": "#ef75ff", "Black": "#090b10",
            }
            self.pen_color = palette[label]
            self.set_mode("PENCIL")
            self.status.config(text=f"Pen colour set to {label} · no desktop click performed.", fg=ACCENT)
        else:
            raise ValueError(f"No action is registered for radial-menu item {label!r}")

    def open_swipe_keyboard(self):
        """Open an on-board glide-typing keyboard inside Smart Write.

        The laser is used only as a mapped pointer/path here. No OS mouse click
        is injected for the keyboard; its actions are handled by this canvas.
        """
        if not self.running_board or self.board_canvas is None:
            self.status.config(text="Open the Smart Board before using Swipe Type.", fg="#ffbd69")
            return
        self.set_mode("SWIPE_TYPE")
        self.swipe_keyboard_open = True
        self.swipe_type_active = False
        self.swipe_type_path = []
        self.swipe_type_path_length = 0.0
        self.swipe_type_start_time = 0.0
        self.swipe_type_last_seen = 0.0
        self.swipe_type_last_path_point = None
        self.swipe_type_suggestions = []
        self.swipe_type_hover_target = None
        self.swipe_type_dwell_target = None
        self.swipe_type_dwell_since = 0.0
        self.swipe_type_dwell_fired_target = None
        self.swipe_type_last_predict_time = 0.0
        self._draw_swipe_keyboard()
        # If a draft already exists, keep its board text visible while reopening.
        if self.swipe_type_text.strip():
            self._sync_swipe_type_text_object(finalize=False, redraw=False)
        self.status.config(text=f"Swipe Type ready · hold each key for {SWIPE_TYPE_KEY_DWELL_SECONDS:.1f}s to enter one character; letters combine into words. Use Space to separate words.", fg=ACCENT)
        if self.mode_label is not None:
            self.mode_label.config(text=f"SWIPE TYPE · hold each key {SWIPE_TYPE_KEY_DWELL_SECONDS:.1f}s · letters combine into words · Z opens tools")

    def _swipe_word_list(self):
        """Load the built-in word list plus optional custom words, all locally."""
        if self._swipe_type_word_cache is not None:
            return self._swipe_type_word_cache
        words = list(SWIPE_TYPE_BUILTIN_WORDS)
        custom = ROOT / "swipe_words.txt"
        try:
            if custom.is_file():
                for line in custom.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
                    value = line.strip().lower()
                    if value and value.isalpha() and 2 <= len(value) <= 24:
                        words.append(value)
        except OSError as exc:
            self._log_runtime_event(f"Could not read optional swipe_words.txt: {exc}")
        self._swipe_type_word_cache = tuple(dict.fromkeys(words))
        return self._swipe_type_word_cache

    def _draw_swipe_keyboard(self):
        """Draw a responsive, touchless QWERTY keyboard over the lower canvas."""
        canvas = self.board_canvas
        if canvas is None or not self.swipe_keyboard_open:
            return
        canvas.delete("swipe-keyboard")
        canvas.delete("swipe-type-header")
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        gap = max(3.0, min(9.0, cw * 0.004))
        key_w = max(34.0, min(82.0, (cw - 80.0) / 11.5))
        key_h = max(32.0, min(54.0, ch * 0.055))
        row_gap = max(4.0, min(8.0, key_h * 0.12))
        bottom_margin = 12.0
        top_y = ch - (4 * key_h + 3 * row_gap + bottom_margin)
        self.swipe_type_key_width = key_w
        self.swipe_keyboard_keys = []
        self.swipe_suggestion_targets = []
        self.swipe_type_hover_target = None

        # Keyboard background and instruction/status header.
        panel_x0, panel_x1 = max(8.0, (cw - (10 * key_w + 9 * gap)) / 2 - 16), min(cw - 8.0, (cw + (10 * key_w + 9 * gap)) / 2 + 16)
        panel_y0 = max(6.0, top_y - 82.0)
        canvas.create_rectangle(panel_x0, panel_y0, panel_x1, ch - 5,
                                fill="#101a28", outline="#38536b", width=2,
                                tags=("swipe-keyboard",))

        def draw_key(label, x1, y1, width, height, kind, value=None):
            x2, y2 = x1 + width, y1 + height
            base = "#26394d" if kind == "letter" else "#345366"
            rid = canvas.create_rectangle(x1, y1, x2, y2, fill=base,
                                          outline="#7b9ab1", width=1.5,
                                          tags=("swipe-keyboard",))
            tid = canvas.create_text((x1+x2)/2, (y1+y2)/2, text=label,
                                     fill="#f2f6fb",
                                     font=("Segoe UI", max(9, int(min(13, key_h * 0.26))), "bold"),
                                     tags=("swipe-keyboard",))
            self.swipe_keyboard_keys.append({
                "label": label, "value": value if value is not None else label.lower(),
                "kind": kind, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1+x2)/2, "cy": (y1+y2)/2, "rect_id": rid, "text_id": tid,
                "base_fill": base,
            })

        def draw_letter_row(letters, y, offset=None):
            width = len(letters) * key_w + max(0, len(letters)-1) * gap
            x = (cw - width) / 2 if offset is None else offset
            for ch_letter in letters:
                draw_key(ch_letter.upper(), x, y, key_w, key_h, "letter", ch_letter)
                x += key_w + gap
            return x

        y1 = top_y
        y2 = y1 + key_h + row_gap
        y3 = y2 + key_h + row_gap
        y4 = y3 + key_h + row_gap
        draw_letter_row("qwertyuiop", y1)
        draw_letter_row("asdfghjkl", y2)
        # Center the seven-letter row; Backspace is a distinct, larger key on its right.
        row3_total = 7 * key_w + 6 * gap + gap + 1.45 * key_w
        x3 = (cw - row3_total) / 2
        for ch_letter in "zxcvbnm":
            draw_key(ch_letter.upper(), x3, y3, key_w, key_h, "letter", ch_letter)
            x3 += key_w + gap
        draw_key("⌫", x3, y3, 1.45*key_w, key_h, "action", "backspace")

        # Phone-like bottom row: Space, period, Enter, Clear, Done.
        widths = [("Space", 3.8, "action", "space"), (".", 0.9, "action", "period"),
                  ("Enter", 1.25, "action", "enter"), ("Clear", 1.25, "action", "clear"),
                  ("Done", 1.25, "action", "done")]
        total = sum(units * key_w for _, units, _, _ in widths) + (len(widths)-1)*gap
        x4 = (cw - total) / 2
        for label, units, kind, value in widths:
            width = units * key_w
            draw_key(label, x4, y4, width, key_h, kind, value)
            x4 += width + gap

        self.swipe_keyboard_bounds = (panel_x0, top_y, panel_x1, ch - 5)
        self.swipe_type_key_y_top = top_y
        self.swipe_type_key_y_bottom = y4 + key_h
        self._draw_swipe_type_header()
        self._redraw_swipe_type_trace()
        canvas.tag_raise("laser-cursor")

    def _draw_swipe_type_header(self):
        canvas = self.board_canvas
        if canvas is None or not self.swipe_keyboard_open or self.swipe_keyboard_bounds is None:
            return
        canvas.delete("swipe-type-header")
        x0, _ytop, x1, _ybottom = self.swipe_keyboard_bounds
        row_top = self.swipe_type_key_y_top
        hy0, hy1 = max(5.0, row_top - 79.0), row_top - 6.0
        canvas.create_rectangle(x0+4, hy0, x1-4, hy1, fill="#18283a", outline="#4c6d84", width=1,
                                tags=("swipe-type-header",))
        typed = self.swipe_type_text[-80:] if self.swipe_type_text else ""
        display = f"Swipe Type  ·  Text: {typed}" if typed else "Swipe Type  ·  hold each key for 1 second; letters form words"
        canvas.create_text(x0+14, hy0+17, anchor="w", text=display,
                           fill="#edf5fc", font=("Segoe UI", 10, "bold"),
                           tags=("swipe-type-header",))
        suggestions = list(self.swipe_type_suggestions[:3])
        if not suggestions:
            canvas.create_text(x0+14, hy0+51, anchor="w", text="Pause on one key for 1 second to type it. Crossing keys quickly does not type them. Use Space between words; Z opens tools.",
                               fill="#a8bfd0", font=("Segoe UI", 8), tags=("swipe-type-header",))
            self.swipe_suggestion_targets = []
            return
        usable_w = max(80.0, (x1-x0-24.0)/len(suggestions))
        self.swipe_suggestion_targets = []
        for i, item in enumerate(suggestions):
            word = item[0] if isinstance(item, (tuple, list)) else str(item)
            score = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None
            bx1 = x0+12.0+i*usable_w
            bx2 = min(x1-10.0, bx1+usable_w-7.0)
            by1, by2 = hy0+32.0, hy1-5.0
            rid = canvas.create_rectangle(bx1, by1, bx2, by2, fill="#25445a", outline="#7cb8d1", width=1,
                                          tags=("swipe-type-header",))
            label = word if score is None else word
            tid = canvas.create_text((bx1+bx2)/2, (by1+by2)/2, text=label, fill="#eaf7ff",
                                     font=("Segoe UI", 9, "bold"), tags=("swipe-type-header",))
            self.swipe_suggestion_targets.append({"label": word, "kind": "suggestion", "slot": i,
                                                  "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                                                  "cx": (bx1+bx2)/2, "cy": (by1+by2)/2,
                                                  "rect_id": rid, "text_id": tid, "base_fill": "#25445a"})

    def _redraw_swipe_type_trace(self):
        canvas = self.board_canvas
        if canvas is None:
            return
        canvas.delete("swipe-type-trace")
        if len(self.swipe_type_path) >= 2:
            coords = []
            for x, y, _t in self.swipe_type_path:
                coords.extend((x, y))
            canvas.create_line(*coords, fill="#47f3ce", width=4, smooth=True,
                               capstyle="round", joinstyle="round", splinesteps=12,
                               tags=("swipe-type-trace",))
            canvas.tag_raise("swipe-type-trace")
        canvas.tag_raise("swipe-keyboard")
        canvas.tag_raise("swipe-type-header")
        canvas.tag_raise("swipe-type-trace")
        canvas.tag_raise("laser-cursor")

    def _swipe_type_target_at(self, x, y):
        for item in self.swipe_suggestion_targets:
            if item["x1"] <= x <= item["x2"] and item["y1"] <= y <= item["y2"]:
                return item
        for item in self.swipe_keyboard_keys:
            if item["x1"] <= x <= item["x2"] and item["y1"] <= y <= item["y2"]:
                return item
        return None

    def _swipe_type_set_hover(self, target):
        new_id = target.get("rect_id") if target else None
        old_id = self.swipe_type_hover_target
        if old_id == new_id:
            return
        for item in self.swipe_keyboard_keys + self.swipe_suggestion_targets:
            fill = item.get("base_fill", "#26394d")
            if item.get("rect_id") == old_id:
                try:
                    self.board_canvas.itemconfigure(old_id, fill=fill)
                except Exception:
                    pass
        if new_id is not None:
            try:
                self.board_canvas.itemconfigure(new_id, fill="#287c79")
            except Exception:
                pass
        self.swipe_type_hover_target = new_id

    def _swipe_type_target_identity(self, target):
        """Stable identity for a key even if the suggestion header is redrawn."""
        if target is None:
            return None
        kind = str(target.get("kind", ""))
        if kind == "suggestion":
            return (kind, int(target.get("slot", 0)))
        return (kind, str(target.get("value", target.get("label", ""))))

    def _reset_swipe_type_dwell(self):
        self.swipe_type_dwell_target = None
        self.swipe_type_dwell_since = 0.0
        self.swipe_type_dwell_fired_target = None

    def _update_swipe_type_point(self, x, y, now):
        """Type only after a stable per-key dwell; rapid crossings are ignored.

        This deliberately disables automatic glide-path word commitment in the
        default mode. It prevents a noisy laser path that crosses many keys from
        producing a guessed word. Each letter/action is accepted once after a
        one-second dwell, then the next key can be targeted.
        """
        target = self._swipe_type_target_at(float(x), float(y))
        target_id = self._swipe_type_target_identity(target)
        self._swipe_type_set_hover(target)

        if target is None:
            if self.swipe_type_active:
                self.swipe_type_active = False
                self.swipe_type_path = []
                self.swipe_type_path_length = 0.0
                self.swipe_type_start_time = 0.0
                self.swipe_type_last_seen = 0.0
                self.swipe_type_last_path_point = None
                if self.board_canvas is not None:
                    self.board_canvas.delete("swipe-type-trace")
            self._reset_swipe_type_dwell()
            return

        self.swipe_type_active = True
        self.swipe_type_last_seen = float(now)
        if target_id != self.swipe_type_dwell_target:
            self.swipe_type_dwell_target = target_id
            self.swipe_type_dwell_since = float(now)
            self.swipe_type_dwell_fired_target = None
            self.swipe_type_path = [(float(x), float(y), float(now))]
            self.swipe_type_path_length = 0.0
            self.swipe_type_start_time = float(now)
            self.swipe_type_last_path_point = (float(x), float(y))
            label = str(target.get("label", target.get("value", "key")))
            self.status.config(
                text=f"Swipe Type: hold {label} steadily for {SWIPE_TYPE_KEY_DWELL_SECONDS:.1f}s to enter it.",
                fg=ACCENT)
        else:
            previous = self.swipe_type_last_path_point
            if previous is not None:
                distance = math.hypot(float(x)-previous[0], float(y)-previous[1])
                if distance >= 2.0:
                    self.swipe_type_path.append((float(x), float(y), float(now)))
                    self.swipe_type_path_length += distance
                    self.swipe_type_last_path_point = (float(x), float(y))
                    if len(self.swipe_type_path) > SWIPE_TYPE_MAX_PATH_POINTS:
                        self.swipe_type_path = self.swipe_type_path[::2]
                        if self.swipe_type_path[-1] != (float(x), float(y), float(now)):
                            self.swipe_type_path.append((float(x), float(y), float(now)))

        # Draw the path for feedback only. It is not interpreted as a swipe word.
        self._redraw_swipe_type_trace()
        required_dwell = (SWIPE_TYPE_SUGGESTION_DWELL_SECONDS
                          if target.get("kind") == "suggestion"
                          else SWIPE_TYPE_KEY_DWELL_SECONDS)
        if (self.swipe_type_dwell_fired_target != target_id and
                float(now) - float(self.swipe_type_dwell_since) >= required_dwell):
            # Mark it before dispatch because dispatch redraws the header and may
            # refresh suggestion targets. This guarantees one action per dwell.
            self.swipe_type_dwell_fired_target = target_id
            self._swipe_type_tap_target(target)
            label = str(target.get("label", target.get("value", "key")))
            self._log_runtime_event(f"Swipe Type dwell accepted {label!r} after {required_dwell:.1f}s")
            self.status.config(text=f"Swipe Type: accepted {label!r}. Move to another key to continue typing.", fg=ACCENT)

    def _predict_swipe_type_path(self, path):
        centers = {}
        for item in self.swipe_keyboard_keys:
            if item["kind"] == "letter":
                centers[item["value"]] = (item["cx"], item["cy"])
        ranked = predict_swipe_words(path, centers, self.swipe_type_key_width, self._swipe_word_list(), max_results=3)
        return ranked

    def _finish_swipe_type_gesture(self, now=None):
        """End tracking after laser release; never auto-type crossed keys."""
        if not self.swipe_type_active:
            self._reset_swipe_type_dwell()
            return
        self.swipe_type_active = False
        self.swipe_type_path = []
        self.swipe_type_path_length = 0.0
        self.swipe_type_start_time = 0.0
        self.swipe_type_last_seen = 0.0
        self.swipe_type_last_path_point = None
        self._reset_swipe_type_dwell()
        if self.board_canvas is not None:
            self.board_canvas.delete("swipe-type-trace")
        self._draw_swipe_keyboard()

    def _commit_swipe_type_word(self, word):
        word = str(word).strip().lower()
        if not word:
            return
        if self.swipe_type_text and not self.swipe_type_text[-1].isspace():
            self.swipe_type_text += " "
        start = len(self.swipe_type_text)
        self.swipe_type_text += word
        self.swipe_type_last_word_span = (start, len(self.swipe_type_text))
        self.swipe_type_last_word = word
        self._sync_swipe_type_text_object(finalize=False)

    def _sync_swipe_type_text_object(self, finalize=False, redraw=True):
        """Mirror the current Swipe Type draft into a live 20-point board text object.

        The object is part of the page model as words are predicted, so text appears
        immediately and is included in exports/saves. It remains editable by the
        keyboard until Done or a mode change finalizes it.
        """
        text = str(getattr(self, "swipe_type_text", "")).strip()
        obj = getattr(self, "_swipe_type_live_text_object", None)
        obj_is_current = obj is not None and any(item is obj for item in self.strokes)
        if obj is not None and not obj_is_current:
            obj = None
            self._swipe_type_live_text_object = None

        if not text:
            if obj is not None and obj_is_current:
                self.strokes[:] = [item for item in self.strokes if item is not obj]
                self.pages[self.page_index] = self.strokes
                self._swipe_type_live_text_object = None
                self.redo_strokes.clear()
                if redraw:
                    self._redraw_board()
            return None

        if obj is None:
            text_objects = [item for item in self.strokes
                            if isinstance(item, dict) and item.get("type") == "smart_write_text"]
            next_y = max((float(item.get("y", 60.0)) + 38.0 for item in text_objects), default=60.0)
            if next_y > self.board_h - 55.0:
                next_y = 60.0
            obj = {
                "type": "smart_write_text", "text": text,
                "x": 60.0, "y": next_y,
                "width": float(max(100.0, self.board_w - 120.0)),
                "color": self.pen_color, "font_points": SWIPE_TYPE_FONT_POINTS,
                "original_strokes": [], "ocr_confidence": 100.0,
                "_swipe_type_draft": True,
            }
            self.strokes.append(obj)
            self._swipe_type_live_text_object = obj
            self.redo_strokes.clear()
        else:
            obj["text"] = text
            obj["width"] = float(max(100.0, self.board_w - 120.0))
            obj["font_points"] = SWIPE_TYPE_FONT_POINTS
            obj["color"] = self.pen_color
            obj["_swipe_type_draft"] = True

        self.pages[self.page_index] = self.strokes
        if finalize:
            obj.pop("_swipe_type_draft", None)
            self._swipe_type_live_text_object = None
        if redraw:
            self._redraw_board()
        return obj

    def _update_swipe_type_prefix_suggestions(self):
        """Offer local completions for the currently typed word prefix."""
        text = str(self.swipe_type_text)
        start = max(text.rfind(" "), text.rfind("\n")) + 1
        prefix = text[start:].strip().lower()
        if not prefix or not prefix.isalpha():
            self.swipe_type_suggestions = []
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
            return
        words = [w for w in self._swipe_word_list() if w.startswith(prefix) and w != prefix]
        # Prefer short, common completions without making the ranking expensive.
        words.sort(key=lambda w: (len(w), self._swipe_word_list().index(w)))
        self.swipe_type_suggestions = [(word, 0.0) for word in words[:3]]
        self.swipe_type_last_word_span = (start, len(text))
        self.swipe_type_last_word = prefix

    def _replace_swipe_type_last_word(self, word):
        word = str(word).strip().lower()
        span = self.swipe_type_last_word_span
        if not word or span is None:
            self.status.config(text="Swipe Type: finish a swipe word before choosing an alternative.", fg="#ffbd69")
            return
        start, end = span
        self.swipe_type_text = self.swipe_type_text[:start] + word + self.swipe_type_text[end:]
        self.swipe_type_last_word_span = (start, start + len(word))
        self.swipe_type_last_word = word
        self._update_swipe_type_prefix_suggestions()
        self._sync_swipe_type_text_object(finalize=False)
        self._draw_swipe_type_header()
        self.status.config(text=f'Swipe Type corrected the last word to "{word}".', fg=ACCENT)

    def _swipe_type_tap_target(self, target):
        kind = target.get("kind")
        value = str(target.get("value", ""))
        if kind == "suggestion":
            self._replace_swipe_type_last_word(target.get("label", ""))
            return
        if kind == "letter":
            if target.get("value"):
                self.swipe_type_text += str(target["value"])
                self._update_swipe_type_prefix_suggestions()
        elif value == "space":
            if self.swipe_type_text and not self.swipe_type_text.endswith((" ", "\n")):
                self.swipe_type_text += " "
            self.swipe_type_suggestions = []
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
        elif value == "period":
            self.swipe_type_text = self.swipe_type_text.rstrip() + ". "
            self.swipe_type_suggestions = []
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
        elif value == "backspace":
            self.swipe_type_text = self.swipe_type_text[:-1]
            self._update_swipe_type_prefix_suggestions()
        elif value == "enter":
            self.swipe_type_text = self.swipe_type_text.rstrip() + "\n"
            self.swipe_type_suggestions = []
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
        elif value == "clear":
            self.swipe_type_text = ""
            self.swipe_type_suggestions = []
            self.swipe_type_last_word_span = None
            self.swipe_type_last_word = ""
        elif value == "done":
            self._commit_swipe_type_text_to_board()
            return
        # Letter and backspace actions refresh prefix suggestions above; other
        # action keys already clear them in their individual branches.
        self._sync_swipe_type_text_object(finalize=False)
        self._draw_swipe_type_header()

    def _commit_swipe_type_text_to_board(self):
        text = str(self.swipe_type_text).strip()
        if not text:
            self.status.config(text="Swipe Type has no text to add. Swipe a word or tap the keys first.", fg="#ffbd69")
            return
        # The text was mirrored into the page after each word/key; finalize that
        # same object instead of appending a duplicate at Done.
        self._sync_swipe_type_text_object(finalize=True, redraw=False)
        self.swipe_type_text = ""
        self.swipe_type_suggestions = []
        self.swipe_type_last_word_span = None
        self.swipe_type_last_word = ""
        self.swipe_keyboard_open = False
        self._cancel_swipe_type_gesture()
        if self.board_canvas is not None:
            self.board_canvas.delete("swipe-keyboard")
            self.board_canvas.delete("swipe-type-header")
            self.board_canvas.delete("swipe-type-trace")
        self.set_mode("CURSOR")
        self._redraw_board()
        self.status.config(text="Swipe Type text is on the board in 20-point text · Undo removes it.", fg=ACCENT)

    def _cancel_swipe_type_gesture(self):
        self.swipe_type_active = False
        if hasattr(self, "swipe_type_dwell_target"):
            self._reset_swipe_type_dwell()
        self.swipe_type_path = []
        self.swipe_type_path_length = 0.0
        self.swipe_type_start_time = 0.0
        self.swipe_type_last_seen = 0.0
        self.swipe_type_last_path_point = None
        if self.board_canvas is not None:
            self.board_canvas.delete("swipe-type-trace")

    def _queue_smart_write_stroke(self, stroke, now=None):
        """Queue one completed Smart Write stroke for word-level OCR.

        The list is made from the existing board segments; this method never
        reads camera frames and never changes the laser detector.
        """
        if not stroke or not isinstance(stroke, list):
            return
        if not any(id(item) == id(stroke) for item in self._smart_write_pending_strokes):
            self._smart_write_pending_strokes.append(stroke)
        self._smart_write_last_stroke_time = float(now if now is not None else time.time())
        if self.mode_label is not None:
            try:
                self.mode_label.config(text="SMART WRITE · word captured · pause to convert")
            except Exception:
                pass

    @staticmethod
    def _smart_write_bounds(strokes):
        points = []
        widths = []
        for stroke in strokes:
            if not isinstance(stroke, (list, tuple)):
                continue
            for segment in stroke:
                try:
                    a, b, _color, width = segment
                    points.extend(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
                    widths.append(float(width))
                except (TypeError, ValueError, IndexError):
                    continue
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs), min(ys), max(xs), max(ys), max(widths or [1.0]))

    def _make_smart_write_ocr_image(self, strokes, bounds):
        """Render only the pending word as dark ink on a light background.

        A high-resolution crop is friendlier to OCR than sending the full dark
        board image with large empty regions. Original board strokes are not
        modified by this preprocessing step.
        """
        min_x, min_y, max_x, max_y, max_width = bounds
        padding = 15.0
        scale = 4
        left, top = min_x - padding, min_y - padding
        width = max(1, int(math.ceil((max_x - min_x + padding * 2) * scale)))
        height = max(1, int(math.ceil((max_y - min_y + padding * 2) * scale)))
        # Limit pathological crops without affecting ordinary handwritten words.
        scale = min(scale, max(1, int(1800 / max(width, height)))) if max(width, height) > 1800 else scale
        width = max(1, int(math.ceil((max_x - min_x + padding * 2) * scale)))
        height = max(1, int(math.ceil((max_y - min_y + padding * 2) * scale)))
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            if not isinstance(stroke, (list, tuple)):
                continue
            for segment in stroke:
                try:
                    a, b, _color, stroke_width = segment
                    pa = (round((float(a[0]) - left) * scale), round((float(a[1]) - top) * scale))
                    pb = (round((float(b[0]) - left) * scale), round((float(b[1]) - top) * scale))
                    line_width = max(3, round(max(1.0, float(stroke_width)) * scale * 0.75))
                    draw.line((pa, pb), fill=(0, 0, 0), width=line_width)
                except (TypeError, ValueError, IndexError):
                    continue
        gray = ImageOps.autocontrast(image.convert("L"))
        # Keep anti-aliased edges while ensuring the background is clean white.
        arr = np.asarray(gray, dtype=np.uint8)
        binary = np.where(arr < 205, 0, 255).astype(np.uint8)
        return Image.fromarray(binary, mode="L")

    def _recognize_smart_write_word(self, image):
        """Return (word, confidence) using Tesseract's single-word OCR modes."""
        import pytesseract
        # On Windows, Tesseract may be installed but not added to PATH. Find
        # common installation locations without touching camera/detector code.
        if shutil.which("tesseract") is None:
            candidates = (
                Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
                Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
                ROOT / "tools" / "tesseract.exe",
            )
            executable = next((candidate for candidate in candidates if candidate.is_file()), None)
            if executable is None:
                raise RuntimeError(
                    "Tesseract OCR executable was not found. Install Tesseract OCR for Windows, "
                    "then add it to PATH or install it in C:\\Program Files\\Tesseract-OCR."
                )
            pytesseract.pytesseract.tesseract_cmd = str(executable)
        candidates = []
        for psm in (8, 7):
            try:
                data = pytesseract.image_to_data(
                    image,
                    output_type=pytesseract.Output.DICT,
                    config=f"--oem 3 --psm {psm}",
                )
            except Exception:
                continue
            tokens, confidences = [], []
            for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
                token = str(raw_text).strip()
                if not token:
                    continue
                try:
                    confidence = float(raw_conf)
                except (TypeError, ValueError):
                    continue
                if confidence < 0:
                    continue
                tokens.append(token)
                confidences.append(confidence)
            if not tokens or not confidences:
                continue
            # This timer groups the strokes of one word, so remove OCR-inserted
            # spaces between fragments but preserve punctuation where present.
            word = "".join("".join(tokens).split()).strip()
            # OCR commonly adds stray quotes or marks around short handwriting.
            # Strip only punctuation at the edges; keep punctuation inside words.
            word = re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)
            if not word or not any(character.isalnum() for character in word):
                continue
            confidence = sum(confidences) / len(confidences)
            candidates.append((word, confidence))
        if not candidates:
            return "", 0.0
        return max(candidates, key=lambda result: result[1])

    def _process_smart_write_pending(self, force=False):
        """Convert a paused batch of pen strokes into 15-point text when confident.

        Failed or ambiguous recognition leaves all ink intact. Successful
        conversion stores the source strokes on the text object so Undo restores
        the sketch and Redo reapplies the text conversion.
        """
        if self._smart_write_processing or not self._smart_write_pending_strokes:
            return False
        if self.current_stroke is not None:
            return False
        now = time.time()
        if not force and now - self._smart_write_last_stroke_time < SMART_WRITE_PAUSE_SECONDS:
            return False
        self._smart_write_processing = True
        pending = list(self._smart_write_pending_strokes)
        self._smart_write_pending_strokes = []
        self._smart_write_last_stroke_time = 0.0
        try:
            # A stroke may have been undone while waiting for the pause timer.
            present_ids = {id(stroke) for stroke in self.strokes}
            pending = [stroke for stroke in pending if id(stroke) in present_ids]
            if not pending:
                return False
            bounds = self._smart_write_bounds(pending)
            if bounds is None:
                self.status.config(text="Smart Write: no usable stroke data; original marks were kept.", fg="#ffbd69")
                return False
            min_x, min_y, max_x, max_y, _max_width = bounds
            if (max_x - min_x) < 8 or (max_y - min_y) < 5:
                self.status.config(text="Smart Write: word is too small to recognize reliably; original ink was kept.", fg="#ffbd69")
                return False
            try:
                image = self._make_smart_write_ocr_image(pending, bounds)
                word, confidence = self._recognize_smart_write_word(image)
            except ImportError:
                self.status.config(
                    text="Smart Write needs pytesseract and the Windows Tesseract OCR app. Original handwriting was kept.",
                    fg="#ffbd69",
                )
                return False
            except Exception as exc:
                self.status.config(text=f"Smart Write OCR failed; original handwriting was kept ({exc}).", fg="#ffbd69")
                return False
            if not word or confidence < SMART_WRITE_MIN_CONFIDENCE:
                self.status.config(
                    text=f"Smart Write could not confidently read this word ({confidence:.0f}%); original ink was kept. Try clearer/larger letters or use Recognize Text.",
                    fg="#ffbd69",
                )
                return False

            selected_ids = {id(stroke) for stroke in pending}
            indices = [index for index, stroke in enumerate(self.strokes) if id(stroke) in selected_ids]
            if not indices:
                return False
            insert_at = min(indices)
            originals = [self.strokes[index] for index in indices]
            # Use the first stroke's colour when it remains visible on the board.
            color = "#f4f7fb"
            for stroke in originals:
                if stroke:
                    try:
                        candidate_color = str(stroke[0][2]).lower()
                        if candidate_color not in ("#090b10", "#000000", "black"):
                            color = candidate_color
                        break
                    except (TypeError, IndexError):
                        continue
            text_obj = {
                "type": "smart_write_text",
                "text": word,
                "x": float(max(0.0, min_x)),
                "y": float(max(0.0, min_y + max(0.0, ((max_y - min_y) - 24.0) / 2.0))),
                "width": float(max(80.0, (max_x - min_x) + 40.0)),
                "color": color,
                "font_points": SMART_WRITE_FONT_POINTS,
                "original_strokes": originals,
                "ocr_confidence": float(confidence),
            }
            for index in reversed(indices):
                self.strokes.pop(index)
            self.strokes.insert(insert_at, text_obj)
            self.pages[self.page_index] = self.strokes
            self.redo_strokes.clear()
            self._redraw_board()
            self.status.config(text=f'Smart Write converted to “{word}” · OCR confidence {confidence:.0f}% · Undo restores the handwriting.', fg=ACCENT)
            if self.mode_label is not None:
                self.mode_label.config(text=f"SMART WRITE · {word} · {confidence:.0f}%")
            return True
        finally:
            self._smart_write_processing = False

    def recognize_board_text(self):
        """OCR the current whiteboard page and show editable, 15-point text.

        Tesseract plus the Python pytesseract package are optional dependencies.
        OCR runs on the whole page rather than each individual stroke so word
        order and spaces can be inferred from the page layout.
        """
        try:
            import pytesseract
            image = self._render_page_image(self.strokes).convert("RGB")
            # Tesseract's block-text mode preserves the order and spaces of a
            # handwritten sentence better than classifying each stroke alone.
            config = "--oem 3 --psm 6"
            recognized = pytesseract.image_to_string(image, config=config).strip()
            recognized = "\n".join(line.rstrip() for line in recognized.splitlines())
            if not recognized:
                messagebox.showinfo("Air Writer OCR", "No text was recognized. Write darker, larger letters with clear gaps between words, then try again.")
                return
            win = tk.Toplevel(self.root)
            win.title("Recognized handwriting")
            win.configure(bg=DARK)
            win.geometry("760x420")
            tk.Label(win, text="RECOGNIZED TEXT · 15 PT", bg=DARK, fg=ACCENT,
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
            editor = tk.Text(win, wrap="word", font=("Segoe UI", 15), bg="#f7f8fa",
                             fg="#101820", insertbackground="#101820", undo=True,
                             padx=12, pady=10, relief="flat")
            editor.pack(fill="both", expand=True, padx=14, pady=(0, 10))
            editor.insert("1.0", recognized)
            tk.Button(win, text="Copy text", command=lambda: (win.clipboard_clear(), win.clipboard_append(editor.get("1.0", "end-1c"))),
                      bg=ACCENT, fg="#071716", relief="flat", padx=14, pady=7).pack(anchor="e", padx=14, pady=(0, 12))
        except ImportError:
            messagebox.showwarning(
                "OCR is not installed",
                "Handwriting-to-text needs Tesseract OCR and the pytesseract Python package.\n\n"
                "Install pytesseract with: python -m pip install pytesseract\n"
                "Then install the Tesseract OCR Windows application and ensure tesseract.exe is on PATH."
            )
        except Exception as exc:
            messagebox.showerror("Text recognition failed", f"Could not recognize this page. Check Tesseract installation.\n\n{exc}")

    def _cycle_color(self):
        colors = ["#f4f7fb", "#ff5353", "#42e695", "#55a7ff", "#ffd45c", "#52e8e8", "#ef75ff"]
        self.pen_color = colors[(colors.index(self.pen_color)+1) % len(colors)] if self.pen_color in colors else colors[0]
        self.set_mode("PENCIL")

    def _redraw_board(self):
        if not self.board_canvas:
            return
        canvas = self.board_canvas
        canvas.delete("art")
        canvas.delete("active-stroke-preview")
        cw, ch = max(1,canvas.winfo_width()), max(1,canvas.winfo_height())
        # Subtle design grid; drawing data is preserved separately from the interface.
        canvas.delete("board-grid")
        spacing = 44
        for gx in range(int(self.pan_x) % spacing, cw, spacing):
            canvas.create_line(gx, 0, gx, ch, fill="#101a28", width=1, tags=("board-grid",))
        for gy in range(int(self.pan_y) % spacing, ch, spacing):
            canvas.create_line(0, gy, cw, gy, fill="#101a28", width=1, tags=("board-grid",))
        zoom = float(getattr(self, "_hand_zoom_scale", 1.0)) if self.hand_mode else 1.0
        sx, sy = (cw/self.board_w) * zoom, (ch/self.board_h) * zoom
        origin_x = cw * (1.0 - zoom) * 0.5 + self.pan_x
        origin_y = ch * (1.0 - zoom) * 0.5 + self.pan_y
        for stroke in self.strokes:
            if isinstance(stroke, dict) and stroke.get("type") == "smart_write_text":
                try:
                    font_points = max(8, int(stroke.get("font_points", SMART_WRITE_FONT_POINTS)))
                    canvas.create_text(
                        float(stroke.get("x", 0.0))*sx + origin_x,
                        float(stroke.get("y", 0.0))*sy + origin_y,
                        text=str(stroke.get("text", "")),
                        anchor="nw",
                        fill=stroke.get("color", "#f4f7fb"),
                        font=("Segoe UI", font_points),
                        width=max(60.0, float(stroke.get("width", self.board_w - 120.0)) * sx),
                        justify="left",
                        tags=("art",),
                    )
                except Exception:
                    pass
                continue
            if not stroke:
                continue
            if id(stroke) in self._freehand_stroke_ids and len(stroke) >= 2:
                pts = [stroke[0][0]] + [segment[1] for segment in stroke]
                coords = []
                for px, py in pts:
                    coords.extend((px*sx+origin_x, py*sy+origin_y))
                color, width = stroke[-1][2], stroke[-1][3]
                canvas.create_line(*coords, fill=color, width=max(1,width*sx),
                                   capstyle="round", joinstyle="round", smooth=True,
                                   splinesteps=16, tags=("art",))
            else:
                for a,b,color,width in stroke:
                    canvas.create_line(a[0]*sx+origin_x, a[1]*sy+origin_y,
                                       b[0]*sx+origin_x, b[1]*sy+origin_y,
                                       fill=color, width=max(1,width*sx), capstyle="round",
                                       joinstyle="round", tags=("art",))
        canvas.tag_lower("board-grid")
        if self.swipe_keyboard_open and self.board_mode == "SWIPE_TYPE":
            self._draw_swipe_keyboard()

    def close(self):
        self._cancel_pending_desktop_click()
        if getattr(self, "hand_academy_window", None) is not None:
            try:
                self.hand_academy_window.destroy()
            except Exception:
                pass
            self.hand_academy_window = None
        tracker = getattr(self, "hand_tracker", None)
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
            self.hand_tracker = None
        self.close_board()
        if self._drag_down:
            try:
                import pyautogui
                pyautogui.mouseUp()
            except Exception:
                pass
            self._drag_down = False
        self._stop_camera_reader()
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AIRWRITE desktop studio")
    parser.add_argument("--mode", choices=("laser", "hand"),
                        default=os.environ.get("AIRWRITE_MODE", "laser"))
    args, _unknown = parser.parse_known_args()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    AirWriterApp(root, mode=args.mode)
    root.mainloop()

if __name__ == "__main__":
    main()




