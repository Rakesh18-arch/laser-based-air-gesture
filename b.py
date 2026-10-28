"""Air Writer Smart Board — integrated desktop application.

Important design rule: the application ONLY connects to a device whose name
contains 'DroidCam'. It never silently falls back to the laptop webcam.
Existing air_writer.py candidate detection is reused; the app does not retrain YOLO.
"""
from __future__ import annotations
import os, sys, time, math, threading, traceback, re, shutil
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import air_writer as detector

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
RADIAL_MENU_DWELL_SECONDS = 0.22
RADIAL_MENU_HIT_RADIUS = 48.0
LASER_OFF_CONFIRM_SECONDS = 0.13
LASER_MIN_ON_SECONDS = 0.075
LASER_MIN_ACTIVE_SAMPLES = 3
MOUSE_CLICK_COOLDOWN_SECONDS = 0.26
# Two separate ON→OFF cycles at nearly the same location indicate a right-click.
# Singles are briefly delayed so the second cycle can disambiguate the gesture.
RIGHT_CLICK_SEQUENCE_WINDOW_SECONDS = 0.95
RIGHT_CLICK_MAX_DISTANCE_PX = 135.0
# Keep automated OS pointer movement away from exact screen corners, which
# trigger PyAutoGUI's global fail-safe and previously disabled all click input.
DESKTOP_POINTER_SAFE_MARGIN = 5
RUNTIME_LOG_FILE = ROOT / "air_writer_runtime.log"
MENU_GESTURE_WINDOW_SECONDS = 4.2
MENU_GESTURE_MIN_DURATION = 0.28
MENU_GESTURE_MAX_DURATION = 3.6
# Z-gesture recognition is deliberately structured to reject ordinary curves.
# These values describe board-canvas pixels; the existing spot detector is untouched.
MENU_GESTURE_MIN_WIDTH = 135.0
MENU_GESTURE_MIN_DEPTH = 75.0
MENU_GESTURE_TRACKING_GAP_SECONDS = 0.38
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
            for _ in range(12):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    return cap, frame, name
                time.sleep(.08)
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
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.root.configure(bg=DARK)
        self.cap = None
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
        self.camera_profile = load_camera_profile()
        self.profile_combo_labels = {}
        self.calibration_source_saved = CAL_FILE.is_file()
        self.calibration = self.load_calibration()
        self.H = self.make_homography(self.calibration)
        # Keep the last/default map only as a visual calibration guide. A fresh
        # five-point fit is required before cursor control is enabled.
        self.calibration_guide_H = None if self.H is None else self.H.copy()
        self.yolo_model = None
        self.yolo_class_id = None
        self.yolo_path = self._find_yolo_weights()
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
        # The visually tested DroidCam profile is connected automatically.
        self.root.after(900, self._auto_connect_saved_camera)

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
        tk.Label(left, text="Touchless workflow: after calibration, the laser moves the cursor. In Smart Board, draw a capital Z: top line left-to-right, diagonal down-left, then bottom line left-to-right to open the radial menu. Release the laser and wait for MENU READY, then aim at an item and switch OFF to select it. Pen/Eraser/Shapes then perform real board actions while the cursor reticle stays visible. No keyboard is needed for normal use.", bg=PANEL, fg=MUTED, wraplength=285, justify="left", font=("Segoe UI", 8)).pack(anchor="w", pady=(14,0))
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
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
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

    def _auto_connect_saved_camera(self):
        """Connect to the already visually verified camera profile without a click."""
        if self.auto_connect_started or self.cap is not None:
            return
        self.auto_connect_attempts += 1
        self.camera_profile = load_camera_profile() or self.camera_profile
        if not self.camera_profile:
            self.status.config(text="●  No saved camera profile. Use Scan OpenCV cameras once to identify the phone.", fg="#ffbd69")
            return
        # Fail closed: don't open a saved numeric index automatically if Windows
        # no longer enumerates a DroidCam device (otherwise index 0 could be the laptop webcam).
        enumerated = enumerate_cameras()
        if not any("droidcam" in name.lower() for _, name in enumerated):
            self.status.config(text="Waiting for the Windows DroidCam Video device. A numeric camera index will not be opened without DroidCam being present.", fg="#ffbd69")
            if self.auto_connect_attempts < 8:
                self.root.after(1200, self._auto_connect_saved_camera)
            else:
                self.status.config(text="DroidCam Video is not enumerated by Windows. Start/reconnect the client, then click Scan OpenCV cameras.", fg="#ff7474")
            return
        self.refresh_devices()
        values = list(self.device_combo["values"])
        if not values:
            if self.auto_connect_attempts < 8:
                self.status.config(text="Waiting for the DroidCam Windows virtual camera…", fg="#ffbd69")
                self.root.after(1200, self._auto_connect_saved_camera)
            else:
                self.status.config(text="DroidCam was not found. Start the Windows client, then click Connect.", fg="#ff7474")
            return
        wanted = int(self.camera_profile["index"])
        pick = next((i for i, value in enumerate(values)
                     if value.split(":", 1)[0].strip().isdigit()
                     and int(value.split(":", 1)[0].strip()) == wanted), 0)
        self.device_combo.current(pick)
        self.auto_connect_started = True
        self.connect_camera()

    def connect_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
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
            # Automatic start must not wait for a modal dialog/keyboard action.
            if self.camera_profile is not None and self.auto_connect_attempts < 8:
                self.auto_connect_started = False
                self.root.after(1500, self._auto_connect_saved_camera)

    def confirm_phone_preview(self):
        if self.cap is None or self.frame is None or not self.cap.isOpened():
            messagebox.showwarning("No live preview", "Connect to a camera first."); return
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
        cam_im.thumbnail(state.get("camera_preview_size", (430, 320)), Image.Resampling.LANCZOS)
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
        mask_im.thumbnail(state.get("mask_preview_size", (250, 300)), Image.Resampling.NEAREST)
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
        if self.cap is None or not self.cap.isOpened() or self.background_gray is None:
            self.cal_status.config(text="Calibration waits for a connected DroidCam and automatic background capture.", fg="#ffbd69")
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
        self._wait_for_laser_off = True
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
            self.mode_label.config(text="LASER MODE: CURSOR · laser OFF = left-click")
        # Use Tk's event queue so the calibration window is fully destroyed
        # before the full-screen board is created. This avoids focus/z-order races.
        try:
            self.root.after_idle(self.open_board)
        except Exception:
            self.open_board()

    def _tick(self):
        now = time.time()
        if self.cap is not None:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None or frame.size == 0:
                    self.laser_point = None
                    self.laser_score = 0.0
                    self.laser_mask = None
                    self.last_point_time = 0.0
                    self.smoothed = None
                    self._pending_jump_point = None
                    self._pending_jump_count = 0
                    self.status.config(text="●  DroidCam stream interrupted — reconnect required", fg="#ff7474")
                    if self.calibration_state is not None:
                        state = self.calibration_state
                        state["fresh_spot"] = False
                        state["last_detection_point"] = None
                        state["last_detection_score"] = 0.0
                        if state.get("monitor_status") is not None and state["monitor_status"].winfo_exists():
                            state["monitor_status"].configure(
                                text="CAMERA FRAME LOST — calibration is paused; reconnect DroidCam before continuing.",
                                fg="#ff8585")
                else:
                    self.frame = frame
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
                                    self._pending_jump_point = None
                                    self._pending_jump_count = 0
                                    calibration_point = tuple(map(float, cand))
                                    calibration_score = float(detected_score)
                        elif now-self.last_point_time > 0.18:
                            self.laser_point = None
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
                self.preview_info.config(text=f"Camera processing error: {exc}")
        self.root.after(25, self._tick)

    def _show_preview(self, frame):
        w=max(320,self.preview.winfo_width()); h=max(240,self.preview.winfo_height())
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); im=Image.fromarray(rgb); im.thumbnail((w,h),Image.Resampling.LANCZOS)
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

        self.mode_label = tk.Label(bar, text="CURSOR MODE · Laser ON moves · laser OFF clicks",
                                   bg="#151e2b", fg=ACCENT, font=("Segoe UI", 8, "bold"))
        self.mode_label.pack(side="right", padx=(12, 2))

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
            self.board_win.focus_force()
        except Exception:
            pass
        self.status.config(text="Smart Board open · draw a capital Z: top line left-to-right, diagonal down-left, then bottom line left-to-right. Use a large Z in one continuous laser movement. After the menu appears, release and wait for MENU READY before selecting an item. Toolbar buttons are also laser-clickable.", fg=ACCENT)

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
        if self.H is not None and self.camera_confirmed:
            self.mouse_enabled = True
            try:
                self.root.deiconify()
                self.root.lift()
                if getattr(self, "open_board_btn", None) is not None:
                    self.open_board_btn.configure(text="OPEN SMART BOARD  ·  LASER CLICK")
            except Exception:
                pass
            self.status.config(text="Board closed · laser cursor active; switch laser OFF to click Open Smart Board.", fg=ACCENT)

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
                "CURSOR": "CURSOR · one ON→OFF = left-click; two nearby ON→OFF cycles = right-click",
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
        """Project the camera laser coordinate into the real desktop pixel space."""
        if self.laser_point is None or self.H is None or time.time()-self.last_point_time > 0.22:
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

    def _point_in_board_toolbar(self, desktop_point):
        """True when the mapped laser cursor is over the board's top toolbar."""
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
            return x0 <= tx <= x1 and y0 <= ty <= y1
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

    def _laser_invoke_ui_button(self, desktop_point):
        """Invoke a button directly when a laser click lands on it.

        Do not depend solely on synthetic Windows mouse events for the app's own
        buttons. This avoids missed Tk clicks on high-DPI screens and lets the
        Smart Board launch deterministically after calibration.
        """
        if desktop_point is None:
            return False
        candidates = []
        if self.running_board and self.board_toolbar is not None:
            try:
                candidates = list(self.board_toolbar.winfo_children())
            except Exception:
                candidates = []
        elif not self.running_board:
            def walk(parent):
                try:
                    children = list(parent.winfo_children())
                except Exception:
                    return
                for child in children:
                    # The calibration wizard and other toplevels must not be targets.
                    try:
                        if child.winfo_toplevel() is not self.root:
                            continue
                    except Exception:
                        continue
                    if child.winfo_class() in ("Button", "TButton"):
                        candidates.append(child)
                    walk(child)
            walk(self.root)
        # Prefer the smallest/deepest button if bounds overlap.
        candidates.sort(key=lambda w: (w.winfo_width() * w.winfo_height()))
        for widget in candidates:
            try:
                if widget.winfo_class() not in ("Button", "TButton"):
                    continue
                state = str(widget.cget("state")).lower()
                if state == "disabled":
                    continue
            except Exception:
                pass
            if not self._widget_contains_laser_point(widget, desktop_point):
                continue
            try:
                # Let the invoked callback own the status text. In particular,
                # open_board() reports its own success/guard reason and must not
                # be overwritten by a generic "click activated" message.
                widget.invoke()
                return True
            except Exception:
                # Fall back to the normal OS click when a themed/nonstandard
                # control does not expose invoke().
                return False
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
        self._last_mapped_was_toolbar = False
        self._drag_off_since = None

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
        pressed = False
        try:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0
            pyautogui.moveTo(sx, sy, _pause=False)
            pyautogui.mouseDown(button=button, _pause=False)
            pressed = True
            # A short explicit press/release is reliable for both buttons and
            # makes it possible to guarantee release even if an exception occurs.
            time.sleep(0.035)
            pyautogui.mouseUp(button=button, _pause=False)
            pressed = False
            self._log_runtime_event(f"Desktop {button}-click executed at ({sx},{sy})")
            self.status.config(
                text=("Right-click executed after two ON→OFF cycles." if button == "right"
                      else "Left-click executed." if button == "left"
                      else "Middle-click executed."),
                fg=ACCENT)
            return True
        except Exception as exc:
            if pressed:
                try:
                    pyautogui.mouseUp(button=button, _pause=False)
                except Exception:
                    pass
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

    def _resolve_pending_desktop_left_click(self):
        """Resolve a single cycle as left-click if no second cycle followed."""
        after_id = getattr(self, "_pending_desktop_click_after_id", None)
        self._pending_desktop_click_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        pending = self._pending_desktop_click
        self._pending_desktop_click = None
        if pending is None:
            return
        self._perform_desktop_click(pending["x"], pending["y"], button="left")

    def _is_right_click_followup(self, x, y, now):
        pending = getattr(self, "_pending_desktop_click", None)
        if not pending:
            return False
        elapsed = float(now) - float(pending["time"])
        distance = math.hypot(float(x) - float(pending["x"]), float(y) - float(pending["y"]))
        return 0.0 <= elapsed <= RIGHT_CLICK_SEQUENCE_WINDOW_SECONDS and distance <= RIGHT_CLICK_MAX_DISTANCE_PX

    def _register_desktop_click_cycle(self, x, y, now):
        """Queue a single left click; two nearby valid cycles become one right click."""
        pending = getattr(self, "_pending_desktop_click", None)
        if pending and self._is_right_click_followup(x, y, now):
            elapsed = float(now) - float(pending["time"])
            distance = math.hypot(float(x) - float(pending["x"]),
                                  float(y) - float(pending["y"]))
            self._cancel_pending_desktop_click()
            self._log_runtime_event(
                f"Right-click gesture confirmed: two ON->OFF cycles, elapsed={elapsed:.3f}s, distance={distance:.1f}px")
            success = self._perform_desktop_click(x, y, button="right")
            if success:
                return "right-click"
            return "right-click-failed"

        if pending:
            # Resolve the previous single click before beginning a distinct one.
            # This avoids losing a valid left click when the next cycle is late or distant.
            self._log_runtime_event("Second cycle outside right-click window/location; resolving previous left click")
            self._resolve_pending_desktop_left_click()

        self._pending_desktop_click = {
            "x": float(x), "y": float(y), "time": float(now)
        }
        try:
            self._pending_desktop_click_after_id = self.root.after(
                int(RIGHT_CLICK_SEQUENCE_WINDOW_SECONDS * 1000),
                self._resolve_pending_desktop_left_click)
        except Exception as exc:
            self._pending_desktop_click = None
            self._pending_desktop_click_after_id = None
            self._log_runtime_event(f"Could not schedule click disambiguation timer: {exc}")
            return "schedule-failed"

        self.status.config(
            text=(f"Click pending · repeat ON→OFF at nearly the same point within "
                  f"{RIGHT_CLICK_SEQUENCE_WINDOW_SECONDS:.2f}s for right-click."),
            fg=ACCENT)
        self._log_runtime_event(f"Single ON->OFF cycle queued at ({float(x):.1f},{float(y):.1f})")
        return "left-click-pending"

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
            return
        point = self._screen_point()
        now = time.time()
        active = point is not None and self.laser_point is not None and now-self.last_point_time < 0.20
        toolbar_now = bool(self.running_board and not self.menu_ids and self._point_in_board_toolbar(point))
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
                    self.status.config(text="Cursor armed · one ON→OFF = left-click; two nearby ON→OFF cycles within 0.95s = right-click.", fg=ACCENT)
            # Never convert the final calibration target into an accidental click.
            self._laser_on_since = None
            self._laser_was_active = False
            self._laser_active_samples = 0
            self._laser_off_since = None
            self._last_mapped_point = None
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
                        text=f"Radial menu ready · hold over one item for {RADIAL_MENU_DWELL_SECONDS:.2f} seconds, then switch the laser OFF to select.",
                        fg=ACCENT)
            # The menu-opening Z must never be treated as an item click.
            self._laser_on_since = None
            self._laser_was_active = False
            self._laser_active_samples = 0
            self._laser_off_since = None
            self._last_mapped_point = None
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
                self._laser_active_samples += 1
                self._last_mapped_point = point
                self._last_mapped_was_toolbar = bool(toolbar_target)
                self._laser_was_active = True
            elif self._laser_was_active:
                if self._laser_off_since is None:
                    self._laser_off_since = now
                # Require a confirmed OFF state. The ON duration is measured only
                # across frames where the existing detector actually saw the spot;
                # the tracker-expiry grace period must not inflate a short ON pulse.
                if now-self._laser_off_since >= LASER_OFF_CONFIRM_SECONDS:
                    on_start = self._laser_on_since if self._laser_on_since is not None else self.last_point_time
                    observed_on_seconds = max(0.0, float(self.last_point_time) - float(on_start))
                    enough_on_evidence = (
                        observed_on_seconds >= LASER_MIN_ON_SECONDS
                        and self._laser_active_samples >= LASER_MIN_ACTIVE_SAMPLES
                    )
                    if enough_on_evidence and self._last_mapped_point is not None:
                        x, y = int(self._last_mapped_point[0]), int(self._last_mapped_point[1])
                        normal_cooldown_ok = (now - self.last_mouse_click >= MOUSE_CLICK_COOLDOWN_SECONDS)
                        possible_right_followup = self._is_right_click_followup(x, y, now)
                        if not (normal_cooldown_ok or possible_right_followup):
                            # Too soon to count as a fresh cycle; the state is still
                            # re-armed below, but no click action is emitted.
                            enough_on_evidence = False
                    if enough_on_evidence and self._last_mapped_point is not None:
                        x, y = int(self._last_mapped_point[0]), int(self._last_mapped_point[1])
                        action_performed = False
                        if self.running_board and self.menu_ids:
                            # Menu actions are dispatched internally, never as Windows
                            # clicks. Convert the final point of this ON interval using
                            # the same screen→Tk→canvas transform used for drawing.
                            action_performed = bool(self._select_radial_item_at_screen_point((x, y)))
                        elif self._laser_invoke_ui_button((x, y)):
                            # Tk button callback was invoked directly.
                            action_performed = True
                        elif not self._desktop_mouse_suspended and (
                                toolbar_target or not (self.running_board and self.board_mode in ("PENCIL", "HIGHLIGHTER", "ERASER", "SHAPES", "SWIPE", "SCROLL"))):
                            click_result = self._register_desktop_click_cycle(x, y, now)
                            action_performed = click_result not in ("schedule-failed", "right-click-failed")
                        elif self._desktop_mouse_suspended:
                            self.status.config(
                                text="Desktop cursor safety is paused. Board drawing and radial-menu actions remain available; move the physical mouse away from a screen corner, then toggle mouse control off/on.",
                                fg="#ffbd69")
                        # Rejected menu aim attempts should not consume the normal
                        # click cooldown; the same ON→OFF still cannot repeat because
                        # its state is reset below and must be re-armed by a new ON.
                        if action_performed:
                            self.last_mouse_click = now
                            self._log_runtime_event(
                                f"ON->OFF action accepted at ({x},{y}); detail={locals().get('click_result', 'internal-action')}; board={self.running_board}; menu_open={bool(self.menu_ids)}")
                        elif self.running_board and self.menu_ids:
                            self._log_runtime_event(
                                f"ON->OFF menu selection rejected at ({x},{y}); hover={self.menu_hover_label!r}; dwell={self.menu_hover_since!r}")
                    self._laser_on_since = None
                    self._laser_was_active = False
                    self._laser_active_samples = 0
                    self._laser_off_since = None
                    self._last_mapped_was_toolbar = False
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

    def _board_tick(self):
        if not self.board_canvas or not self.board_win:
            return
        # Switching from the dashboard to the fullscreen board must not reuse
        # the same laser ON movement as a drawing stroke or Z gesture.
        if self._wait_for_laser_off:
            if self.board_canvas:
                self.board_canvas.delete("active-stroke-preview")
            if self.current_stroke is not None:
                self.board_canvas.delete("active-stroke-preview")
                self._redraw_board()
            self.last_board_point = None
            self.current_stroke = None
            self._stroke_filter_time = 0.0
            self.menu_gesture_points = []
            self._last_menu_gesture_center = None
            return
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
        # Content coordinates are offset by the board viewport pan state.
        x = max(0, min(self.board_w-1, (cx-self.pan_x)*self.board_w/cw))
        y = max(0, min(self.board_h-1, (cy-self.pan_y)*self.board_h/ch))

        # A deliberate Z-shaped gesture (top-left → top-right → bottom-left → bottom-right) is the universal escape gesture from EVERY tool mode. It is evaluated from the last 2.8 seconds of mapped cursor history. This is
        # essential for keyboard-free use: users must be able to leave Pen/Erase/
        # Shapes/Drag/Scroll/Swipe without becoming trapped in that mode.
        # In SHAPES mode ordinary geometry remains available; the Z gesture is large and
        # directional to reduce confusion with normal shape drawing. The toolbar
        # menu remains available as a visible fallback.
        if not self.menu_ids:
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
                self.menu_gesture_points = [item for item in self.menu_gesture_points if now-item[2] <= MENU_GESTURE_WINDOW_SECONDS][-180:]
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
                    self.last_menu_trigger = now
                    menu_at = self._last_menu_gesture_center or (cx, cy)
                    old_mode = self.board_mode
                    # Discard the deliberate Z trace; it is a menu gesture, not
                    # drawing/type content. Leaving Swipe Type finalizes its live
                    # sentence and closes the keyboard before drawing the menu.
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
                    self.show_radial_menu(at=menu_at, page="main")
                    self._suppress_next_laser_click = False
                    self.menu_gesture_points = []
                    self._last_menu_gesture_center = None
                    self.last_board_point = None
                    self._draw_board_laser_cursor(cx, cy)
                    self.board_canvas.tag_raise("laser-cursor")
                    self.status.config(text="Radial menu opened · release the laser, wait for MENU READY, then aim at an option and switch OFF.", fg=ACCENT)
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
        """Recognize a deliberately drawn Z with tolerance for natural movement.

        The existing mapped cursor stream is used unchanged. Candidate paths are
        re-sampled by travelled distance, so drawing one arm slowly and another
        quickly does not break the detector. Directional checks still reject
        ordinary curves and V-shaped paths.
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

        # Light median smoothing only for gesture classification; it does not
        # affect the cursor, drawing points, or the laser detector.
        smooth_all = xy_all.copy()
        if len(xy_all) >= 5:
            for i in range(1, len(xy_all) - 1):
                smooth_all[i] = np.median(xy_all[i-1:i+2], axis=0)

        try:
            canvas_w = max(1.0, float(self.board_canvas.winfo_width())) if self.board_canvas else 1280.0
            canvas_h = max(1.0, float(self.board_canvas.winfo_height())) if self.board_canvas else 720.0
        except Exception:
            canvas_w, canvas_h = 1280.0, 720.0

        # Smaller than the former threshold, but still too large for most letters.
        min_width = max(90.0, min(MENU_GESTURE_MIN_WIDTH, canvas_w * 0.16))
        min_depth = max(55.0, min(MENU_GESTURE_MIN_DEPTH, canvas_h * 0.14))
        max_width = min(1000.0, canvas_w * 0.80)
        n = len(smooth_all)
        first_start = max(0, n - 160)

        # Test only suffixes, from recent to older. A brief movement used to aim
        # the laser at the start of the Z must not become part of the Z itself.
        for start_idx in range(n - 12, first_start - 1, -1):
            raw_sub = smooth_all[start_idx:]
            times = ts_all[start_idx:]
            if len(raw_sub) < 12:
                continue
            duration = float(times[-1] - times[0])
            if duration < MENU_GESTURE_MIN_DURATION or duration > MENU_GESTURE_MAX_DURATION:
                continue

            # Collapse near-identical frames before distance-based re-sampling.
            deltas = np.linalg.norm(np.diff(raw_sub, axis=0), axis=1)
            keep = np.concatenate(([True], deltas >= 1.0))
            sub_unique = raw_sub[keep]
            if len(sub_unique) < 8:
                continue
            step = np.linalg.norm(np.diff(sub_unique, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(step)))
            path_length = float(cumulative[-1])
            if path_length < min_width * 1.55:
                continue

            # Uniform-by-distance samples avoid timing bias when arm speeds differ.
            sample_count = 40
            targets = np.linspace(0.0, path_length, sample_count)
            sub = np.column_stack((
                np.interp(targets, cumulative, sub_unique[:, 0]),
                np.interp(targets, cumulative, sub_unique[:, 1]),
            )).astype(np.float32)
            m = len(sub)
            x, y = sub[:, 0], sub[:, 1]
            x_min, x_max = float(x.min()), float(x.max())
            y_min, y_max = float(y.min()), float(y.max())
            x_span, depth_span = x_max - x_min, y_max - y_min
            if x_span < min_width or x_span > max_width or depth_span < min_depth * 0.75:
                continue

            # Standard Z: A(top-left) -> B(top-right) -> C(bottom-left) -> D(bottom-right).
            # Search the first half for B, and the middle/later portion for C.
            b_end = max(8, min(m - 7, int(m * 0.60)))
            b_idx = int(np.argmax(x[:b_end]))
            if b_idx < 4 or not (0.13 * (m - 1) <= b_idx <= 0.53 * (m - 1)):
                continue
            c_start = min(m - 5, b_idx + 2)
            c_end = max(c_start + 1, min(m - 2, int(m * 0.94)))
            if c_end <= c_start:
                continue
            c_idx = c_start + int(np.argmin(x[c_start:c_end]))
            if not (0.40 * (m - 1) <= c_idx <= 0.88 * (m - 1)):
                continue
            if c_idx - b_idx < 3 or m - 1 - c_idx < 3:
                continue

            a, b, c, d = sub[0], sub[b_idx], sub[c_idx], sub[-1]
            top_dx, top_dy = float(b[0]-a[0]), float(b[1]-a[1])
            diag_dx, diag_dy = float(c[0]-b[0]), float(c[1]-b[1])
            bottom_dx, bottom_dy = float(d[0]-c[0]), float(d[1]-c[1])

            # Let the user draw smaller / less geometrically perfect Zs.
            if top_dx < 0.43 * x_span:
                continue
            if diag_dx > -0.34 * x_span:
                continue
            if bottom_dx < 0.43 * x_span:
                continue
            if float(d[0]-a[0]) < 0.43 * x_span:
                continue
            if diag_dy < max(26.0, min_depth * 0.42):
                continue
            if c[1]-b[1] < min_depth * 0.72:
                continue

            # Endpoints should start near the upper-left and finish near the lower-right;
            # this rejects a curve that merely happens to contain a left/right bend.
            if a[0] > x_min + 0.33 * x_span or a[1] > y_min + 0.42 * depth_span:
                continue
            if d[0] < x_max - 0.33 * x_span or d[1] < y_max - 0.42 * depth_span:
                continue
            if abs(top_dy) > max(36.0, depth_span * 0.45):
                continue
            if abs(bottom_dy) > max(38.0, depth_span * 0.48):
                continue

            top_points = sub[:b_idx + 1]
            diag_points = sub[b_idx:c_idx + 1]
            bottom_points = sub[c_idx:]
            top_band = float(np.percentile(top_points[:, 1], 85) - np.percentile(top_points[:, 1], 15))
            bottom_band = float(np.percentile(bottom_points[:, 1], 85) - np.percentile(bottom_points[:, 1], 15))
            horizontal_tol = max(32.0, depth_span * 0.46)
            if top_band > horizontal_tol or bottom_band > horizontal_tol:
                continue
            if float(np.median(bottom_points[:, 1]) - np.median(top_points[:, 1])) < min_depth * 0.52:
                continue

            # Direction consistency is tolerant of jitter, but each arm must be clear.
            dx_top = np.diff(top_points[:, 0])
            dx_diag = np.diff(diag_points[:, 0])
            dy_diag = np.diff(diag_points[:, 1])
            dx_bottom = np.diff(bottom_points[:, 0])
            if min(len(dx_top), len(dx_diag), len(dx_bottom)) < 2:
                continue
            if float(np.mean(dx_top > -4.0)) < 0.60:
                continue
            if float(np.mean(dx_diag < 4.0)) < 0.58 or float(np.mean(dy_diag > -4.0)) < 0.56:
                continue
            if float(np.mean(dx_bottom > -4.0)) < 0.60:
                continue

            diag_slope = abs(diag_dy) / max(1.0, abs(diag_dx))
            if not (0.22 <= diag_slope <= 3.2):
                continue
            if path_length < 1.70 * x_span:
                continue

            self._last_menu_gesture_center = (
                float((a[0] + b[0] + c[0] + d[0]) / 4.0),
                float((b[1] + c[1]) / 2.0),
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

    def _radial_hover(self, x, y):
        nearest, dmin = None, float("inf")
        for item in self.menu_ids:
            distance = math.hypot(x - item[3], y - item[4])
            if distance < dmin:
                nearest, dmin = item, distance
        label = nearest[0] if nearest and dmin <= RADIAL_MENU_HIT_RADIUS else None
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
        nearest, dmin = None, 1e9
        for item in self.menu_ids:
            d = math.hypot(x-item[3], y-item[4])
            if d < dmin:
                nearest, dmin = item, d
        # Keep the acceptance radius smaller than the distance between adjacent
        # item centers. This avoids selecting a neighboring action when the spot
        # falls between two menu options.
        if nearest is None or dmin > RADIAL_MENU_HIT_RADIUS:
            self.status.config(text="Aim at the center of one option and switch the laser OFF to select it.", fg="#ffbd69")
            return False
        label = nearest[0]
        # Require a stable dwell on this exact item before accepting the explicit
        # ON->OFF selection. If the dwell is too short, keep the menu open so the
        # user can re-aim and try again; do not execute a neighboring command.
        hover = self.menu_hover_since
        now = time.time()
        if (self.menu_hover_label != label or not hover or hover[1] != label or
                now - float(hover[0]) < RADIAL_MENU_DWELL_SECONDS):
            self.status.config(
                text=f"Keep the laser over {label} for {RADIAL_MENU_DWELL_SECONDS:.2f}s, then switch OFF to select.",
                fg="#ffbd69")
            return False
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
        sx, sy = cw/self.board_w, ch/self.board_h
        for stroke in self.strokes:
            if isinstance(stroke, dict) and stroke.get("type") == "smart_write_text":
                try:
                    font_points = max(8, int(stroke.get("font_points", SMART_WRITE_FONT_POINTS)))
                    canvas.create_text(
                        float(stroke.get("x", 0.0))*sx + self.pan_x,
                        float(stroke.get("y", 0.0))*sy + self.pan_y,
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
                    coords.extend((px*sx+self.pan_x, py*sy+self.pan_y))
                color, width = stroke[-1][2], stroke[-1][3]
                canvas.create_line(*coords, fill=color, width=max(1,width*sx),
                                   capstyle="round", joinstyle="round", smooth=True,
                                   splinesteps=16, tags=("art",))
            else:
                for a,b,color,width in stroke:
                    canvas.create_line(a[0]*sx+self.pan_x, a[1]*sy+self.pan_y,
                                       b[0]*sx+self.pan_x, b[1]*sy+self.pan_y,
                                       fill=color, width=max(1,width*sx), capstyle="round",
                                       joinstyle="round", tags=("art",))
        canvas.tag_lower("board-grid")
        if self.swipe_keyboard_open and self.board_mode == "SWIPE_TYPE":
            self._draw_swipe_keyboard()

    def close(self):
        self._cancel_pending_desktop_click()
        self.close_board()
        if self._drag_down:
            try:
                import pyautogui
                pyautogui.mouseUp()
            except Exception:
                pass
            self._drag_down = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        self.cap = None
        self.root.destroy()


def main():
    root=tk.Tk()
    try: ttk.Style().theme_use("clam")
    except Exception: pass
    AirWriterApp(root)
    root.mainloop()

if __name__ == "__main__": main()
