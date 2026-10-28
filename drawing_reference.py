"""Laser Air Writer drawing interface.
Uses the existing air_writer detector pipeline; this module does not perform
independent red-object detection.
"""
from __future__ import annotations

import importlib
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import air_writer as aw
except Exception as exc:
    raise SystemExit(f"Could not import air_writer.py: {exc}")

WINDOW = "LASER AIR WRITER"
PREVIEW_WINDOW = "DroidCam Preview"
CAMERA_INDEX = getattr(aw, "CAMERA_INDEX", 1)
BOARD_W, BOARD_H = 1280, 720
BLACK = (0, 0, 0)

COLORS = {
    "WHITE": (255, 255, 255), "RED": (0, 0, 255), "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0), "YELLOW": (0, 255, 255), "CYAN": (255, 255, 0),
    "MAGENTA": (255, 0, 255),
}
PEN_SIZES = (2, 4, 6, 10, 15)
ERASER_SIZES = (10, 20, 30, 50)
MAX_JUMP = 90.0
MIN_CONFIDENCE = 55.0
MIN_STABLE_FRAMES = 2
SMOOTH_ALPHA = 0.55


def _first(name: str, default: Any = None) -> Any:
    return getattr(aw, name, default)


def _call_detector(frame, state: dict) -> tuple[bool, tuple[float, float] | None, float]:
    """Reuse air_writer's public detector if it exposes one.

    The included project file is a mapping test whose inline detector is not
    importable as a function. For that legacy layout, this adapter executes the
    same candidate/YOLO/tracking stages without changing their thresholds.
    """
    names = ("detect_laser", "get_laser_position", "process_frame", "detect")
    for name in names:
        fn = getattr(aw, name, None)
        if callable(fn):
            try:
                result = fn(frame)
            except TypeError:
                result = fn(frame, state)
            if isinstance(result, dict):
                p = result.get("position", result.get("center"))
                valid = bool(result.get("valid", result.get("detected", p is not None)))
                conf = float(result.get("confidence", 100.0 if valid else 0.0))
                return valid, tuple(map(float, p)) if p is not None else None, conf
            if isinstance(result, tuple) and len(result) >= 2:
                p, conf = result[0], result[1]
                return p is not None, tuple(map(float, p)) if p is not None else None, float(conf)

    # Legacy air_writer-compatible path. It is intentionally the same pipeline
    # shown in the supplied screen_mapping_test.py, not a new detector.
    if "background" not in state:
        state["background"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    diff = cv2.absdiff(gray, state["background"])
    _, dm = cv2.threshold(diff, _first("DIFF_THRESHOLD", 25), 255, cv2.THRESH_BINARY)
    _, bm = cv2.threshold(gray, _first("MIN_BRIGHTNESS", 150), 255, cv2.THRESH_BINARY)
    color_fn = getattr(aw, "build_color_mask", None)
    cand_fn = getattr(aw, "find_candidates", None)
    if not callable(color_fn) or not callable(cand_fn):
        return False, None, 0.0
    mask = cv2.bitwise_and(cv2.bitwise_and(dm, bm), color_fn(hsv))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates = cand_fn(mask, gray, diff, frame)
    if not candidates:
        return False, None, 0.0
    candidates.sort(key=lambda c: float(c.get("score", 0)), reverse=True)
    c = candidates[0]
    p = tuple(map(float, c["center"]))
    score = float(c.get("score", 0))
    return True, p, score * 100.0 if score <= 1 else score


def draw_segment(img, a, b, color, width, eraser=False):
    if eraser:
        cv2.line(img, a, b, BLACK, width, cv2.LINE_AA)
        cv2.circle(img, b, max(1, width // 2), BLACK, -1, cv2.LINE_AA)
    else:
        cv2.line(img, a, b, color, width, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open DroidCam at CAMERA_INDEX={CAMERA_INDEX}")
    board = np.zeros((BOARD_H, BOARD_W, 3), np.uint8)
    undo, redo = [], []
    current = None
    mode, color_name, size = "PEN", "WHITE", 6
    preview = True
    detector_state = {}
    stable = deque(maxlen=MIN_STABLE_FRAMES)
    filtered = None
    last = None
    last_time = time.perf_counter()
    fps = 0.0
    os.makedirs("drawings", exist_ok=True)

    def finish_stroke():
        nonlocal current, last, filtered
        if current:
            undo.append(current)
            redo.clear()
        current = None; last = None; filtered = None

    def restore():
        board[:] = 0
        for stroke in undo:
            for a, b, c, w, e in stroke:
                draw_segment(board, a, b, c, w, e)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                finish_stroke(); continue
            valid, pos, confidence = _call_detector(frame, detector_state)
            valid = bool(valid and pos is not None and confidence >= MIN_CONFIDENCE)
            if valid:
                stable.append(pos)
                if len(stable) >= MIN_STABLE_FRAMES:
                    p = np.mean(np.asarray(stable), axis=0)
                    filtered = p if filtered is None else SMOOTH_ALPHA * p + (1-SMOOTH_ALPHA) * filtered
                    x = int(np.clip(filtered[0] * BOARD_W / max(1, frame.shape[1]), 0, BOARD_W-1))
                    y = int(np.clip(filtered[1] * BOARD_H / max(1, frame.shape[0]), 0, BOARD_H-1))
                    point = (x, y)
                    if last is not None and np.hypot(point[0]-last[0], point[1]-last[1]) > MAX_JUMP:
                        finish_stroke(); point = None
                    if point is not None:
                        if mode in ("PEN", "ERASER"):
                            if current is None: current = []
                            if last is not None:
                                w = size
                                draw_segment(board, last, point, COLORS[color_name], w, mode == "ERASER")
                                current.append((last, point, COLORS[color_name], w, mode == "ERASER"))
                            last = point
            else:
                stable.clear(); finish_stroke()

            now = time.perf_counter(); fps = 0.9*fps + 0.1/(now-last_time); last_time = now
            view = board.copy()
            cv2.putText(view, f"LASER: {'DETECTED' if valid else 'NOT DETECTED'}   MODE: {mode}   COLOR: {color_name}   SIZE: {size}   FPS: {fps:.0f}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (180,180,180), 1, cv2.LINE_AA)
            cv2.putText(view, "P pen | E eraser | C clear | U undo | R redo | S save | V preview | Q exit", (18, BOARD_H-18), cv2.FONT_HERSHEY_SIMPLEX, .5, (130,130,130), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            if preview:
                small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                cv2.imshow(PREVIEW_WINDOW, small)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('p'): finish_stroke(); mode = "PEN"
            elif key == ord('e'): finish_stroke(); mode = "ERASER"
            elif key == ord('v'): preview = not preview; cv2.destroyWindow(PREVIEW_WINDOW)
            elif key == ord('u') and undo: redo.append(undo.pop()); restore()
            elif key == ord('r') and redo: undo.append(redo.pop()); restore()
            elif key == ord('c'): finish_stroke(); undo.append([]); redo.clear(); board[:] = 0
            elif key == ord('s'):
                name = Path('drawings') / f"AirWriter_Drawing_{datetime.now():%Y-%m-%d_%H-%M-%S}.png"
                cv2.imwrite(str(name), board)
            elif key in b"12345": size = PEN_SIZES[int(chr(key))-1]; finish_stroke()
            elif key in b"6789": size = ERASER_SIZES[int(chr(key))-6]; mode = "ERASER"; finish_stroke()
            elif key in b"wrgb ycm".replace(b" ", b""):
                chars = {ord('w'):'WHITE',ord('r'):'RED',ord('g'):'GREEN',ord('b'):'BLUE',ord('y'):'YELLOW',ord('c'):'CYAN',ord('m'):'MAGENTA'}
                color_name = chars[key]; mode = "PEN"; finish_stroke()
    finally:
        finish_stroke(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()