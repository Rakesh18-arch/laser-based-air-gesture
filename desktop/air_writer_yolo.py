"""
Laser-Based Air Writer Smart Board
==================================
YOLO laser verifier + robust candidate generator.

IMPORTANT:
- Reuses the user's existing air_writer.py and YOLO model.
- Does NOT retrain YOLO.
- The previous version was too strict and could reject the real laser
  before YOLO saw it. This version adds a relaxed fallback candidate path.
- Only ONE final point is treated as the laser.

Controls:
    Q = Clear the current lock and continue
    B = Recapture background (LASER OFF)
    ESC = Quit
"""

import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from pathlib import Path
import air_writer as aw

# ==============================================================
# CONFIG
# ==============================================================
YOLO_WEIGHT = (
    Path(__file__).resolve().parent
    / "runs" / "classify" / "runs_airwriter"
    / "laser_verifier" / "weights" / "best.pt"
)
CAMERA_INDEX = aw.CAMERA_INDEX
CROP_SIZE = 64

# The classifier can be imperfectly calibrated. Start here and tune later.
MIN_YOLO_LASER_PROB = 0.55

# OpenCV is only candidate generation; YOLO makes the final decision.
YOLO_WEIGHT_VALUE = 0.85
OPENCV_WEIGHT_VALUE = 0.15
MIN_FINAL_SCORE = 0.50

# Relaxed fallback candidate generation.
FALLBACK_DIFF_THRESHOLD = 25
FALLBACK_MIN_BRIGHTNESS = 150
FALLBACK_MIN_AREA = 2
FALLBACK_MAX_AREA = 180
FALLBACK_MIN_SATURATION = 45
FALLBACK_MAX_ASPECT_RATIO = 3.0
FALLBACK_MIN_CIRCULARITY = 0.30

# Laser should have red/orange pixels near the candidate, but the core may clip white.
MIN_NEAR_RED_PIXELS = 2
RED_RING_RADIUS = 10

# Tracking / anti-jump.
TRACK_GATE_DISTANCE = 85
SEED_TOLERANCE = 28
CONFIRM_FRAMES_TO_LOCK = 3
LOW_CONFIDENCE_SWITCH_THRESHOLD = 20
SEED_START_CONFIDENCE = 45
LOCK_DISPLAY_THRESHOLD = 60
CONFIDENCE_GAIN = 20
CONFIDENCE_DECAY = 8
LOST_TRACK_FRAMES = 12
SMOOTHING_ALPHA = 0.50

# Temporal YOLO confirmation.
YOLO_HISTORY_SIZE = 5
YOLO_GOOD_FRAMES = 3

# Only send a limited number of candidates to YOLO.
MAX_YOLO_CANDIDATES = 10

WINDOW = "Air Writer - YOLO Laser"


# ==============================================================
# LOAD YOLO
# ==============================================================
print("[YOLO] Loading model...")
if not YOLO_WEIGHT.is_file():
    raise FileNotFoundError(f"Trained YOLO weights not found: {YOLO_WEIGHT}")
model = YOLO(str(YOLO_WEIGHT))
print("[YOLO] Classes:", model.names)

names = model.names
laser_class_id = None
if isinstance(names, dict):
    for idx, name in names.items():
        if str(name).strip().lower() == "laser":
            laser_class_id = int(idx)
            break
else:
    for idx, name in enumerate(names):
        if str(name).strip().lower() == "laser":
            laser_class_id = idx
            break

if laser_class_id is None:
    raise RuntimeError(f"Class 'laser' not found. Model classes: {names}")

print("[YOLO] Laser class ID:", laser_class_id)


# ==============================================================
# IMAGE HELPERS
# ==============================================================
def extract_crop(frame, center, crop_size=CROP_SIZE):
    cx, cy = int(round(center[0])), int(round(center[1]))
    half = crop_size // 2
    h, w = frame.shape[:2]

    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + crop_size, y0 + crop_size

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)

    if pad_l or pad_t or pad_r or pad_b:
        padded = cv2.copyMakeBorder(
            frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
        crop = padded[y0:y1, x0:x1]
    else:
        crop = frame[y0:y1, x0:x1]

    if crop.shape[:2] != (crop_size, crop_size):
        crop = cv2.resize(crop, (crop_size, crop_size))
    return crop


def red_near_candidate(hsv, cx, cy, radius=12):
    """
    Count genuinely red/orange pixels around a candidate.

    This is deliberately based on hue/saturation rather than absolute
    brightness because phone auto-exposure can lower the laser brightness
    when the scene becomes brighter.
    """
    h, w = hsv.shape[:2]
    x0 = max(0, int(cx) - radius)
    x1 = min(w, int(cx) + radius + 1)
    y0 = max(0, int(cy) - radius)
    y1 = min(h, int(cy) + radius + 1)

    sub = hsv[y0:y1, x0:x1]
    H, S, V = cv2.split(sub)

    red = (
        (((H <= 15) | (H >= 170)) & (S >= 55) & (V >= 70))
        | ((H >= 15) & (H <= 35) & (S >= 75) & (V >= 90))
    )

    return int(np.count_nonzero(red))


def red_change_mask(frame, background_gray):
    """
    Build a mask around RED pixels that are also brighter/different from
    the background.

    The important change is that we do not require the laser to be
    globally bright (e.g. gray > 200). This makes detection much more
    resistant to phone auto-exposure and brighter rooms/screens.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    diff = cv2.absdiff(gray, background_gray)

    H, S, V = cv2.split(hsv)

    # Red laser hue. Orange/yellow is allowed only as a secondary halo.
    red_hue = (
        (
            ((H <= 15) | (H >= 170))
            & (S >= 50)
            & (V >= 60)
        )
        |
        (
            (H >= 15)
            & (H <= 35)
            & (S >= 75)
            & (V >= 80)
        )
    )

    # True red dominance in BGR. This suppresses white screen glare.
    b, g, r = cv2.split(frame)
    red_dominant = (
        (r.astype(np.int16) >= g.astype(np.int16) + 12)
        & (r.astype(np.int16) >= b.astype(np.int16) + 12)
    )

    red_hue = red_hue & red_dominant

    # Much lower than the old absolute-brightness requirement.
    changed = diff >= 10

    red_changed = (red_hue & changed).astype(np.uint8) * 255

    # Connect the red halo/core into one small laser candidate.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    red_changed = cv2.morphologyEx(
        red_changed, cv2.MORPH_OPEN, k1, iterations=1
    )
    red_changed = cv2.dilate(
        red_changed, k2, iterations=1
    )

    return red_changed, gray, hsv, diff


def local_peak_score(gray, cx, cy):
    """
    Relative brightness peak around the candidate.

    This uses a normalized contrast ratio rather than requiring an absolute
    brightness value, so it survives exposure changes.
    """
    h, w = gray.shape
    radius = 12

    x0 = max(0, int(cx) - radius)
    x1 = min(w, int(cx) + radius + 1)
    y0 = max(0, int(cy) - radius)
    y1 = min(h, int(cy) + radius + 1)

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner = patch[dist <= 4]
    ring = patch[(dist > 6) & (dist <= 12)]

    if inner.size < 3 or ring.size < 10:
        return 0.0

    inner_mean = float(inner.mean())
    ring_mean = float(ring.mean())

    # Normalized local contrast: 0 means no peak; larger means sharper peak.
    return max(0.0, (inner_mean - ring_mean) / (ring_mean + 15.0))


def refine_to_brightest(frame, center, radius=9):
    """
    Move the candidate center from the red halo toward the brightest nearby
    pixel. This helps when the laser core clips to white.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape

    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)

    patch = gray[y0:y1, x0:x1]

    if patch.size == 0:
        return center

    _, max_val, _, max_loc = cv2.minMaxLoc(patch)

    # Only refine when the local maximum is reasonably bright.
    if max_val >= 120:
        return (float(x0 + max_loc[0]), float(y0 + max_loc[1]))

    return center


def relaxed_candidates(frame, background_gray):
    """
    Brightness-stable fallback candidate generator.

    Old fallback:
        diff + absolute brightness + color

    New fallback:
        red/orange hue + true red dominance + background change
        + small local brightness peak

    This is designed specifically for the case where the phone camera
    changes exposure because the surrounding scene becomes brighter.
    """
    red_mask, gray, hsv, diff = red_change_mask(
        frame, background_gray
    )

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        red_mask, 8
    )

    out = []

    for label in range(1, n):
        x, y, ww, hh, area = stats[label]

        # The dilation can make the red halo larger than the old laser blob.
        if area < 3 or area > 250:
            continue

        aspect = max(ww, hh) / max(1, min(ww, hh))
        if aspect > 2.8:
            continue

        component = np.zeros_like(red_mask)
        component[labels == label] = 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
        )

        if circularity < 0.25:
            continue

        cx, cy = centroids[label]

        near_red = red_near_candidate(
            hsv, cx, cy, radius=12
        )

        # Require an actual red halo, not just a bright/white patch.
        if near_red < MIN_NEAR_RED_PIXELS:
            continue

        peak = local_peak_score(
            gray, cx, cy
        )

        # Do not demand a strong peak; just use it as a ranking signal.
        red_score = min(near_red / 40.0, 1.0)
        diff_score = min(
            float(cv2.mean(diff, mask=component)[0]) / 80.0,
            1.0,
        )

        score = (
            0.45 * red_score
            + 0.25 * diff_score
            + 0.20 * min(peak / 0.7, 1.0)
            + 0.10 * min(circularity, 1.0)
        )

        refined = refine_to_brightest(
            frame,
            (cx, cy),
            radius=9,
        )

        out.append({
            "center": refined,
            "score": float(score),
            "near_red_pixels": near_red,
            "local_peak": float(peak),
            "source": "brightness_stable_fallback",
        })

    return out, red_mask


# ==============================================================
# YOLO CLASSIFICATION
# ==============================================================
def classify(frame, candidate):
    crop = extract_crop(frame, candidate["center"])
    try:
        result = model.predict(source=crop, imgsz=CROP_SIZE, verbose=False)[0]
        if result.probs is None:
            return 0.0
        probs = result.probs.data.detach().cpu().numpy().ravel()
        if laser_class_id >= len(probs):
            return 0.0
        return float(probs[laser_class_id])
    except Exception as exc:
        print("YOLO prediction warning:", exc)
        return 0.0


# ==============================================================
# TEMPORAL HISTORY
# ==============================================================
class History:
    def __init__(self):
        self.items = deque(maxlen=YOLO_HISTORY_SIZE)

    def clear(self):
        self.items.clear()

    def add(self, center, prob):
        self.items.append((center, prob))

    def good_count(self):
        return sum(p >= MIN_YOLO_LASER_PROB for _, p in self.items)

    def stable(self):
        return len(self.items) >= YOLO_GOOD_FRAMES and self.good_count() >= YOLO_GOOD_FRAMES


# ==============================================================
# VERIFY CANDIDATES
# ==============================================================
def verify(frame, strict, fallback, tracked_position):
    candidates = strict if strict else fallback

    # Prefer strict candidates, but don't let the real laser disappear just
    # because one strict geometric test rejected it.
    if strict and fallback:
        by_pos = {tuple(round(v, 1) for v in c["center"]): c for c in strict}
        for c in fallback:
            key = tuple(round(v, 1) for v in c["center"])
            if key not in by_pos:
                candidates.append(c)

    if not candidates:
        return []

    # Prioritize continuity if a track already exists.
    if tracked_position is not None:
        tx, ty = tracked_position
        candidates.sort(
            key=lambda c: (
                np.hypot(c["center"][0] - tx, c["center"][1] - ty),
                -c["score"],
            )
        )
    else:
        candidates.sort(key=lambda c: c["score"], reverse=True)

    candidates = candidates[:MAX_YOLO_CANDIDATES]
    verified = []

    for c in candidates:
        p = classify(frame, c)
        c = dict(c)
        c["yolo_probability"] = p
        c["final_score"] = YOLO_WEIGHT_VALUE * p + OPENCV_WEIGHT_VALUE * c["score"]

        if p >= MIN_YOLO_LASER_PROB and c["final_score"] >= MIN_FINAL_SCORE:
            verified.append(c)

    verified.sort(key=lambda c: c["final_score"], reverse=True)
    return verified


# ==============================================================
# MAIN
# ==============================================================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        return

    print("=" * 70)
    print("AIR WRITER - YOLO LASER TRACKER")
    print("=" * 70)
    print("Keep laser OFF during background capture.")
    print("YOLO threshold:", MIN_YOLO_LASER_PROB)
    print("Brightness-stable red fallback: ON")
    print("Global background-change guard: warning only")
    print("Controls: Q=Clear current lock | B=Recapture background | ESC=Quit")

    background_bgr, background_gray = aw.capture_background(cap)

    tracked = None
    confidence = 0.0
    lost = 0
    pending = None
    pending_count = 0
    history = History()
    last_best = None
    fps = 0.0
    t0 = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        diff = cv2.absdiff(gray, background_gray)
        _, diff_mask = cv2.threshold(diff, aw.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        change_ratio = cv2.countNonZero(diff_mask) / float(diff_mask.size)
        stale = change_ratio > aw.STALE_CHANGE_RATIO

        # Existing detector's normal path.
        color_mask = aw.build_color_mask(hsv)
        _, bright_mask = cv2.threshold(gray, aw.MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_and(diff_mask, bright_mask)
        combined = cv2.bitwise_and(combined, color_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        strict = aw.find_candidates(combined, gray, diff, frame)
        strict.sort(key=lambda c: c["score"], reverse=True)

        # Fallback is especially important when the strict detector rejects
        # the actual laser due to circularity/local-contrast at a particular
        # camera exposure.
        fallback, fallback_mask = relaxed_candidates(frame, background_gray)

        verified = verify(frame, strict, fallback, tracked)
        best = None

        ranked = []
        if verified:
            if tracked is None:
                ranked = sorted(verified, key=lambda c: c["final_score"], reverse=True)
                rank_score = lambda c: c["final_score"]
            else:
                tx, ty = tracked
                for candidate in verified:
                    distance = float(np.hypot(candidate["center"][0] - tx,
                                              candidate["center"][1] - ty))
                    if distance <= TRACK_GATE_DISTANCE:
                        candidate["distance"] = distance
                        candidate["continuity_score"] = (
                            0.65 * candidate["final_score"]
                            + 0.35 * max(0.0, 1.0 - distance / TRACK_GATE_DISTANCE)
                        )
                        ranked.append(candidate)
                ranked.sort(key=lambda c: c["continuity_score"], reverse=True)
                rank_score = lambda c: c["continuity_score"]
                if not ranked and confidence <= LOW_CONFIDENCE_SWITCH_THRESHOLD:
                    ranked = sorted(verified, key=lambda c: c["final_score"], reverse=True)
                    rank_score = lambda c: c["final_score"]

        if ranked:
            best = ranked[0]
            if len(ranked) > 1:
                first, second = ranked[0], ranked[1]
                separation = float(np.hypot(first["center"][0] - second["center"][0],
                                            first["center"][1] - second["center"][1]))
                if separation > SEED_TOLERANCE and rank_score(first) - rank_score(second) < 0.08:
                    best = None

        last_best = best

        if best is not None:
            center = best["center"]
            prob = best["yolo_probability"]
            history.add(center, prob)

            if tracked is None:
                if pending is not None and np.hypot(center[0] - pending[0], center[1] - pending[1]) <= SEED_TOLERANCE:
                    pending_count += 1
                else:
                    pending = center
                    pending_count = 1

                if pending_count >= CONFIRM_FRAMES_TO_LOCK and history.stable():
                    tracked = center
                    confidence = SEED_START_CONFIDENCE
                    lost = 0
                    pending = None
                    pending_count = 0
            else:
                tx, ty = tracked
                nx, ny = center
                tracked = (
                    tx + SMOOTHING_ALPHA * (nx - tx),
                    ty + SMOOTHING_ALPHA * (ny - ty),
                )
                confidence = min(100.0, confidence + CONFIDENCE_GAIN * prob)
                lost = 0
                pending = None
                pending_count = 0
        else:
            # A gap or ambiguity breaks consecutive-frame confirmation.
            # Clear stale candidate evidence immediately; never let an old
            # position keep showing as a current laser detection or let it
            # regain lock from one returning frame.
            history.clear()
            pending = None
            pending_count = 0
            tracked = None
            confidence = 0.0
            lost = 0

        # ----------------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------------
        display = frame.copy()

        if tracked is not None and best is not None:
            px, py = int(tracked[0]), int(tracked[1])
            locked = confidence >= LOCK_DISPLAY_THRESHOLD
            col = (0, 255, 0) if locked else (0, 200, 255)
            cv2.circle(display, (px, py), 13, col, 2)
            cv2.circle(display, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(display, f"X:{px} Y:{py}", (px + 16, py - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(display, f"{'LASER LOCKED' if locked else 'LASER TENTATIVE'} {int(confidence)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        else:
            cv2.putText(display, "LASER NOT VERIFIED THIS FRAME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        best_prob = max((c["yolo_probability"] for c in verified), default=0.0)
        cv2.putText(display, f"OpenCV strict: {len(strict)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(display, f"Red-stable fallback: {len(fallback)}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(display, f"YOLO verified: {len(verified)}", (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(display, f"Best YOLO: {best_prob:.2f}", (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(display, f"BG change: {change_ratio*100:.1f}% (warning only)", (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        if pending_count:
            cv2.putText(display, f"Confirming: {pending_count}/{CONFIRM_FRAMES_TO_LOCK}", (10, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

        if best is not None:
            bx, by = map(int, best["center"])
            cv2.circle(display, (bx, by), 8, (255, 255, 0), 1)
            cv2.putText(display, f"YOLO {best['yolo_probability']:.2f}", (bx + 10, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            crop = extract_crop(frame, best["center"])
            cv2.imshow("YOLO Crop", crop)
        else:
            no_spot = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            cv2.putText(no_spot, "NO SPOT", (3, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1)
            cv2.imshow("YOLO Crop", no_spot)

        if stale:
            cv2.putText(display, "BACKGROUND/EXPOSURE CHANGED - B OPTIONAL", (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "Q=Clear lock | B=Recapture background | ESC=Quit", (10, display.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow(WINDOW, display)
        cv2.imshow("Diagnostic: Combined Mask", combined)
        cv2.imshow("Diagnostic: Fallback Mask", fallback_mask)
        cv2.imshow("Diagnostic: Color Mask", color_mask)
        cv2.imshow("Diagnostic: Brightness Diff", diff_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('q'):
            tracked = None
            confidence = 0.0
            lost = 0
            pending = None
            pending_count = 0
            history.clear()
        elif key == ord('b'):
            background_bgr, background_gray = aw.capture_background(cap)
            tracked = None
            confidence = 0.0
            lost = 0
            pending = None
            pending_count = 0
            history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
