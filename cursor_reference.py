"""
AirWriter - Cursor + LEFT/RIGHT Click Control
Combines:
1) Existing air_writer.py detection/tracking pipeline
2) Exact 9-point camera calibration supplied by the user
3) Piecewise 9-point screen mapping (not one global homography)
4) Verified ON/OFF gesture state machine

Gestures:
    LEFT  = laser ON -> OFF
    RIGHT = laser ON -> OFF -> ON -> OFF

Safety:
    - Mouse clicks are generated ONLY from a verified laser lock.
    - Ambiguous/lost detection never clicks.
    - C toggles LIVE/DRY-RUN clicks.
    - ESC quits.
    - Move physical mouse to a screen corner to trigger PyAutoGUI fail-safe.

Files expected in the same project:
    air_writer.py
    runs/classify/runs_airwriter/laser_verifier/weights/best.pt
"""

import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
import air_writer as aw


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW = "AirWriter - Cursor + Click Control"

MODEL_PATH = (
    r"C:\Users\HP\Desktop\AirWriter"
    r"\runs\classify\runs_airwriter"
    r"\laser_verifier\weights\best.pt"
)

CAMERA_INDEX = aw.CAMERA_INDEX

CROP_SIZE = 64

YOLO_WEIGHT = 0.70
OPENCV_WEIGHT = 0.30
MIN_YOLO_LASER_PROB = 0.50

# Tracking from the existing cursor pipeline.
TRACK_GATE_DISTANCE = 60
SEED_TOLERANCE = 25
CONFIRM_FRAMES_TO_LOCK = 4
LOST_TRACK_FRAMES = 12

CONFIDENCE_GAIN = 22
CONFIDENCE_DECAY = 12
SEED_START_CONFIDENCE = 45

# Cursor smoothing.
SMOOTHING_ALPHA = 0.50
CURSOR_MARGIN = 2

# Click confirmation.
ON_CONFIRM_FRAMES = 4
OFF_CONFIRM_FRAMES = 4
OFF_MAX_GAP_SECONDS = 0.45

# A clean OFF must happen after a stable verified ON.
MIN_ON_FRAMES = 5
STATIONARY_FRAMES = 4
STATIONARY_MAX_PX = 14

# Right-click second cycle.
SECOND_CYCLE_TIMEOUT = 0.80
SECOND_CYCLE_GRACE = 0.35

# Prevent accidental repeated clicks.
MIN_CLICK_INTERVAL = 0.35

# Start safe.
DRY_RUN_AT_START = True


# ============================================================
# EXACT USER CALIBRATION
# Camera -> Screen
# ============================================================

CAMERA_POINTS = np.array(
    [
        [91, 101],
        [309, 77],
        [530, 77],

        [94, 226],
        [321, 251],
        [532, 238],

        [270, 390],
        [316, 399],
        [534, 398],
    ],
    dtype=np.float64,
)

# These are the user's actual target screen coordinates
# for a 1920 x 1080 reference screen.
REFERENCE_SCREEN = (1920, 1080)

REFERENCE_SCREEN_POINTS = np.array(
    [
        [80, 80],
        [960, 80],
        [1840, 80],

        [80, 540],
        [960, 540],
        [1840, 540],

        [80, 1000],
        [960, 1000],
        [1840, 1000],
    ],
    dtype=np.float64,
)


# ============================================================
# 9-POINT PIECEWISE TRIANGULAR MAPPER
# ============================================================

class NinePointMapper:
    """
    Uses the measured 3x3 calibration points directly.

    A single homography was intentionally NOT used because the
    supplied nine camera points do not fit one global projective
    transform accurately enough. The camera grid is divided into
    eight triangles and each triangle maps camera coordinates
    affinely to the corresponding screen triangle.

    At every calibration point the mapping is exact.
    """

    TRIANGLES = (
        (0, 1, 4),
        (0, 4, 3),

        (1, 2, 5),
        (1, 5, 4),

        (3, 4, 7),
        (3, 7, 6),

        (4, 5, 8),
        (4, 8, 7),
    )

    def __init__(self, screen_w, screen_h):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)

        sx = self.screen_w / REFERENCE_SCREEN[0]
        sy = self.screen_h / REFERENCE_SCREEN[1]

        self.screen_points = (
            REFERENCE_SCREEN_POINTS
            * np.array([sx, sy], dtype=np.float64)
        )

        self.triangles = []

        for ids in self.TRIANGLES:
            cp = CAMERA_POINTS[list(ids)]
            sp = self.screen_points[list(ids)]

            # 3x3 affine matrix:
            # [cx cy 1] * A = [sx sy]
            M = np.column_stack(
                [cp, np.ones(3, dtype=np.float64)]
            )

            try:
                A = np.linalg.solve(M, sp)
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    f"Invalid calibration triangle: {ids}"
                )

            self.triangles.append(
                {
                    "ids": ids,
                    "camera": cp,
                    "screen": sp,
                    "A": A,
                }
            )

    @staticmethod
    def barycentric(p, tri):
        a, b, c = tri

        v0 = b - a
        v1 = c - a
        v2 = p - a

        den = (
            v0[0] * v1[1]
            - v1[0] * v0[1]
        )

        if abs(den) < 1e-9:
            return None

        v = (
            v2[0] * v1[1]
            - v1[0] * v2[1]
        ) / den

        w = (
            v0[0] * v2[1]
            - v2[0] * v0[1]
        ) / den

        u = 1.0 - v - w

        return np.array([u, v, w], dtype=np.float64)

    def camera_to_screen(self, point):
        p = np.asarray(point, dtype=np.float64)

        best = None
        best_penalty = float("inf")

        for item in self.triangles:
            weights = self.barycentric(
                p,
                item["camera"]
            )

            if weights is None:
                continue

            # Inside triangle, including a small tolerance.
            if np.all(weights >= -1e-6):
                mapped = np.array(
                    [
                        p[0],
                        p[1],
                        1.0,
                    ],
                    dtype=np.float64
                ) @ item["A"]

                return self.clamp(
                    float(mapped[0]),
                    float(mapped[1])
                )

            # If slightly outside, remember the closest triangle.
            penalty = float(
                np.sum(np.maximum(-weights, 0.0))
            )

            if penalty < best_penalty:
                best_penalty = penalty
                best = item

        # Camera point can be just outside the calibrated grid.
        # Use the least-extrapolating triangle rather than failing.
        if best is not None:
            mapped = np.array(
                [
                    p[0],
                    p[1],
                    1.0,
                ],
                dtype=np.float64
            ) @ best["A"]

            return self.clamp(
                float(mapped[0]),
                float(mapped[1])
            )

        # Last-resort nearest calibration point.
        d = np.linalg.norm(
            CAMERA_POINTS - p.reshape(1, 2),
            axis=1
        )

        nearest = int(np.argmin(d))
        q = self.screen_points[nearest]

        return self.clamp(
            float(q[0]),
            float(q[1])
        )

    def clamp(self, x, y):
        x = int(round(x))
        y = int(round(y))

        x = max(
            CURSOR_MARGIN,
            min(self.screen_w - 1 - CURSOR_MARGIN, x)
        )

        y = max(
            CURSOR_MARGIN,
            min(self.screen_h - 1 - CURSOR_MARGIN, y)
        )

        return x, y

    def calibration_error(self):
        errors = []

        for cp, sp in zip(
            CAMERA_POINTS,
            self.screen_points
        ):
            mapped = self.camera_to_screen(cp)
            errors.append(
                float(
                    np.hypot(
                        mapped[0] - sp[0],
                        mapped[1] - sp[1],
                    )
                )
            )

        return errors


# ============================================================
# YOLO
# ============================================================

print("[YOLO] Loading model...")

model = YOLO(MODEL_PATH)

laser_class_id = None

for class_id, name in model.names.items():
    if str(name).lower() == "laser":
        laser_class_id = int(class_id)
        break

if laser_class_id is None:
    laser_class_id = 0

print("[YOLO] Laser class ID:", laser_class_id)


def extract_crop(frame, center):
    cx, cy = center

    half = CROP_SIZE // 2

    x0 = int(cx - half)
    y0 = int(cy - half)
    x1 = x0 + CROP_SIZE
    y1 = y0 + CROP_SIZE

    h, w = frame.shape[:2]

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        return None

    return frame[y0:y1, x0:x1].copy()


def classify_candidate(crop):
    if crop is None:
        return 0.0, 1.0

    result = model(
        crop,
        verbose=False
    )[0]

    if result.probs is None:
        return 0.0, 1.0

    probabilities = (
        result.probs.data
        .detach()
        .cpu()
        .numpy()
    )

    laser_prob = float(
        probabilities[laser_class_id]
    )

    return laser_prob, 1.0 - laser_prob


# ============================================================
# MOUSE
# ============================================================

class MouseController:
    def __init__(self):
        self.dry_run = DRY_RUN_AT_START
        self.last_click_time = -1e9
        self.last_position = None

        pyautogui.PAUSE = 0
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.FAILSAFE = True

    def move(self, x, y):
        if self.last_position == (x, y):
            return

        pyautogui.moveTo(
            int(x),
            int(y),
            duration=0,
            _pause=False
        )

        self.last_position = (int(x), int(y))

    def click(self, button, x, y):
        now = time.monotonic()

        if (
            now - self.last_click_time
            < MIN_CLICK_INTERVAL
        ):
            print(
                "[CLICK] Suppressed by rate limit."
            )
            return False

        self.last_click_time = now

        if self.dry_run:
            print(
                f"[DRY RUN] {button.upper()} CLICK "
                f"at ({x}, {y})"
            )
            return True

        # Re-anchor before clicking.
        pyautogui.moveTo(
            int(x),
            int(y),
            duration=0,
            _pause=False
        )

        pyautogui.click(
            button=button
        )

        print(
            f"[CLICK] {button.upper()} CLICK "
            f"at ({x}, {y})"
        )

        return True


# ============================================================
# VERIFIED LASER SIGNAL
# ============================================================

class SignalState(Enum):
    OFF = 0
    CONFIRMING = 1
    ON = 2


class SignalEvent(Enum):
    ON = 1
    OFF_CLEAN = 2
    LOST = 3


class LaserSignal:
    def __init__(self):
        self.state = SignalState.OFF

        self.pending = None
        self.pending_frames = 0

        self.position = None
        self.history = deque(maxlen=12)

        self.on_frames = 0

        self.missing_frames = 0
        self.missing_since = None

    def reset(self):
        self.__init__()

    def stationary(self):
        if len(self.history) < STATIONARY_FRAMES:
            return False

        points = list(
            self.history
        )[-STATIONARY_FRAMES:]

        last = points[-1]

        return all(
            np.hypot(
                p[0] - last[0],
                p[1] - last[1],
            )
            <= STATIONARY_MAX_PX
            for p in points
        )

    def update(self, verified, now):
        # ----------------------------
        # OFF / CONFIRMING
        # ----------------------------
        if self.state != SignalState.ON:

            if verified is None:
                self.state = SignalState.OFF
                self.pending = None
                self.pending_frames = 0
                return None

            pos = verified["center"]

            if self.pending is None:
                self.pending = pos
                self.pending_frames = 1

            else:
                d = np.hypot(
                    pos[0] - self.pending[0],
                    pos[1] - self.pending[1],
                )

                if d <= SEED_TOLERANCE:
                    self.pending_frames += 1
                else:
                    self.pending = pos
                    self.pending_frames = 1

            self.state = SignalState.CONFIRMING

            if (
                self.pending_frames
                >= ON_CONFIRM_FRAMES
            ):
                self.state = SignalState.ON
                self.position = self.pending

                self.history.clear()
                self.history.append(
                    self.position
                )

                self.on_frames = 1
                self.missing_frames = 0
                self.missing_since = None

                self.pending = None
                self.pending_frames = 0

                return SignalEvent.ON

            return None

        # ----------------------------
        # ON
        # ----------------------------
        if verified is not None:
            self.position = verified["center"]

            self.history.append(
                self.position
            )

            self.on_frames += 1

            self.missing_frames = 0
            self.missing_since = None

            return None

        # ----------------------------
        # Missing frame
        # ----------------------------
        self.missing_frames += 1

        if self.missing_since is None:
            self.missing_since = now

        elapsed = (
            now - self.missing_since
        )

        if (
            self.missing_frames
            >= OFF_CONFIRM_FRAMES
            and elapsed >= 0.10
        ):
            steady = (
                self.on_frames >= MIN_ON_FRAMES
                and self.stationary()
            )

            self.state = SignalState.OFF
            self.position = None

            self.missing_frames = 0
            self.missing_since = None

            if steady:
                return SignalEvent.OFF_CLEAN

            return SignalEvent.LOST

        if elapsed >= OFF_MAX_GAP_SECONDS:
            self.state = SignalState.OFF
            self.position = None
            self.missing_frames = 0
            self.missing_since = None

            return SignalEvent.LOST

        return None


# ============================================================
# CLICK GESTURE STATE MACHINE
# ============================================================

class GestureState(Enum):
    IDLE = 0
    FIRST_ON = 1
    WAIT_SECOND = 2
    SECOND_ON = 3


class GestureMachine:
    """
    LEFT:
        ON -> OFF

    RIGHT:
        ON -> OFF -> ON -> OFF

    The first OFF starts a short waiting window.
    If no second ON arrives, LEFT is generated.
    If a second ON arrives and then cleanly turns OFF,
    RIGHT is generated.
    """

    def __init__(self, mouse):
        self.mouse = mouse
        self.state = GestureState.IDLE

        self.first_off_position = None
        self.wait_start = None

    def reset(self):
        self.state = GestureState.IDLE
        self.first_off_position = None
        self.wait_start = None

    def event(self, event, now, screen_position):
        if event == SignalEvent.LOST:
            # Never click because of a lost/ambiguous signal.
            if self.state == GestureState.FIRST_ON:
                self.reset()

            elif self.state == GestureState.SECOND_ON:
                self.state = GestureState.WAIT_SECOND
                self.wait_start = now

            return

        if event == SignalEvent.ON:

            if self.state == GestureState.IDLE:
                self.state = GestureState.FIRST_ON

            elif self.state == GestureState.WAIT_SECOND:
                self.state = GestureState.SECOND_ON

            return

        if event == SignalEvent.OFF_CLEAN:

            if self.state == GestureState.FIRST_ON:
                self.first_off_position = screen_position
                self.wait_start = now
                self.state = GestureState.WAIT_SECOND

            elif self.state == GestureState.SECOND_ON:
                # Second complete cycle => RIGHT CLICK.
                if screen_position is not None:
                    self.mouse.click(
                        "right",
                        *screen_position
                    )

                self.reset()

    def tick(self, now, second_on_confirming):
        if self.state != GestureState.WAIT_SECOND:
            return

        if self.wait_start is None:
            self.reset()
            return

        elapsed = (
            now - self.wait_start
        )

        if elapsed < SECOND_CYCLE_TIMEOUT:
            return

        if (
            second_on_confirming
            and elapsed
            < SECOND_CYCLE_TIMEOUT
            + SECOND_CYCLE_GRACE
        ):
            return

        # No second complete cycle => LEFT CLICK.
        if self.first_off_position is not None:
            self.mouse.click(
                "left",
                *self.first_off_position
            )

        self.reset()


# ============================================================
# CANDIDATE VERIFICATION
# ============================================================

def verify_candidates(
    frame,
    candidates,
    tracked_position,
):
    if not candidates:
        return None, 0.0

    ordered = list(candidates)

    if tracked_position is not None:
        ordered.sort(
            key=lambda c: np.hypot(
                c["center"][0]
                - tracked_position[0],
                c["center"][1]
                - tracked_position[1],
            )
        )
    else:
        ordered.sort(
            key=lambda c: c.get(
                "score",
                0.0
            ),
            reverse=True
        )

    verified = []
    best_yolo = 0.0

    for candidate in ordered[:4]:

        crop = extract_crop(
            frame,
            candidate["center"]
        )

        if crop is None:
            continue

        laser_prob, not_laser_prob = (
            classify_candidate(crop)
        )

        best_yolo = max(
            best_yolo,
            laser_prob
        )

        # Stronger threshold while acquiring;
        # existing lock can use the same threshold here
        # because the ON/OFF state machine provides temporal safety.
        if laser_prob < MIN_YOLO_LASER_PROB:
            continue

        opencv_score = float(
            candidate.get("score", 0.0)
        )

        final_score = (
            YOLO_WEIGHT * laser_prob
            + OPENCV_WEIGHT * opencv_score
        )

        verified.append(
            {
                **candidate,
                "laser_prob": laser_prob,
                "not_laser_prob": not_laser_prob,
                "final_score": final_score,
            }
        )

    if not verified:
        return None, best_yolo

    # Continuity selection when already tracking.
    if tracked_position is not None:

        gated = []

        for candidate in verified:
            d = np.hypot(
                candidate["center"][0]
                - tracked_position[0],
                candidate["center"][1]
                - tracked_position[1],
            )

            if d <= TRACK_GATE_DISTANCE:
                candidate["distance"] = d
                gated.append(candidate)

        if not gated:
            return None, best_yolo

        gated.sort(
            key=lambda c:
            0.70 * c["final_score"]
            + 0.30 * (
                1.0
                - c["distance"]
                / TRACK_GATE_DISTANCE
            ),
            reverse=True
        )

        return gated[0], best_yolo

    # Acquisition: choose strongest.
    verified.sort(
        key=lambda c: c["final_score"],
        reverse=True
    )

    # Reject two almost equally good distant spots.
    if len(verified) >= 2:
        a = verified[0]
        b = verified[1]

        distance = np.hypot(
            a["center"][0]
            - b["center"][0],
            a["center"][1]
            - b["center"][1],
        )

        if (
            distance > SEED_TOLERANCE
            and (
                a["final_score"]
                - b["final_score"]
            ) < 0.08
        ):
            return None, best_yolo

    return verified[0], best_yolo


# ============================================================
# BACKGROUND
# ============================================================

def capture_background(cap):
    print()
    print("=" * 60)
    print("BACKGROUND CAPTURE")
    print("KEEP LASER OFF")
    print("=" * 60)

    time.sleep(1.5)

    frames = []

    for i in range(30):
        ok, frame = cap.read()

        if not ok:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        frames.append(gray)

        display = frame.copy()

        cv2.putText(
            display,
            f"Background {i + 1}/30",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "LASER OFF",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(
            WINDOW,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None

    if not frames:
        return None

    return np.median(
        np.asarray(frames),
        axis=0
    ).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():

    screen_w, screen_h = pyautogui.size()

    mapper = NinePointMapper(
        screen_w,
        screen_h
    )

    print()
    print(
        f"[Screen] {screen_w} x {screen_h}"
    )

    print(
        "[Mapping] Nine-point piecewise mapping loaded."
    )

    errors = mapper.calibration_error()

    print(
        "[Mapping] Calibration errors:",
        [round(e, 6) for e in errors]
    )

    if max(errors) > 1.0:
        print(
            "[Mapping] WARNING: calibration error > 1 px"
        )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():
        print(
            f"ERROR: Cannot open camera "
            f"index {CAMERA_INDEX}"
        )
        return

    try:
        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )
    except Exception:
        pass

    background = capture_background(
        cap
    )

    if background is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    # Read one frame to verify camera size.
    ok, first = cap.read()

    if not ok:
        print("ERROR: Camera read failed.")
        cap.release()
        cv2.destroyAllWindows()
        return

    frame_h, frame_w = first.shape[:2]

    print(
        f"[Camera] {frame_w} x {frame_h}"
    )

    print()
    print("=" * 60)
    print("AIRWRITER CURSOR + CLICK CONTROL")
    print("=" * 60)
    print("LEFT  : ON -> OFF")
    print("RIGHT : ON -> OFF -> ON -> OFF")
    print("B     : recapture background")
    print("C     : toggle DRY-RUN / LIVE clicks")
    print("ESC   : quit")
    print(
        "DRY-RUN at startup:",
        DRY_RUN_AT_START
    )
    print("=" * 60)

    mouse = MouseController()
    signal = LaserSignal()
    gestures = GestureMachine(mouse)

    tracked_position = None
    confidence = 0.0
    lost_frames = 0

    tentative_position = None
    tentative_frames = 0

    smooth_camera = None

    background_gray = background

    fps = 0.0
    fps_count = 0
    fps_time = time.monotonic()

    last_event = "-"
    last_click_message = ""

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        now = time.monotonic()

        fps_count += 1

        if (
            now - fps_time
            >= 1.0
        ):
            fps = (
                fps_count
                / (now - fps_time)
            )
            fps_count = 0
            fps_time = now

        # ====================================================
        # EXISTING AIR_WRITER DETECTION PIPELINE
        # ====================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        diff = cv2.absdiff(
            gray,
            background_gray
        )

        _, diff_mask = cv2.threshold(
            diff,
            aw.DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        _, bright_mask = cv2.threshold(
            gray,
            aw.MIN_BRIGHTNESS,
            255,
            cv2.THRESH_BINARY
        )

        color_mask = aw.build_color_mask(
            hsv
        )

        combined = cv2.bitwise_and(
            diff_mask,
            bright_mask
        )

        combined = cv2.bitwise_and(
            combined,
            color_mask
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        # Correct OpenCV morphology call.
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        candidates = aw.find_candidates(
            combined,
            gray,
            diff,
            frame
        )

        # ====================================================
        # VERIFIED LASER
        # ====================================================

        verified, best_yolo = (
            verify_candidates(
                frame,
                candidates,
                tracked_position
            )
        )

        chosen = None

        # ====================================================
        # TRACKING
        # ====================================================

        if tracked_position is None:

            if verified is not None:

                pos = verified["center"]

                if tentative_position is None:
                    tentative_position = pos
                    tentative_frames = 1

                else:
                    distance = np.hypot(
                        pos[0]
                        - tentative_position[0],
                        pos[1]
                        - tentative_position[1],
                    )

                    if distance <= SEED_TOLERANCE:
                        tentative_frames += 1
                    else:
                        tentative_position = pos
                        tentative_frames = 1

                if (
                    tentative_frames
                    >= CONFIRM_FRAMES_TO_LOCK
                ):
                    tracked_position = (
                        tentative_position
                    )

                    confidence = (
                        SEED_START_CONFIDENCE
                    )

                    tentative_position = None
                    tentative_frames = 0
                    lost_frames = 0

                    chosen = verified

            else:
                tentative_position = None
                tentative_frames = 0

        else:

            if verified is not None:

                chosen = verified

                tracked_position = (
                    verified["center"]
                )

                confidence = min(
                    100.0,
                    confidence
                    + CONFIDENCE_GAIN
                    * verified["final_score"]
                )

                lost_frames = 0

            else:

                confidence = max(
                    0.0,
                    confidence
                    - CONFIDENCE_DECAY
                )

                lost_frames += 1

                if (
                    lost_frames
                    >= LOST_TRACK_FRAMES
                ):
                    tracked_position = None
                    confidence = 0.0
                    lost_frames = 0

        # ====================================================
        # VERIFIED SIGNAL
        # ====================================================

        signal_event = signal.update(
            chosen,
            now
        )

        # ====================================================
        # CURSOR MOVEMENT
        # ====================================================

        if (
            signal.state == SignalState.ON
            and signal.position is not None
        ):

            cx, cy = signal.position

            if smooth_camera is None:
                smooth_camera = (
                    float(cx),
                    float(cy)
                )
            else:
                smooth_camera = (
                    smooth_camera[0]
                    + SMOOTHING_ALPHA
                    * (
                        float(cx)
                        - smooth_camera[0]
                    ),

                    smooth_camera[1]
                    + SMOOTHING_ALPHA
                    * (
                        float(cy)
                        - smooth_camera[1]
                    ),
                )

            screen_x, screen_y = (
                mapper.camera_to_screen(
                    smooth_camera
                )
            )

            mouse.move(
                screen_x,
                screen_y
            )

        else:
            smooth_camera = None

        # ====================================================
        # CLICK GESTURE
        # ====================================================

        if signal_event is not None:

            last_event = signal_event.name

            screen_position = None

            if (
                signal_event
                == SignalEvent.OFF_CLEAN
            ):
                if signal.history:
                    recent = list(
                        signal.history
                    )[-3:]

                    median_camera = (
                        float(
                            np.median(
                                [
                                    p[0]
                                    for p in recent
                                ]
                            )
                        ),
                        float(
                            np.median(
                                [
                                    p[1]
                                    for p in recent
                                ]
                            )
                        ),
                    )

                    screen_position = (
                        mapper.camera_to_screen(
                            median_camera
                        )
                    )

            gestures.event(
                signal_event,
                now,
                screen_position
            )

            if signal_event != SignalEvent.ON:
                smooth_camera = None

        gestures.tick(
            now,
            signal.state
            == SignalState.CONFIRMING
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        display = frame.copy()

        if chosen is not None:
            x, y = map(
                int,
                chosen["center"]
            )

            cv2.circle(
                display,
                (x, y),
                10,
                (0, 255, 0),
                2
            )

            cv2.circle(
                display,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

        if signal.state == SignalState.ON:
            laser_text = "LASER ON"
            laser_color = (0, 255, 0)

        elif (
            signal.state
            == SignalState.CONFIRMING
        ):
            laser_text = "CONFIRMING"
            laser_color = (0, 255, 255)

        else:
            laser_text = "LASER OFF"
            laser_color = (0, 0, 255)

        cv2.putText(
            display,
            laser_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            laser_color,
            2
        )

        cv2.putText(
            display,
            f"Gesture: {gestures.state.name}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"YOLO: {best_yolo:.2f}  "
            f"Confidence: {confidence:.0f}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"Clicks: "
            f"{'DRY-RUN' if mouse.dry_run else 'LIVE'}",
            (15, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255)
            if mouse.dry_run
            else (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Last event: {last_event}",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            f"FPS: {fps:.0f}",
            (15, 166),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display,
            "LEFT=ON/OFF  RIGHT=ON/OFF/ON/OFF",
            (15, 192),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            WINDOW,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:
            break

        # C = live/dry run
        elif key == ord("c"):
            mouse.dry_run = (
                not mouse.dry_run
            )

            print(
                "[Clicks]",
                "DRY-RUN"
                if mouse.dry_run
                else "LIVE"
            )

        # B = recapture background
        elif key == ord("b"):

            new_background = (
                capture_background(cap)
            )

            if new_background is not None:

                background_gray = (
                    new_background
                )

                tracked_position = None
                confidence = 0.0
                lost_frames = 0

                tentative_position = None
                tentative_frames = 0

                smooth_camera = None

                signal.reset()
                gestures.reset()

                print(
                    "[Background] Recaptured."
                )

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("AirWriter stopped.")


if __name__ == "__main__":
    main()
