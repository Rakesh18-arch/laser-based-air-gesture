"""AIRWRITE hand landmark utilities.

The module deliberately uses MediaPipe's 21 hand landmarks rather than palm
segmentation or a palm-state gate. It chooses one stable primary hand when
more than one hand is visible and uses the index fingertip as the only pointer.
Other fingertips are used only to classify gestures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


FINGER_LAYOUT = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def _xyz(landmark: Any, width: int, height: int) -> tuple[float, float, float]:
    # MediaPipe z is scaled approximately like normalized x; multiply by width.
    return float(landmark.x) * width, float(landmark.y) * height, float(getattr(landmark, "z", 0.0)) * width


def _distance(a: Any, b: Any, width: int, height: int) -> float:
    pa, pb = _xyz(a, width, height), _xyz(b, width, height)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))


def _distance_2d(a: Any, b: Any, width: int, height: int) -> float:
    """Image-plane distance for pinch detection.

    MediaPipe's z value can fluctuate when fingertips overlap or are partly
    occluded. Pinch is a visual contact gesture, so using x/y avoids letting
    noisy depth estimates incorrectly keep thumb and index apart.
    """
    dx = (float(a.x) - float(b.x)) * width
    dy = (float(a.y) - float(b.y)) * height
    return math.hypot(dx, dy)


# Ratios are measured against palm width, not camera resolution. Separate
# engage/release thresholds provide hysteresis so a held pinch stays held.
# Pinch thresholds are deliberately forgiving for laptop webcams. The ratios
# scale with the hand instead of depending on a fixed number of pixels. A
# wider release threshold keeps a stroke/click held through small landmark
# jumps, while still requiring a deliberate separation to release.
PINCH_ENGAGE_RATIO = 0.48
PINCH_RELEASE_RATIO = 0.66
ZOOM_ENGAGE_RATIO = 0.40
ZOOM_RELEASE_RATIO = 0.54


def _angle_degrees(a: Any, b: Any, c: Any, width: int, height: int) -> float:
    """Angle ABC in 3D; an almost straight finger approaches 180 degrees."""
    pa, pb, pc = _xyz(a, width, height), _xyz(b, width, height), _xyz(c, width, height)
    v1 = tuple(x - y for x, y in zip(pa, pb))
    v2 = tuple(x - y for x, y in zip(pc, pb))
    n1 = math.sqrt(sum(v * v for v in v1))
    n2 = math.sqrt(sum(v * v for v in v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosine = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def finger_extension(landmarks: Any, name: str, width: int, height: int) -> tuple[bool, float]:
    """Return orientation-independent finger extension and PIP angle.

    Unlike comparing tip.y with pip.y, joint angles work when the hand is
    tilted, rotated, or held at the top/bottom of the camera image.
    """
    mcp, pip, tip = FINGER_LAYOUT[name]
    angle = _angle_degrees(landmarks[mcp], landmarks[pip], landmarks[tip], width, height)
    # Use the local PIP joint angle only. Wrist-distance gates are fragile when
    # the hand is rotated, foreshortened, or held near a camera-frame edge.
    thresholds = {"index": 142.0, "middle": 150.0, "ring": 150.0, "pinky": 150.0}
    extended = angle >= thresholds.get(name, 145.0)
    return extended, angle


def hand_center_and_area(landmarks: Any, width: int, height: int) -> tuple[tuple[float, float], float]:
    xs = [float(p.x) * width for p in landmarks]
    ys = [float(p.y) * height for p in landmarks]
    center = ((xs[0] + xs[9]) * 0.5, (ys[0] + ys[9]) * 0.5)
    area = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    return center, area


def select_primary_hand(
    hands: Iterable[Any], width: int, height: int,
    previous_center: tuple[float, float] | None = None,
) -> tuple[Any | None, tuple[float, float] | None]:
    """Select a consistent hand, avoiding arbitrary switching between palms."""
    candidates = []
    for landmarks in hands or []:
        center, area = hand_center_and_area(landmarks.landmark, width, height)
        candidates.append((landmarks, center, area))
    if not candidates:
        return None, None
    if previous_center is None:
        chosen = max(candidates, key=lambda item: item[2])
        return chosen[0], chosen[1]
    diagonal = max(1.0, math.hypot(width, height))
    def score(item: tuple[Any, tuple[float, float], float]) -> float:
        _, center, area = item
        proximity = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) / diagonal
        size_bonus = min(0.12, math.sqrt(area) / diagonal * 0.12)
        return proximity - size_bonus
    chosen = min(candidates, key=score)
    return chosen[0], chosen[1]


@dataclass
class HandPose:
    index_extended: bool
    fingers: dict[str, bool]
    angles: dict[str, float]
    pinch_ratio: float
    thumb_middle_ratio: float
    pinch_active: bool
    zoom_active: bool
    pan_active: bool
    menu_active: bool
    pointer_active: bool
    gesture: str
    index_point: tuple[float, float]
    wrist_center_y: float


def classify_hand_pose(
    landmarks: Any,
    width: int,
    height: int,
    pinch_was_active: bool = False,
    zoom_was_active: bool = False,
) -> HandPose:
    """Classify a small, intentional gesture vocabulary with hysteresis.

    There is no palm-open/palm-closed status requirement. The index tip is the
    sole pointer; the other fingertips only contribute to gesture recognition.
    Pinch state is level-triggered: it remains active while the thumb/index tips
    stay within the release radius. The desktop controller can therefore treat
    pinch start as mouse-down/draw and pinch release as mouse-up/click.
    """
    fingers: dict[str, bool] = {}
    angles: dict[str, float] = {}
    for name in FINGER_LAYOUT:
        fingers[name], angles[name] = finger_extension(landmarks, name, width, height)

    # Use the visible palm width as the reference scale. The wrist-to-middle-
    # MCP distance is a backup for edge-on hands where palm width is foreshortened.
    palm_width = _distance_2d(landmarks[5], landmarks[17], width, height)
    wrist_to_middle = _distance_2d(landmarks[0], landmarks[9], width, height)
    # Scale the minimum with the frame diagonal so the same physical hand
    # produces similar ratios at 640x480, 1280x720 and higher resolutions.
    frame_scale = math.hypot(max(1, width), max(1, height))
    palm_scale = max(24.0, frame_scale * 0.035, palm_width, wrist_to_middle * 0.82)

    # Thumb/index contact is detected in the image plane. This is more reliable
    # than 3-D distance from a monocular webcam, especially when one fingertip
    # partly hides the other. The thresholds are intentionally forgiving for
    # ordinary webcam resolutions and different hand sizes.
    pinch_tip_distance = _distance_2d(landmarks[4], landmarks[8], width, height)
    pinch_ratio = pinch_tip_distance / palm_scale
    thumb_middle_ratio = _distance_2d(landmarks[4], landmarks[12], width, height) / palm_scale

    # Schmitt-trigger hysteresis: enter draw at a modestly separated pinch,
    # then keep drawing until the fingers separate clearly. This avoids flicker
    # during a continuous stroke and makes the gesture less dependent on an
    # exact fingertip-touch frame.
    pinch_release = PINCH_RELEASE_RATIO if pinch_was_active else PINCH_ENGAGE_RATIO
    pinch_active = bool(pinch_ratio <= pinch_release)

    # Zoom is only allowed when thumb-middle is close AND thumb-index has
    # opened decisively. The dead band prevents a slightly imperfect drawing
    # pinch from switching to zoom while the hand is moving.
    zoom_release = ZOOM_RELEASE_RATIO if zoom_was_active else ZOOM_ENGAGE_RATIO
    zoom_candidate = thumb_middle_ratio <= zoom_release
    index_is_clearly_released = pinch_ratio >= (PINCH_RELEASE_RATIO + 0.10)
    zoom_active = bool(zoom_candidate and index_is_clearly_released)
    if zoom_active:
        pinch_active = False

    # Keep the pointer/drawing vocabulary simple: V-sign pans; three raised
    # fingers opens the tool menu; thumb-index pinch draws/selects.
    pan_active = bool(fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"])
    menu_active = bool(fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"])
    index_extended = fingers["index"]

    if zoom_active:
        gesture = "ZOOM · THUMB-MIDDLE"
    elif pinch_active:
        gesture = "PINCH · DRAW / SELECT"
    elif pan_active:
        gesture = "PAN · V SIGN"
    elif menu_active:
        gesture = "THREE FINGERS · TOOLS"
    elif index_extended:
        gesture = "POINT · MOVE"
    else:
        # No palm-status gate: the hand is simply waiting for an index/pinch pose.
        gesture = "HAND DETECTED · EXTEND INDEX"

    # The index fingertip is the pointer in every state, including pinch.
    # Keeping one pointer anchor prevents the cursor jumping when the thumb
    # closes and makes toolbar clicks land where the index was aimed.
    pointer_active = bool(index_extended or pinch_active or zoom_active or pan_active or menu_active)
    index = landmarks[8]
    pointer_x = float(index.x) * width
    pointer_y = float(index.y) * height
    wrist_center_y = (float(landmarks[0].y) + float(landmarks[9].y)) * 0.5
    return HandPose(
        index_extended=index_extended,
        fingers=fingers,
        angles=angles,
        pinch_ratio=float(pinch_ratio),
        thumb_middle_ratio=float(thumb_middle_ratio),
        pinch_active=bool(pinch_active),
        zoom_active=bool(zoom_active),
        pan_active=bool(pan_active),
        menu_active=bool(menu_active),
        pointer_active=pointer_active,
        gesture=gesture,
        index_point=(pointer_x, pointer_y),
        wrist_center_y=wrist_center_y,
    )


class GestureConsensus:
    """Reject one-frame pan/menu hallucinations with a short vote window."""
    def __init__(self, window: int = 5, min_votes: int = 3):
        from collections import deque
        self.window = max(3, int(window))
        self.min_votes = max(2, min(int(min_votes), self.window))
        self._pan = deque(maxlen=self.window)
        self._menu = deque(maxlen=self.window)

    def update(self, pan_candidate: bool, menu_candidate: bool) -> tuple[bool, bool]:
        self._pan.append(bool(pan_candidate))
        self._menu.append(bool(menu_candidate))
        return (sum(self._pan) >= self.min_votes,
                sum(self._menu) >= self.min_votes)

    def reset(self) -> None:
        self._pan.clear()
        self._menu.clear()


class PinchStabilizer:
    """Debounce pinch transitions so a real held pinch never flickers per frame.

    Two consecutive detections engage a pinch; four consecutive open frames
    release it. At a 25-30 FPS webcam cadence this rejects one-frame landmark
    noise while keeping the drawing response quick. Call reset() when the hand
    track is genuinely lost or Hand mode is stopped.
    """
    def __init__(self, engage_frames: int = 2, release_frames: int = 4):
        self.engage_frames = max(1, int(engage_frames))
        self.release_frames = max(1, int(release_frames))
        self.active = False
        self._engage_count = 0
        self._release_count = 0

    def update(self, candidate: bool, force_release: bool = False) -> bool:
        if force_release:
            self.reset()
            return False
        if candidate:
            self._engage_count += 1
            self._release_count = 0
            if not self.active and self._engage_count >= self.engage_frames:
                self.active = True
        else:
            self._release_count += 1
            self._engage_count = 0
            if self.active and self._release_count >= self.release_frames:
                self.active = False
        return self.active

    def reset(self) -> None:
        self.active = False
        self._engage_count = 0
        self._release_count = 0


def map_hand_pointer_to_canvas(
    point_px: tuple[float, float],
    frame_size: tuple[int, int],
    canvas_rect: tuple[float, float, float, float],
    mirror_x: bool = True,
) -> tuple[float, float]:
    """Map a filtered fingertip into the board with a tiny edge-stability band.

    The outer 1.2% of the camera image maps smoothly to the board edge. This
    prevents one-pixel landmark changes at the camera border from making the
    pointer bounce vertically/horizontally along a screen corner.
    """
    fw, fh = max(1, int(frame_size[0])), max(1, int(frame_size[1]))
    left, top, cw, ch = map(float, canvas_rect)
    if cw <= 0 or ch <= 0:
        raise ValueError("Canvas width and height must be positive")

    def stable_axis(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        edge = 0.012
        if value <= edge:
            return 0.0
        if value >= 1.0 - edge:
            return 1.0
        return (value - edge) / (1.0 - 2.0 * edge)

    nx = stable_axis(float(point_px[0]) / fw)
    ny = stable_axis(float(point_px[1]) / fh)
    if mirror_x:
        nx = 1.0 - nx
    return left + nx * cw, top + ny * ch


class OneEuroPointFilter:
    """Noise-aware pointer filter for stable camera hand tracking.

    MediaPipe can produce isolated landmark jumps, especially when the hand is
    near the image border or the thumb briefly occludes the index finger. A
    one-frame jump is held for one update and confirmed only when the next
    frame continues toward the new position. Small stationary jitter is
    suppressed before it can inflate the One-Euro velocity estimate.

    The constructor remains compatible with existing b.py calls such as
    OneEuroPointFilter(min_cutoff=2.0, beta=0.045).
    """
    def __init__(
        self,
        min_cutoff: float = 1.35,
        beta: float = 0.025,
        derivative_cutoff: float = 0.85,
        noise_velocity: float = 78.0,
        minimum_jump_px: float = 22.0,
        deadband_px: float = 0.65,
    ):
        self.min_cutoff = max(0.75, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.derivative_cutoff = max(0.2, float(derivative_cutoff))
        self.noise_velocity = max(0.0, float(noise_velocity))
        self.minimum_jump_px = max(4.0, float(minimum_jump_px))
        self.deadband_px = max(0.0, float(deadband_px))
        self.raw_prev = None
        self.filtered_prev = None
        self.derivative_prev = (0.0, 0.0)
        self.time_prev = None
        self.pending_jump = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        dt = max(1e-3, min(0.25, float(dt)))
        tau = 1.0 / (2.0 * math.pi * max(1e-3, cutoff))
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _jump_limit(self, point) -> float:
        # Pixel scale is inferred from observed coordinates, so the same
        # stabilizer works with 640x480 and 1920x1080 camera frames.
        values = [abs(float(point[0])), abs(float(point[1]))]
        if self.raw_prev is not None:
            values.extend((abs(float(self.raw_prev[0])), abs(float(self.raw_prev[1]))))
        scale = max(240.0, *values)
        # A conservative fixed proportional gate rejects only large single-
        # frame jumps; intentional movement over consecutive frames is kept.
        return max(self.minimum_jump_px, scale * 0.034)

    def reset(self) -> None:
        self.raw_prev = None
        self.filtered_prev = None
        self.derivative_prev = (0.0, 0.0)
        self.time_prev = None
        self.pending_jump = None

    def update(self, point: tuple[float, float], timestamp: float) -> tuple[float, float]:
        raw = (float(point[0]), float(point[1]))
        if not all(math.isfinite(v) for v in raw):
            return self.filtered_prev if self.filtered_prev is not None else (0.0, 0.0)

        if self.raw_prev is None or self.filtered_prev is None or self.time_prev is None:
            self.raw_prev = raw
            self.filtered_prev = raw
            self.time_prev = float(timestamp)
            self.derivative_prev = (0.0, 0.0)
            self.pending_jump = None
            return raw

        dt = max(1e-3, min(0.25, float(timestamp) - self.time_prev))
        if dt > 0.18:
            # A long gap means the tracker was paused/lost. Re-acquire rather
            # than blending a new hand position with stale coordinates.
            self.raw_prev = raw
            self.filtered_prev = raw
            self.time_prev = float(timestamp)
            self.derivative_prev = (0.0, 0.0)
            self.pending_jump = None
            return raw

        jump_limit = self._jump_limit(raw)
        jump_from_previous = self._distance(raw, self.raw_prev)

        if self.pending_jump is not None:
            pending = self.pending_jump
            old_to_pending = (pending[0] - self.raw_prev[0], pending[1] - self.raw_prev[1])
            pending_to_now = (raw[0] - pending[0], raw[1] - pending[1])
            len_a = math.hypot(*old_to_pending)
            len_b = math.hypot(*pending_to_now)
            continuing = False
            if len_a > 1e-6 and len_b > 1e-6:
                cosine = (old_to_pending[0] * pending_to_now[0] + old_to_pending[1] * pending_to_now[1]) / (len_a * len_b)
                continuing = cosine >= 0.20
            near_pending = self._distance(raw, pending) <= max(jump_limit * 1.6, 12.0)
            returned_to_old = jump_from_previous <= jump_limit * 1.15
            if returned_to_old:
                # The previous frame was a transient landmark spike.
                self.pending_jump = None
            elif near_pending or continuing or jump_from_previous > jump_limit * 1.8:
                # The movement persisted for another frame: accept it.
                self.pending_jump = None
            else:
                # A second inconsistent large movement is treated as a new
                # candidate and held once, preventing a cursor teleport.
                self.pending_jump = raw
                return self.filtered_prev
        elif jump_from_previous > jump_limit:
            self.pending_jump = raw
            return self.filtered_prev

        candidate = raw
        raw_derivative = (
            (candidate[0] - self.raw_prev[0]) / dt,
            (candidate[1] - self.raw_prev[1]) / dt,
        )
        da = self._alpha(self.derivative_cutoff, dt)
        derivative = tuple(
            da * raw_derivative[i] + (1.0 - da) * self.derivative_prev[i]
            for i in range(2)
        )
        measured_speed = math.hypot(*derivative)
        # Ignore the normal camera-landmark noise floor. Without this offset,
        # small jitter looks like very fast hand motion and disables smoothing.
        intentional_speed = max(0.0, measured_speed - self.noise_velocity)
        cutoff = self.min_cutoff + self.beta * intentional_speed
        alpha = self._alpha(cutoff, dt)
        filtered = tuple(
            alpha * candidate[i] + (1.0 - alpha) * self.filtered_prev[i]
            for i in range(2)
        )

        # Stationary deadband: retain the filtered location for sub-pixel
        # changes instead of painting a trembling line while the hand is held.
        if self._distance(candidate, self.filtered_prev) <= self.deadband_px:
            filtered = self.filtered_prev

        self.raw_prev = candidate
        self.filtered_prev = filtered
        self.derivative_prev = derivative
        self.time_prev = float(timestamp)
        return filtered

