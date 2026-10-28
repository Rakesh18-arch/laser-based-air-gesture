"""
AirWriter - Cursor Control with laser LEFT / RIGHT click
========================================================
 
Uses the project's existing pieces:
    air_writer.py        -> camera index, background capture
    air_writer_yolo.py   -> trained YOLO laser verifier (classify)
    laser_calibration.py -> laser_homography.txt (camera -> screen mapping)
 
GESTURES (exactly two, nothing else)
    LEFT  click :  ON -> OFF
    RIGHT click :  ON -> OFF -> ON -> OFF
 
A click is only ever produced from the VERIFIED laser signal:
    halo+core geometry  ->  YOLO  ->  multi-frame confirmation.
Anything uncertain is treated as "tracking lost" and NEVER clicks.
 
Controls (preview window focused):
    B   = recapture background (laser OFF!)
    C   = enable / disable real clicks (dry run)
    V   = hide / show preview  (hidden: quit with Ctrl+C in the terminal)
    ESC = quit
"""
 
import sys
import time
from collections import deque
from enum import Enum
from pathlib import Path
 
import cv2
import numpy as np
 
import air_writer as aw
 
try:
    import pyautogui
except Exception:                       # lets the logic be unit-tested headless
    pyautogui = None
 
 
# ======================================================================
# CONFIG
# ======================================================================
HOMOGRAPHY_FILE = "laser_homography.txt"     # written by laser_calibration.py
CAMERA_INDEX = aw.CAMERA_INDEX
 
# ---- YOLO gate (same numbers your HUD already shows) ------------------
YOLO_SEED_PROB = 0.80        # needed on every frame while ACQUIRING a lock
YOLO_KEEP_PROB = 0.60        # needed to KEEP an existing lock
YOLO_WEIGHT_VALUE = 0.85
OPENCV_WEIGHT_VALUE = 0.15
MIN_FINAL_SCORE = 0.50
MAX_YOLO_CANDIDATES = 4
 
# ---- Candidate geometry: clipped-white core + coloured halo ----------
CORE_MIN_CHANNEL = 170       # every channel >= this  -> "clipped core" pixel
CORE_MIN_DIFF = 30           # core must also be brighter than background
CORE_MIN_AREA = 2
CORE_MAX_AREA = 150          # bigger = glare / white patch, not a laser
PINK_RG_MIN = 40             # halo: R - G >= this (works for red AND pink/magenta)
PINK_R_MIN = 100
PINK_RG_CHANGE_MIN = 25      # ...and it must be NEW vs background (static red UI ignored)
HALO_RADIUS = 10
HALO_MIN_PIXELS = 10
HALO_MIN_QUADRANTS = 3       # halo must surround the core, not sit on one side
HALO_QUADRANT_MIN_PIXELS = 2
CONTRAST_MIN = 40            # core vs ring 9-15px away
WIDE_BRIGHT_MAX_FRACTION = 0.12
WINDOW_R = 15
 
# ---- Multi-frame ON confirmation --------------------------------------
CONFIRM_ON_FRAMES = 3
SEED_TOLERANCE = 28          # px: same spot frame-to-frame while confirming
TRACK_GATE_DISTANCE = 85     # px: max jump for "same laser"
SMOOTHING_ALPHA = 0.50
 
# ---- Clean-OFF rules (what separates OFF from "tracking lost") --------
OFF_CONFIRM_FRAMES = 3
OFF_CONFIRM_SECONDS = 0.10
LOST_GRACE_FRAMES = 8
LOST_GRACE_SECONDS = 0.40
MIN_ON_FRAMES = 4            # laser must have been verified this long before OFF counts
STATIONARY_FRAMES = 4
STATIONARY_MAX_PX = 14       # laser must be (nearly) still right before it turns off
NEAR_EVIDENCE_PX = 45        # laser-like blob this close to last spot => NOT a clean OFF
PATCH_R = 20
PATCH_CHANGED_DIFF = 40
PATCH_MAX_CHANGED_FRACTION = 0.15   # area where laser was must look like background again
 
# ---- Gesture timing ----------------------------------------------------
SECOND_CYCLE_TIMEOUT = 0.80  # after 1st OFF: wait this long for a 2nd ON
SECOND_ON_CONFIRM_GRACE = 0.40  # extra time if a 2nd ON is mid-confirmation
MIN_CLICK_INTERVAL = 0.25    # hard rate limit between any two clicks
 
# ---- Screen / cursor ---------------------------------------------------
CURSOR_EDGE_INSET = 2        # keeps cursor off (0,0) -> pyautogui fail-safe corner
ANCHOR_RECHECK_PX = 3
 
# ---- Background --------------------------------------------------------
BG_ADAPT_ALPHA = 0.01        # slow adaptation, only when no laser evidence at all
 
# ---- Preview (camera sees the screen -> preview must not be detected) --
START_WITH_PREVIEW = True
PREVIEW_SCALE = 0.6
PREVIEW_POS = (0, 0)
PREVIEW_EXCLUDE_MARGIN = 45  # screen px around the preview window
WINDOW = "AirWriter - Cursor Control"
 
DRY_RUN_AT_START = False     # True = log clicks but do not send them
 
 
# ======================================================================
# SCREEN MAPPING (existing calibration output)
# ======================================================================
class ScreenMapper:
    """camera px -> screen px using laser_homography.txt from calibration."""
 
    def __init__(self, H, screen_w, screen_h):
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            raise ValueError("Homography must be a finite 3x3 matrix.")
        self.H = H
        self.Hinv = np.linalg.inv(H)
        self.W = int(screen_w)
        self.H_px = int(screen_h)
 
    @classmethod
    def load(cls, screen_w, screen_h):
        for base in (Path(__file__).resolve().parent, Path.cwd()):
            path = base / HOMOGRAPHY_FILE
            if path.is_file():
                print(f"[Mapping] Loaded {path}")
                return cls(np.loadtxt(str(path)), screen_w, screen_h)
        raise FileNotFoundError(
            f"{HOMOGRAPHY_FILE} not found. Run laser_calibration.py first.")
 
    @staticmethod
    def _apply(M, pts):
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, M.astype(np.float32)).reshape(-1, 2)
 
    def cam_to_screen(self, pt):
        q = self._apply(self.H, [pt])[0]
        return float(q[0]), float(q[1])
 
    def screen_to_cam(self, pts):
        return self._apply(self.Hinv, pts)
 
    def on_screen(self, sx, sy):
        return 0.0 <= sx <= self.W - 1 and 0.0 <= sy <= self.H_px - 1
 
    def clamp(self, sx, sy):
        i = CURSOR_EDGE_INSET
        return (int(min(max(round(sx), i), self.W - 1 - i)),
                int(min(max(round(sy), i), self.H_px - 1 - i)))
 
    def screen_polygon_in_camera(self):
        corners = [(0, 0), (self.W - 1, 0), (self.W - 1, self.H_px - 1), (0, self.H_px - 1)]
        return self.screen_to_cam(corners)
 
 
# ======================================================================
# MOUSE (single choke point for every button / move)
# ======================================================================
class Mouse:
    def __init__(self, dry_run=False, clock=time.monotonic):
        self.dry_run = dry_run
        self.clock = clock
        self.held = set()               # buttons WE pressed (none in this program)
        self.last_click_t = -1e9
        self.last_xy = None
 
    def move(self, x, y):
        if self.last_xy is not None and abs(x - self.last_xy[0]) <= 0 and abs(y - self.last_xy[1]) <= 0:
            return
        self.last_xy = (x, y)
        pyautogui.moveTo(x, y, duration=0, _pause=False)
 
    def click(self, button, x, y):
        """Click at (x, y). Returns True if a click was really sent."""
        now = self.clock()
        if now - self.last_click_t < MIN_CLICK_INTERVAL:
            print(f"[Mouse] {button.upper()} click suppressed (rate limit)")
            return False
        self.last_click_t = now
        label = "LEFT" if button == "left" else "RIGHT"
        if self.dry_run:
            print(f"[DRY RUN] {label} CLICK at ({x}, {y})")
            return True
        cur = pyautogui.position()
        if abs(cur[0] - x) > ANCHOR_RECHECK_PX or abs(cur[1] - y) > ANCHOR_RECHECK_PX:
            pyautogui.moveTo(x, y, duration=0, _pause=False)
            self.last_xy = (x, y)
        pyautogui.click(button=button)
        print(f"{label} CLICK at ({x}, {y})")
        return True
 
    def release_all(self):
        """Fail-safe: release any button this program might be holding."""
        for button in list(self.held):
            try:
                if not self.dry_run:
                    pyautogui.mouseUp(button=button)
            finally:
                self.held.discard(button)
 
 
# ======================================================================
# CANDIDATE DETECTION  (clipped core + coloured halo, background-aware)
# ======================================================================
def detect_laser_candidates(frame, bg_f, stats):
    """
    Returns (candidates, diff). A candidate passes ONLY if all hold:
      * clipped-white core, brighter than background, small
      * halo (R >> G) that is NEW vs background and surrounds the core
      * sharp local peak (not a wide bright patch)
    The colour test is R-G (not a hue band) so red, orange and the
    magenta-pink halo of phone cameras are all accepted.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(cv2.convertScaleAbs(bg_f), cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray)
 
    b, g, r = cv2.split(frame)
    rg = r.astype(np.int16) - g.astype(np.int16)
    bg_rg = bg_f[:, :, 2] - bg_f[:, :, 1]
    minch = np.minimum(np.minimum(b, g), r)
 
    core = ((minch >= CORE_MIN_CHANNEL) & (diff >= CORE_MIN_DIFF)).astype(np.uint8) * 255
    pink = (rg >= PINK_RG_MIN) & (r >= PINK_R_MIN) & ((rg - bg_rg) >= PINK_RG_CHANGE_MIN)
 
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core_c = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k)
    n, labels, st, cen = cv2.connectedComponentsWithStats(core_c, connectivity=8)
 
    cands = []
    for label in range(1, n):
        area = int(st[label, cv2.CC_STAT_AREA])
        if area < CORE_MIN_AREA or area > CORE_MAX_AREA:
            stats["size"] += 1
            continue
        stats["core"] += 1
 
        cx0, cy0 = float(cen[label][0]), float(cen[label][1])
        ix, iy = int(round(cx0)), int(round(cy0))
        x0, x1 = max(0, ix - WINDOW_R), min(w, ix + WINDOW_R + 1)
        y0, y1 = max(0, iy - WINDOW_R), min(h, iy + WINDOW_R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        pk = pink[y0:y1, x0:x1]
        core_l = core[y0:y1, x0:x1] > 0
 
        # ---- halo present? ----
        near = pk & (np.hypot(xx - cx0, yy - cy0) <= HALO_RADIUS)
        if int(near.sum()) < HALO_MIN_PIXELS:
            stats["no_halo"] += 1
            continue
        px, py = float(xx[near].mean()), float(yy[near].mean())
 
        # ---- centre = core pixels around the halo centroid ----
        cr = core_l & (np.hypot(xx - px, yy - py) <= 6)
        if cr.any():
            cx, cy = float(xx[cr].mean()), float(yy[cr].mean())
        else:
            cx, cy = px, py
        d = np.hypot(xx - cx, yy - cy)
 
        # ---- halo must SURROUND the core ----
        halo = pk & (d <= HALO_RADIUS + 1) & (d >= 1)
        hx, hy = xx[halo] - cx, yy[halo] - cy
        quads = 0
        for sx in (hx >= 0, hx < 0):
            for sy in (hy >= 0, hy < 0):
                if int((sx & sy).sum()) >= HALO_QUADRANT_MIN_PIXELS:
                    quads += 1
        near_border = (cx < WINDOW_R or cy < WINDOW_R or
                       cx > w - 1 - WINDOW_R or cy > h - 1 - WINDOW_R)
        if quads < (2 if near_border else HALO_MIN_QUADRANTS):
            stats["shape"] += 1
            continue
 
        # ---- sharp peak, not a wide bright patch ----
        gl = gray[y0:y1, x0:x1].astype(np.float32)
        inner = gl[d <= 3]
        ring_mask = (d >= 9) & (d <= 15)
        if inner.size < 3 or int(ring_mask.sum()) < 20:
            stats["shape"] += 1
            continue
        contrast = float(inner.mean() - gl[ring_mask].mean())
        wide_frac = float((core_l & ring_mask).sum()) / float(ring_mask.sum())
        core_total = int((core_l & (d <= 15)).sum())
        if contrast < CONTRAST_MIN or wide_frac > WIDE_BRIGHT_MAX_FRACTION or core_total > CORE_MAX_AREA:
            stats["wide"] += 1
            continue
 
        dsub = diff[y0:y1, x0:x1]
        cm = core_l & (d <= 6)
        mean_diff = float(dsub[cm].mean()) if cm.any() else 0.0
        halo_count = int(halo.sum())
        score = (0.35 * min(halo_count / 40.0, 1.0)
                 + 0.25 * min(contrast / 120.0, 1.0)
                 + 0.15 * min(mean_diff / 150.0, 1.0)
                 + 0.15 * quads / 4.0
                 + 0.10 * (1.0 - min(core_total / float(CORE_MAX_AREA), 1.0)))
        cands.append({"center": (cx, cy), "score": float(score), "halo": halo_count,
                      "contrast": contrast, "inside": True, "excluded": False,
                      "yolo": 0.0})
 
    # de-duplicate blobs that resolved to the same centre
    cands.sort(key=lambda c: c["score"], reverse=True)
    unique = []
    for c in cands:
        if all(np.hypot(c["center"][0] - u["center"][0], c["center"][1] - u["center"][1]) > 6
               for u in unique):
            unique.append(c)
    return unique, diff
 
 
# ======================================================================
# YOLO GATE (existing verifier)
# ======================================================================
class YoloGate:
    def __init__(self):
        import air_writer_yolo as ay        # loads your trained best.pt
        self.ay = ay
 
    def prob(self, frame, center):
        return float(self.ay.classify(frame, {"center": center}))
 
 
# ======================================================================
# VERIFIED ON / OFF SIGNAL
# ======================================================================
class Sig(Enum):
    ABSENT = "OFF"
    PENDING = "CONFIRMING"
    PRESENT = "ON"
 
 
class Ev(Enum):
    ON = "ON"
    OFF_CLEAN = "OFF"
    LOST = "LOST"
 
 
class LaserSignal:
    """
    Turns per-frame verification results into three clean events:
        ON         laser verified for CONFIRM_ON_FRAMES consecutive frames
        OFF_CLEAN  laser was steady, then vanished with NO laser-like evidence
                   left and the area looking like background again
        LOST       vanished/degraded in any other way (moving, ambiguous,
                   occluded, YOLO unsure...)  -> must never click
    """
 
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.state = Sig.ABSENT
        self.pending_pos = None
        self.pending_n = 0
        self.pos = None                       # raw camera position while ON
        self.hist = deque(maxlen=8)
        self.on_frames = 0
        self.missing_n = 0
        self.missing_t0 = None
        self.clean_n = 0
        self.clean_t0 = None
 
    def recent_positions(self, n=3):
        return list(self.hist)[-n:]
 
    def _stationary(self):
        pts = list(self.hist)[-STATIONARY_FRAMES:]
        if len(pts) < STATIONARY_FRAMES:
            return False
        last = pts[-1]
        return all(np.hypot(p[0] - last[0], p[1] - last[1]) <= STATIONARY_MAX_PX for p in pts)
 
    def update(self, now, verified, evidence, healthy, patch_clean):
        """verified: candidate dict or None. Returns Ev or None."""
        if self.state in (Sig.ABSENT, Sig.PENDING):
            if verified is None:
                self.state = Sig.ABSENT
                self.pending_pos, self.pending_n = None, 0
                return None
            pos = verified["center"]
            if (self.pending_pos is not None and
                    np.hypot(pos[0] - self.pending_pos[0], pos[1] - self.pending_pos[1]) <= SEED_TOLERANCE):
                self.pending_n += 1
            else:
                self.pending_n = 1
            self.pending_pos = pos
            self.state = Sig.PENDING
            if self.pending_n >= CONFIRM_ON_FRAMES:
                self.state = Sig.PRESENT
                self.pos = pos
                self.hist.clear()
                self.hist.append(pos)
                self.on_frames = 1
                self.missing_n, self.missing_t0 = 0, None
                self.clean_n, self.clean_t0 = 0, None
                self.pending_pos, self.pending_n = None, 0
                return Ev.ON
            return None
 
        # ---- PRESENT ----
        if verified is not None:
            self.pos = verified["center"]
            self.hist.append(self.pos)
            self.on_frames += 1
            self.missing_n, self.missing_t0 = 0, None
            self.clean_n, self.clean_t0 = 0, None
            return None
 
        self.missing_n += 1
        if self.missing_t0 is None:
            self.missing_t0 = now
        if (not evidence) and healthy and patch_clean:
            self.clean_n += 1
            if self.clean_t0 is None:
                self.clean_t0 = now
        else:
            self.clean_n, self.clean_t0 = 0, None
 
        if (self.clean_n >= OFF_CONFIRM_FRAMES and
                now - self.clean_t0 >= OFF_CONFIRM_SECONDS):
            steady = self.on_frames >= MIN_ON_FRAMES and self._stationary()
            self.state = Sig.ABSENT
            return Ev.OFF_CLEAN if steady else Ev.LOST
 
        if self.missing_n >= LOST_GRACE_FRAMES or now - self.missing_t0 >= LOST_GRACE_SECONDS:
            self.state = Sig.ABSENT
            return Ev.LOST
        return None
 
 
# ======================================================================
# GESTURE STATE MACHINE
# ======================================================================
class G(Enum):
    IDLE = "IDLE"
    FIRST_ON = "FIRST_ON"
    WAITING_FOR_SECOND_CYCLE = "WAITING_FOR_SECOND_CYCLE"
    SECOND_ON = "SECOND_ON"
 
 
class GestureMachine:
    """
    IDLE --ON--> FIRST_ON --OFF--> WAITING_FOR_SECOND_CYCLE
    WAITING --ON--> SECOND_ON --OFF--> RIGHT CLICK -> IDLE
    WAITING --timeout--> LEFT CLICK -> IDLE
    Any LOST: never clicks by itself.
        FIRST_ON  + LOST -> IDLE
        SECOND_ON + LOST -> WAITING (fresh timeout; original ON->OFF becomes
                            a LEFT click only if the timeout expires)
    """
 
    def __init__(self, mouse):
        self.mouse = mouse
        self.state = G.IDLE
        self.anchor = None          # screen (x, y) where the 1st cycle ended
        self.t_off = None
 
    def reset(self):
        self.state = G.IDLE
        self.anchor = None
        self.t_off = None
 
    def on_event(self, ev, now, screen_pos):
        if ev is Ev.LOST:
            self.mouse.release_all()
            if self.state is G.FIRST_ON:
                self.reset()
            elif self.state is G.SECOND_ON:
                self.state = G.WAITING_FOR_SECOND_CYCLE
                self.t_off = now
            return
 
        if ev is Ev.ON:
            if self.state is G.IDLE:
                self.state = G.FIRST_ON
            elif self.state is G.WAITING_FOR_SECOND_CYCLE:
                self.state = G.SECOND_ON
            return
 
        if ev is Ev.OFF_CLEAN:
            if self.state is G.FIRST_ON:
                self.anchor = screen_pos
                self.t_off = now
                self.state = G.WAITING_FOR_SECOND_CYCLE
            elif self.state is G.SECOND_ON:
                self.mouse.click("right", *screen_pos)
                self.reset()
 
    def tick(self, now, second_on_confirming):
        if self.state is not G.WAITING_FOR_SECOND_CYCLE:
            return
        elapsed = now - self.t_off
        if elapsed < SECOND_CYCLE_TIMEOUT:
            return
        if second_on_confirming and elapsed < SECOND_CYCLE_TIMEOUT + SECOND_ON_CONFIRM_GRACE:
            return
        self.mouse.click("left", *self.anchor)
        self.reset()
 
 
# ======================================================================
# HELPERS
# ======================================================================
def median_point(points):
    a = np.asarray(points, dtype=np.float64)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))
 
 
def select_verified(frame, usable, tracked, yolo):
    """
    YOLO-verify candidates. Returns (chosen, verified_far, best_prob).
    tracked = raw camera pos of current lock (or None).
    """
    if not usable:
        return None, [], 0.0
    if tracked is not None:
        order = sorted(usable, key=lambda c: np.hypot(c["center"][0] - tracked[0],
                                                     c["center"][1] - tracked[1]))
    else:
        order = sorted(usable, key=lambda c: c["score"], reverse=True)
 
    verified, best_p = [], 0.0
    for c in order[:MAX_YOLO_CANDIDATES]:
        dist = (np.hypot(c["center"][0] - tracked[0], c["center"][1] - tracked[1])
                if tracked is not None else None)
        in_gate = dist is not None and dist <= TRACK_GATE_DISTANCE
        thr = YOLO_KEEP_PROB if in_gate else YOLO_SEED_PROB
        p = yolo.prob(frame, c["center"])
        c["yolo"] = p
        c["final"] = YOLO_WEIGHT_VALUE * p + OPENCV_WEIGHT_VALUE * c["score"]
        c["dist"] = dist
        best_p = max(best_p, p)
        if p >= thr and c["final"] >= MIN_FINAL_SCORE:
            verified.append(c)
 
    if tracked is not None:
        gated = [c for c in verified if c["dist"] <= TRACK_GATE_DISTANCE]
        far = [c for c in verified if c["dist"] > TRACK_GATE_DISTANCE]
        if gated:
            gated.sort(key=lambda c: 0.65 * c["final"] + 0.35 * (1 - c["dist"] / TRACK_GATE_DISTANCE),
                       reverse=True)
            return gated[0], far, best_p
        return None, far, best_p
 
    if not verified:
        return None, [], best_p
    verified.sort(key=lambda c: c["final"], reverse=True)
    if len(verified) > 1:
        a, b = verified[0], verified[1]
        sep = np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
        if sep > SEED_TOLERANCE and a["final"] - b["final"] < 0.08:
            return None, [], best_p             # two equally good spots = ambiguous
    return verified[0], [], best_p
 
 
def preview_exclusion_mask(shape, mapper, win_rect):
    """Camera-space mask of the region where our own preview window sits."""
    mask = np.zeros(shape[:2], np.uint8)
    if win_rect is None:
        return mask
    x, y, ww, hh = win_rect
    m = PREVIEW_EXCLUDE_MARGIN
    quad = [(x - m, y - m - 30), (x + ww + m, y - m - 30),
            (x + ww + m, y + hh + m), (x - m, y + hh + m)]
    poly = mapper.screen_to_cam(quad)
    if np.all(np.isfinite(poly)):
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask
 
 
def get_preview_rect(display_shape):
    try:
        r = cv2.getWindowImageRect(WINDOW)
        if r is not None and r[2] > 0 and r[3] > 0:
            return tuple(int(v) for v in r)
    except cv2.error:
        pass
    return (PREVIEW_POS[0], PREVIEW_POS[1], display_shape[1], display_shape[0])
 
 
# ======================================================================
# MAIN
# ======================================================================
def main():
    if pyautogui is None:
        print("ERROR: pyautogui is not installed (pip install pyautogui).")
        return
    pyautogui.PAUSE = 0
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.FAILSAFE = True                 # slam the real mouse into a corner to abort
    sw, sh = pyautogui.size()
 
    try:
        mapper = ScreenMapper.load(sw, sh)
    except Exception as exc:
        print("ERROR:", exc)
        return
 
    print("[YOLO] Loading verifier...")
    yolo = YoloGate()
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # less latency
    except Exception:
        pass
 
    print("=" * 66)
    print("AIRWRITER - CURSOR CONTROL")
    print("  LEFT  click : ON -> OFF")
    print("  RIGHT click : ON -> OFF -> ON -> OFF")
    print(f"  Second-cycle timeout: {SECOND_CYCLE_TIMEOUT}s")
    print("=" * 66)
 
    bg_bgr, _ = aw.capture_background(cap)
    bg_f = bg_bgr.astype(np.float32)
 
    ok, first = cap.read()
    while not ok:
        ok, first = cap.read()
    fh, fw = first.shape[:2]
    print(f"[Camera] {fw}x{fh}  (must match the resolution used in calibration)")
    cx_s, cy_s = mapper.cam_to_screen((fw / 2.0, fh / 2.0))
    if not mapper.on_screen(cx_s, cy_s):
        print("WARNING: camera centre maps outside the screen - "
              "re-run laser_calibration.py (camera moved / resolution changed?).")
 
    mouse = Mouse(dry_run=DRY_RUN_AT_START)
    signal = LaserSignal()
    machine = GestureMachine(mouse)
    screen_poly = np.round(mapper.screen_polygon_in_camera()).astype(np.int32)
 
    show_preview = START_WITH_PREVIEW
    excl_mask = np.zeros((fh, fw), np.uint8)
    win_ready = False
    frame_i = 0
    bright_ema = None
    smooth = None
    last_event = "-"
    fps, t_fps, n_fps = 0.0, time.time(), 0
 
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.monotonic()
            frame_i += 1
            n_fps += 1
            if time.time() - t_fps >= 1.0:
                fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()
 
            # ---------- detection ----------
            stats = {"core": 0, "size": 0, "no_halo": 0, "shape": 0, "wide": 0,
                     "outside": 0, "excl": 0, "yolo_rej": 0}
            cands, diff = detect_laser_candidates(frame, bg_f, stats)
 
            if show_preview and win_ready and frame_i % 10 == 0:
                rect = get_preview_rect((int(fh * PREVIEW_SCALE), int(fw * PREVIEW_SCALE)))
                excl_mask = preview_exclusion_mask(frame.shape, mapper, rect)
            elif not show_preview:
                excl_mask[:] = 0
 
            usable = []
            for c in cands:
                cx, cy = c["center"]
                sx, sy = mapper.cam_to_screen((cx, cy))
                c["screen_raw"] = (sx, sy)
                c["inside"] = mapper.on_screen(sx, sy)
                ix = int(min(max(round(cx), 0), fw - 1))
                iy = int(min(max(round(cy), 0), fh - 1))
                c["excluded"] = bool(excl_mask[iy, ix])
                if not c["inside"]:
                    stats["outside"] += 1
                elif c["excluded"]:
                    stats["excl"] += 1
                else:
                    usable.append(c)
 
            tracked = signal.pos if signal.state is Sig.PRESENT else None
            chosen, far, best_p = select_verified(frame, usable, tracked, yolo)
            stats["yolo_rej"] = sum(1 for c in usable if c is not chosen and c["yolo"] < YOLO_SEED_PROB and c["yolo"] > 0)
 
            # ---------- evidence used to tell OFF from "lost" ----------
            evidence, patch_clean = False, True
            if signal.state is Sig.PRESENT and chosen is None:
                lx, ly = signal.pos
                evidence = bool(far) or any(
                    np.hypot(c["center"][0] - lx, c["center"][1] - ly) <= NEAR_EVIDENCE_PX
                    for c in cands)
                x0, x1 = max(0, int(lx) - PATCH_R), min(fw, int(lx) + PATCH_R + 1)
                y0, y1 = max(0, int(ly) - PATCH_R), min(fh, int(ly) + PATCH_R + 1)
                patch = diff[y0:y1, x0:x1]
                patch_clean = bool(patch.size and
                                   float((patch >= PATCH_CHANGED_DIFF).mean()) <= PATCH_MAX_CHANGED_FRACTION)
            gmean = float(frame.mean())
            if bright_ema is None:
                bright_ema = gmean
            healthy = 0.5 * bright_ema <= gmean <= 1.8 * bright_ema
            if healthy:
                bright_ema += 0.02 * (gmean - bright_ema)
 
            # ---------- verified signal -> events ----------
            ev = signal.update(now, chosen, evidence, healthy, patch_clean)
 
            # cursor: ONLY on verified frames of a confirmed lock
            if signal.state is Sig.PRESENT and chosen is not None:
                cx, cy = chosen["center"]
                smooth = (cx, cy) if (ev is Ev.ON or smooth is None) else (
                    smooth[0] + SMOOTHING_ALPHA * (cx - smooth[0]),
                    smooth[1] + SMOOTHING_ALPHA * (cy - smooth[1]))
                sx, sy = mapper.cam_to_screen(smooth)
                if mapper.on_screen(sx, sy):
                    mouse.move(*mapper.clamp(sx, sy))
 
            if ev is not None:
                last_event = ev.value
                spos = None
                if ev is Ev.OFF_CLEAN:
                    sxm, sym = mapper.cam_to_screen(median_point(signal.recent_positions(3)))
                    spos = mapper.clamp(sxm, sym)
                machine.on_event(ev, now, spos)
                if ev is not Ev.ON:
                    smooth = None
            machine.tick(now, signal.state is Sig.PENDING)
 
            # ---------- adaptive background (only when nothing laser-like) ----------
            if signal.state is Sig.ABSENT and not cands and machine.state in (G.IDLE, G.WAITING_FOR_SECOND_CYCLE):
                cv2.accumulateWeighted(frame, bg_f, BG_ADAPT_ALPHA)
 
            # ---------- preview ----------
            if show_preview:
                disp = frame.copy()
                cv2.polylines(disp, [screen_poly], True, (255, 120, 0), 1)
                if excl_mask.any():
                    cnts, _ = cv2.findContours(excl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(disp, cnts, -1, (120, 120, 120), 1)
                for c in cands:
                    p = (int(c["center"][0]), int(c["center"][1]))
                    col = (0, 165, 255) if c is not chosen else (0, 255, 0)
                    if not c["inside"] or c["excluded"]:
                        col = (128, 128, 128)
                    cv2.circle(disp, p, 9, col, 1)
                lc = (0, 255, 0) if signal.state is Sig.PRESENT else (
                    (0, 200, 255) if signal.state is Sig.PENDING else (0, 0, 255))
                cv2.putText(disp, f"Laser: {signal.state.value}   Gesture: {machine.state.value}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2)
                cv2.putText(disp, f"core:{stats['core']} nohalo:{stats['no_halo']} shape:{stats['shape']} "
                                  f"wide:{stats['wide']} out:{stats['outside']} excl:{stats['excl']}",
                            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(disp, f"YOLO best:{best_p:.2f} (seed>={YOLO_SEED_PROB:.2f} keep>={YOLO_KEEP_PROB:.2f}) "
                                  f"last:{last_event}",
                            (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(disp, f"clicks:{'DRY-RUN' if mouse.dry_run else 'LIVE'}  fps:{fps:.0f}",
                            (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                small = cv2.resize(disp, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                if not win_ready:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow(WINDOW, *PREVIEW_POS)
                    win_ready = True
                cv2.imshow(WINDOW, small)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
 
            if key == 27:
                break
            elif key == ord("c"):
                mouse.dry_run = not mouse.dry_run
                print("[Clicks]", "DRY-RUN" if mouse.dry_run else "LIVE")
            elif key == ord("v"):
                show_preview = not show_preview
                if not show_preview and win_ready:
                    cv2.destroyWindow(WINDOW)
                    win_ready = False
                print("[Preview]", "shown" if show_preview else "hidden (Ctrl+C in terminal quits)")
            elif key == ord("b"):
                if signal.state is Sig.ABSENT:
                    mouse.release_all()
                    machine.reset()
                    signal.reset()
                    bg_bgr, _ = aw.capture_background(cap)
                    bg_f = bg_bgr.astype(np.float32)
                    print("Background recaptured.")
                else:
                    print("Turn the laser OFF before recapturing the background.")
 
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if pyautogui is not None and isinstance(exc, pyautogui.FailSafeException):
            print("PyAutoGUI fail-safe triggered - stopping.")
        else:
            raise
    finally:
        mouse.release_all()
        cap.release()
        cv2.destroyAllWindows()
        print("Cursor control stopped.")
 
 
if __name__ == "__main__":
    main()
 