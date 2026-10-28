"""
Laser-Based Air Writer Smart Board
====================================
ZERO STAGE: Real-Time Red Laser Spot Detection and Tracking
(v2 - improved false-positive rejection)
 
This script ONLY detects and tracks a red laser dot using a phone
camera (via DroidCam) and OpenCV. It does NOT do any mouse control,
drawing, or calibration yet -- that comes in later stages.
 
WHAT'S NEW IN THIS VERSION (see chat explanation for full reasoning):
    - Local contrast check: a real laser is a sharp brightness PEAK
      compared to its immediate surroundings, not just "bright overall".
    - Halo red/orange dominance: we check color in the RING around the
      bright core, not the core itself (the core often clips to white).
    - Connected components (not raw contours) so touching/merged false
      blobs don't accidentally pass shape filters.
    - A continuous 0-100% confidence score instead of an instant on/off
      lock, with gradual decay and multi-frame confirmation before a
      NEW position is trusted, and before an EXISTING lock can be
      replaced by a "jump" candidate.
 
Controls:
    Q = Clear tracking and continue
    B = Recapture background (make sure the laser is OFF when you press B)
    ESC = Quit
"""
 
import cv2
import numpy as np
import time
from collections import deque
 
# ----------------------------------------------------------------------
# CONFIGURATION - tweak these numbers if detection is too strict/loose
# ----------------------------------------------------------------------
 
CAMERA_INDEX = 1                # DroidCam is camera index 1 on this laptop
 
BACKGROUND_CAPTURE_FRAMES = 30  # frames averaged to build the background
BACKGROUND_WARMUP_SECONDS = 3   # countdown so the user can turn the laser OFF
 
# --- Brightness / difference thresholds ---
DIFF_THRESHOLD = 40             # how much brighter than background a pixel must be
MIN_BRIGHTNESS = 200            # laser center is usually near-white / very bright (0-255)
 
# --- Candidate blob filtering (shape/size) ---
MIN_AREA = 3                    # laser dot is small; reject tiny noise specks
MAX_AREA = 400                  # reject big blobs (reflections, red objects, etc.)
MIN_CIRCULARITY = 0.5           # 1.0 = perfect circle. Laser dot should be fairly round
MAX_ASPECT_RATIO = 2.2          # reject long/thin shapes (reflections, text edges)
 
# --- NEW: local contrast (peak vs surrounding ring) ---
# This is the core anti-false-positive check. A real laser is MUCH
# brighter than the handful of pixels immediately around it. A warm
# reflection or overexposed patch is usually bright over a WIDE area,
# so its local contrast (center vs. immediate ring) is much lower.
CONTRAST_INNER_RADIUS = 6        # radius (px) sampled as "the candidate"
CONTRAST_OUTER_RADIUS = 16       # radius (px) of the outer edge of the ring
MIN_LOCAL_CONTRAST = 25          # inner brightness must exceed ring brightness by this much
 
# --- Background staleness guard (important when background = laptop screen) ---
STALE_CHANGE_RATIO = 0.08       # if more than 8% of pixels changed, flag as stale
STALE_WARNING_FRAMES = 5        # consecutive stale frames before we show a strong warning
 
# --- NEW: confidence-based temporal tracking ---
MAX_JUMP_DISTANCE = 120          # max px distance to count as "still the same laser"
SEED_TOLERANCE = 25              # px tolerance for "same new candidate, frame to frame"
CONFIRM_FRAMES_TO_LOCK = 4        # consecutive frames a NEW candidate must repeat before we trust it
CONFIDENCE_GAIN = 22             # confidence added per frame the laser is re-confirmed
CONFIDENCE_DECAY = 12            # confidence removed per frame the laser is NOT seen
SEED_START_CONFIDENCE = 45       # confidence granted the moment a brand-new lock is confirmed
LOCK_DISPLAY_THRESHOLD = 55      # confidence needed to show "LASER LOCKED" (vs "LASER TENTATIVE")
LOW_CONFIDENCE_SWITCH_THRESHOLD = 20  # a challenger may only take over once confidence drops this low
LOST_TRACK_FRAMES = 12           # frames with confidence at 0 before we fully drop the track
SMOOTHING_ALPHA = 0.5            # 0..1, higher = less smoothing / more responsive
 
# ----------------------------------------------------------------------
 
 
def capture_background(cap):
    """
    Capture and average several frames to build a stable background
    reference image WHILE THE LASER IS OFF.
 
    Why this matters: the laser is detected as a CHANGE against a known,
    stable scene. Averaging multiple frames removes random per-frame
    sensor noise and gives us a clean reference of "the room, laser off".
    """
    print("\n[Background Capture] Keep the laser OFF now...")
    for i in range(BACKGROUND_WARMUP_SECONDS, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
 
    frames = []
    collected = 0
    while collected < BACKGROUND_CAPTURE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame.astype(np.float32))
        collected += 1
 
    background = np.mean(frames, axis=0).astype(np.uint8)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    print("[Background Capture] Done. Background reference stored.\n")
    return background, background_gray
 
 
def build_color_mask(hsv):
    """
    Build a mask for laser-PLAUSIBLE colors.
 
    A laser dot is not simply "red" -- the very center often clips to
    white/pink/orange/yellow because the sensor saturates there. So we
    combine red (with hue wrap-around), orange/yellow, and a near-white
    but very-bright range (the blown-out laser core) into one mask.
    """
    red_lower1 = cv2.inRange(hsv, (0, 60, 150), (10, 255, 255))
    red_lower2 = cv2.inRange(hsv, (170, 60, 150), (180, 255, 255))
    orange_yellow = cv2.inRange(hsv, (10, 60, 150), (35, 255, 255))
    bright_white_core = cv2.inRange(hsv, (0, 0, 230), (180, 80, 255))
 
    color_mask = red_lower1 | red_lower2 | orange_yellow | bright_white_core
    return color_mask
 
 
def measure_local_contrast_and_halo(gray, bgr, cx, cy):
    """
    NEW: The key anti-false-positive measurement.
 
    Compares the brightness INSIDE a small disk at (cx, cy) against the
    brightness in a RING just outside it. A real laser is a sharp point
    of light -- its brightness drops off quickly just a few pixels away.
    A diffuse bright area (reflection, overexposed patch, warm light on
    a wall) stays roughly equally bright over a much wider area, so its
    "inner vs ring" contrast is much lower even if its absolute
    brightness is high.
 
    We also measure red/orange dominance in that same RING (the halo),
    not the center -- because the very center of a real laser is often
    a saturated white/pink blob with very little color information left
    in it. The red/orange glow lives just outside the blown-out core.
 
    Returns None if the candidate is too close to the frame edge to
    measure a full ring, otherwise (contrast, halo_red_dominance).
    """
    h, w = gray.shape[:2]
    r_out = CONTRAST_OUTER_RADIUS
    x0, x1 = int(cx - r_out), int(cx + r_out + 1)
    y0, y1 = int(cy - r_out), int(cy + r_out + 1)
 
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None  # too close to the edge to build a full ring
 
    sub_gray = gray[y0:y1, x0:x1]
    sub_bgr = bgr[y0:y1, x0:x1]
 
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
 
    inner_mask = dist <= CONTRAST_INNER_RADIUS
    ring_mask = (dist > CONTRAST_INNER_RADIUS) & (dist <= CONTRAST_OUTER_RADIUS)
 
    if inner_mask.sum() < 3 or ring_mask.sum() < 3:
        return None
 
    inner_brightness = float(sub_gray[inner_mask].mean())
    ring_brightness = float(sub_gray[ring_mask].mean())
    contrast = inner_brightness - ring_brightness
 
    ring_b = float(sub_bgr[:, :, 0][ring_mask].mean())
    ring_g = float(sub_bgr[:, :, 1][ring_mask].mean())
    ring_r = float(sub_bgr[:, :, 2][ring_mask].mean())
    halo_red_dominance = ring_r - max(ring_g, ring_b)
 
    return contrast, halo_red_dominance
 
 
def find_candidates(combined_mask, gray, diff, bgr):
    """
    Find and score all laser-plausible blobs in the combined binary mask.
 
    Uses CONNECTED COMPONENTS (not raw findContours) so that if two
    separate bright regions happen to touch in the mask, they are still
    treated and measured as separate blobs instead of one merged,
    oddly-shaped region that might slip past the shape filters.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined_mask, connectivity=8
    )
 
    candidates = []
 
    # label 0 is the background component; skip it
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < MIN_AREA or area > MAX_AREA:
            continue
 
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue
 
        # Isolate just this one component to measure its shape cleanly
        component_mask = np.zeros(gray.shape, dtype=np.uint8)
        component_mask[y:y + h, x:x + w] = (
            labels[y:y + h, x:x + w] == label
        ).astype(np.uint8) * 255
 
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
 
        cx, cy = centroids[label]
 
        # ---- NEW: local contrast + halo redness check ----
        contrast_result = measure_local_contrast_and_halo(gray, bgr, cx, cy)
        if contrast_result is None:
            continue
        local_contrast, halo_red_dominance = contrast_result
        if local_contrast < MIN_LOCAL_CONTRAST:
            continue  # not a sharp peak -> almost certainly not the laser
 
        mean_brightness = float(cv2.mean(gray, mask=component_mask)[0])
        mean_diff = float(cv2.mean(diff, mask=component_mask)[0])
        if mean_brightness < MIN_BRIGHTNESS * 0.6:
            continue
 
        # ---- Weighted score combining every independent clue ----
        # No single clue can push the score high on its own; a laser
        # should satisfy MOST of these at once.
        score = 0.0
        score += 0.22 * min(mean_brightness / 255.0, 1.0)
        score += 0.18 * min(mean_diff / 100.0, 1.0)
        score += 0.20 * circularity
        score += 0.20 * min(local_contrast / 80.0, 1.0)          # NEW
        score += 0.15 * min(max(halo_red_dominance, 0) / 80.0, 1.0)  # NEW
        score += 0.05 * (1.0 - min(area / MAX_AREA, 1.0))
 
        candidates.append({
            "center": (float(cx), float(cy)),
            "score": score,
            "local_contrast": local_contrast,
            "halo_red_dominance": halo_red_dominance,
        })
 
    return candidates
 
 
def _legacy_main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}. "
              f"Check that DroidCam is running and connected.")
        return
 
    print("=" * 60)
    print("LASER-BASED AIR WRITER - ZERO STAGE (v2)")
    print("Real-Time Laser Spot Detection and Tracking")
    print("=" * 60)
 
    background_bgr, background_gray = capture_background(cap)
 
    # ---- Tracking state ----
    tracked_position = None     # smoothed (x, y) we currently trust, or None
    confidence = 0.0            # 0..100 running confidence in the current track
    lost_counter = 0            # consecutive frames with nothing matching the track
 
    pending_seed = None         # a brand-new candidate we're waiting to confirm
    pending_count = 0
 
    stale_counter = 0           # consecutive frames where background looks "stale"
 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
 
    print("Laser tracking started.")
    print("Controls: Q = Quit | B = Recapture background (laser OFF)\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame from camera.")
            continue
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
        # ---- 1. Background subtraction ----
        diff = cv2.absdiff(gray, background_gray)
        _, diff_mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
 
        # ---- 1b. Background staleness check ----
        changed_pixels = cv2.countNonZero(diff_mask)
        total_pixels = diff_mask.shape[0] * diff_mask.shape[1]
        change_ratio = changed_pixels / total_pixels
        background_is_stale = change_ratio > STALE_CHANGE_RATIO
        stale_counter = stale_counter + 1 if background_is_stale else 0
 
        # ---- 2. Brightness mask ----
        _, bright_mask = cv2.threshold(gray, MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
 
        # ---- 3. Color mask ----
        color_mask = build_color_mask(hsv)
 
        # ---- 4. Combine ----
        combined = cv2.bitwise_and(diff_mask, bright_mask)
        combined = cv2.bitwise_and(combined, color_mask)
 
        # ---- 5. Morphology cleanup ----
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
 
        # ---- 6/7. Find + score candidates (skipped on stale background frames) ----
        candidates = [] if background_is_stale else find_candidates(combined, gray, diff, frame)
        candidates.sort(key=lambda c: c["score"], reverse=True)
 
        # ---- 8. Confidence-based temporal tracking ----
        matched = None
        if tracked_position is not None and candidates:
            tx, ty = tracked_position
            close_candidates = [
                c for c in candidates
                if np.hypot(c["center"][0] - tx, c["center"][1] - ty) <= MAX_JUMP_DISTANCE
            ]
            if close_candidates:
                close_candidates.sort(key=lambda c: c["score"], reverse=True)
                matched = close_candidates[0]
 
        if matched is not None:
            # The laser (or something consistent with it) is still where we
            # expect it to be. Update position smoothly and grow confidence.
            tx, ty = tracked_position
            nx, ny = matched["center"]
            tracked_position = (
                tx + SMOOTHING_ALPHA * (nx - tx),
                ty + SMOOTHING_ALPHA * (ny - ty),
            )
            confidence = min(100.0, confidence + CONFIDENCE_GAIN * matched["score"])
            lost_counter = 0
            pending_seed = None
            pending_count = 0
        else:
            # Nothing matched near the current track this frame.
            # First, see if there's a plausible NEW seed forming (this also
            # covers the very first lock, since tracked_position starts as None).
            best_new = candidates[0] if candidates else None
 
            can_switch = (tracked_position is None) or (confidence <= LOW_CONFIDENCE_SWITCH_THRESHOLD)
 
            if best_new is not None and can_switch:
                if (pending_seed is not None and
                        np.hypot(best_new["center"][0] - pending_seed["center"][0],
                                 best_new["center"][1] - pending_seed["center"][1]) <= SEED_TOLERANCE):
                    pending_count += 1
                else:
                    pending_seed = best_new
                    pending_count = 1
 
                if pending_count >= CONFIRM_FRAMES_TO_LOCK:
                    # The same spot has repeated for enough consecutive frames
                    # -- NOW we trust it as a genuine new laser position.
                    tracked_position = pending_seed["center"]
                    confidence = SEED_START_CONFIDENCE
                    lost_counter = 0
                    pending_seed = None
                    pending_count = 0
            else:
                # Either no candidates at all, or we're still confidently
                # locked elsewhere and this frame just missed a real laser
                # blob (occlusion, motion blur, brief exposure change).
                # Don't discard the seed progress on a single skipped frame,
                # but don't grow it either.
                pass
 
            if tracked_position is not None and matched is None:
                # We didn't get a fresh confirmation this frame -> decay
                # confidence and hold the last known position (grace period)
                # instead of snapping to something else immediately.
                confidence = max(0.0, confidence - CONFIDENCE_DECAY)
                lost_counter += 1
                if lost_counter > LOST_TRACK_FRAMES or confidence <= 0.0:
                    tracked_position = None
                    confidence = 0.0
                    lost_counter = 0
 
        # ---- 9. Determine display status from confidence ----
        if tracked_position is not None and confidence >= LOCK_DISPLAY_THRESHOLD:
            status_text = f"LASER LOCKED - CONFIDENCE: {int(confidence)}%"
            status_color = (0, 255, 0)
        elif tracked_position is not None:
            status_text = f"LASER TENTATIVE - CONFIDENCE: {int(confidence)}%"
            status_color = (0, 200, 255)
        else:
            status_text = "NO LASER DETECTED"
            status_color = (0, 0, 255)
 
        # ---- 10. Draw overlay ----
        display = frame.copy()
        if tracked_position is not None:
            px, py = int(tracked_position[0]), int(tracked_position[1])
            circle_color = (0, 255, 0) if confidence >= LOCK_DISPLAY_THRESHOLD else (0, 200, 255)
            cv2.circle(display, (px, py), 12, circle_color, 2)
            cv2.circle(display, (px, py), 2, (0, 0, 255), -1)
            cv2.putText(display, f"X:{px} Y:{py}", (px + 15, py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)
 
        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
 
        if background_is_stale:
            warn_text = f"BACKGROUND CHANGED ({change_ratio*100:.1f}% of frame) - press B"
            cv2.putText(display, warn_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            if stale_counter >= STALE_WARNING_FRAMES:
                cv2.putText(display, "SCREEN CONTENT LIKELY MOVING -- USE A STATIC IMAGE",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(display, f"background change: {change_ratio*100:.1f}%",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
 
        cv2.putText(display, "Q=Quit  B=Recapture background (laser OFF)",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
        cv2.imshow("Air Writer - Live Camera", display)
        cv2.imshow("Diagnostic: Combined Mask", combined)
        cv2.imshow("Diagnostic: Color Mask", color_mask)
        cv2.imshow("Diagnostic: Brightness Diff", diff_mask)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            background_bgr, background_gray = capture_background(cap)
            tracked_position = None
            confidence = 0.0
            lost_counter = 0
            pending_seed = None
            pending_count = 0
            stale_counter = 0
 
    cap.release()
    cv2.destroyAllWindows()
 
 
def main():
    """Run the trained YOLO plus OpenCV detector using this module's helpers."""
    from air_writer_yolo import main as verified_main
    verified_main()


if __name__ == "__main__":
    main()
 
