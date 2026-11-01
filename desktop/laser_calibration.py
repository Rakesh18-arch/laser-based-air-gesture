
import cv2
import numpy as np
import time
from ultralytics import YOLO
import air_writer as aw


# ============================================================
# SETTINGS
# ============================================================

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

STABILITY_FRAMES = 8
STABILITY_DISTANCE = 8

CALIBRATION_FILE = "laser_calibration_9points.txt"


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

laser_class_id = None

for class_id, name in model.names.items():
    if str(name).lower() == "laser":
        laser_class_id = int(class_id)
        break

if laser_class_id is None:
    laser_class_id = 0

print("Laser class ID:", laser_class_id)


# ============================================================
# SCREEN SIZE
# ============================================================

try:
    import pyautogui

    screen_width, screen_height = pyautogui.size()

except Exception:

    screen_width = 1920
    screen_height = 1080


print(
    "Screen size:",
    screen_width,
    "x",
    screen_height
)


# ============================================================
# EXTRACT 64x64 CROP
# ============================================================

def extract_crop(frame, center):

    cx, cy = center

    half = CROP_SIZE // 2

    x0 = int(cx - half)
    y0 = int(cy - half)

    x1 = x0 + CROP_SIZE
    y1 = y0 + CROP_SIZE

    h, w = frame.shape[:2]

    # Keep crop inside frame
    x0 = max(0, min(x0, w - CROP_SIZE))
    y0 = max(0, min(y0, h - CROP_SIZE))

    x1 = x0 + CROP_SIZE
    y1 = y0 + CROP_SIZE

    return frame[y0:y1, x0:x1].copy()


# ============================================================
# YOLO CLASSIFICATION
# ============================================================

def classify_candidate(crop):

    results = model(
        crop,
        verbose=False
    )

    result = results[0]

    if result.probs is None:
        return 0.0, 1.0

    probabilities = (
        result.probs.data
        .cpu()
        .numpy()
    )

    laser_prob = float(
        probabilities[laser_class_id]
    )

    not_laser_prob = 1.0 - laser_prob

    return laser_prob, not_laser_prob


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    raise SystemExit


# ============================================================
# CAPTURE BACKGROUND
# ============================================================

print()
print("Capturing background...")
print("Keep the laser OFF.")

time.sleep(2)

background_frames = []

for i in range(30):

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    background_frames.append(gray)

    cv2.putText(
        frame,
        f"Capturing background {i + 1}/30",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "9 Point Laser Calibration",
        frame
    )

    cv2.waitKey(1)


if not background_frames:

    print("ERROR: Background capture failed.")

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


background_gray = np.median(
    np.array(background_frames),
    axis=0
).astype(np.uint8)


print("Background captured.")


# ============================================================
# GET CAMERA RESOLUTION
# ============================================================

ret, frame = cap.read()

if not ret:

    print("ERROR: Could not read camera frame.")

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


frame_h, frame_w = frame.shape[:2]

print()
print(
    "Camera resolution:",
    frame_w,
    "x",
    frame_h
)


# ============================================================
# 9 CAMERA TARGETS
# ============================================================

# These positions are INSIDE the camera frame.
#
# For 640 x 480:
#
# (96,72)     (320,72)     (544,72)
#
# (96,240)    (320,240)    (544,240)
#
# (96,408)    (320,408)    (544,408)

camera_targets = [

    (
        int(0.15 * frame_w),
        int(0.15 * frame_h)
    ),

    (
        int(0.50 * frame_w),
        int(0.15 * frame_h)
    ),

    (
        int(0.85 * frame_w),
        int(0.15 * frame_h)
    ),

    (
        int(0.15 * frame_w),
        int(0.50 * frame_h)
    ),

    (
        int(0.50 * frame_w),
        int(0.50 * frame_h)
    ),

    (
        int(0.85 * frame_w),
        int(0.50 * frame_h)
    ),

    (
        int(0.15 * frame_w),
        int(0.85 * frame_h)
    ),

    (
        int(0.50 * frame_w),
        int(0.85 * frame_h)
    ),

    (
        int(0.85 * frame_w),
        int(0.85 * frame_h)
    )
]


# ============================================================
# 9 SCREEN TARGETS
# ============================================================

MARGIN = 80

sx0 = MARGIN
sx1 = screen_width // 2
sx2 = screen_width - MARGIN

sy0 = MARGIN
sy1 = screen_height // 2
sy2 = screen_height - MARGIN


screen_targets = [

    (sx0, sy0),
    (sx1, sy0),
    (sx2, sy0),

    (sx0, sy1),
    (sx1, sy1),
    (sx2, sy1),

    (sx0, sy2),
    (sx1, sy2),
    (sx2, sy2)
]


# ============================================================
# CALIBRATION STORAGE
# ============================================================

camera_points = []
screen_points = []


# ============================================================
# CALIBRATION LOOP
# ============================================================

for target_index in range(9):

    target_x, target_y = camera_targets[target_index]

    screen_x, screen_y = screen_targets[target_index]

    print()
    print("==============================================")
    print(
        f"TARGET {target_index + 1} / 9"
    )
    print(
        f"Camera target: {target_x}, {target_y}"
    )
    print(
        f"Screen destination: {screen_x}, {screen_y}"
    )
    print("Move laser to BLUE circle.")
    print("Wait for LASER STABLE.")
    print("Press SPACE.")
    print("==============================================")


    stable_points = []

    saved = False


    while not saved:

        ret, frame = cap.read()

        if not ret:
            continue


        # ====================================================
        # SAME DETECTION PIPELINE
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


        color_mask = aw.build_color_mask(hsv)


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


        # CORRECT ORDER:
        # image, operation, kernel

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


        # ====================================================
        # FIND CANDIDATES
        # ====================================================

        candidates = aw.find_candidates(
            combined,
            gray,
            diff,
            frame
        )


        # ====================================================
        # YOLO VERIFY
        # ====================================================

        verified = []


        for candidate in candidates:

            center = candidate["center"]

            crop = extract_crop(
                frame,
                center
            )


            if crop.shape[0] != CROP_SIZE:
                continue

            if crop.shape[1] != CROP_SIZE:
                continue


            laser_prob, not_laser_prob = (
                classify_candidate(crop)
            )


            if laser_prob < MIN_YOLO_LASER_PROB:
                continue


            final_score = (
                YOLO_WEIGHT * laser_prob
                +
                OPENCV_WEIGHT * candidate["score"]
            )


            candidate["laser_prob"] = laser_prob

            candidate["not_laser_prob"] = (
                not_laser_prob
            )

            candidate["final_score"] = (
                final_score
            )


            verified.append(candidate)


        # ====================================================
        # SELECT BEST LASER
        # ====================================================

        selected = None


        if verified:

            verified.sort(
                key=lambda x: x["final_score"],
                reverse=True
            )

            selected = verified[0]


        # ====================================================
        # DISPLAY
        # ====================================================

        display = frame.copy()


        # BLUE TARGET

        cv2.circle(
            display,
            (target_x, target_y),
            30,
            (255, 0, 0),
            3
        )

        cv2.circle(
            display,
            (target_x, target_y),
            5,
            (255, 0, 0),
            -1
        )


        cv2.putText(
            display,
            f"TARGET {target_index + 1}/9",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            "Move LASER to BLUE circle",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # LASER FOUND
        # ====================================================

        if selected is not None:

            lx, ly = selected["center"]

            lx = int(lx)
            ly = int(ly)


            # Laser marker

            cv2.circle(
                display,
                (lx, ly),
                10,
                (0, 255, 0),
                2
            )

            cv2.circle(
                display,
                (lx, ly),
                3,
                (0, 0, 255),
                -1
            )


            # Distance from target

            distance = np.sqrt(
                (lx - target_x) ** 2
                +
                (ly - target_y) ** 2
            )


            # =================================================
            # STABILITY
            # =================================================

            if not stable_points:

                stable_points.append(
                    (lx, ly)
                )

            else:

                last_x, last_y = (
                    stable_points[-1]
                )

                movement = np.sqrt(
                    (lx - last_x) ** 2
                    +
                    (ly - last_y) ** 2
                )


                if movement <= STABILITY_DISTANCE:

                    stable_points.append(
                        (lx, ly)
                    )

                else:

                    stable_points = [
                        (lx, ly)
                    ]


            if len(stable_points) >= STABILITY_FRAMES:

                cv2.putText(
                    display,
                    "LASER STABLE - PRESS SPACE",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    display,
                    f"STABILIZING "
                    f"{len(stable_points)}/"
                    f"{STABILITY_FRAMES}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


            cv2.putText(
                display,
                f"Laser: ({lx}, {ly})",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                display,
                f"Target: ({target_x}, {target_y})",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                display,
                f"Distance: {distance:.1f}",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


        else:

            stable_points = []

            cv2.putText(
                display,
                "LASER NOT DETECTED",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "9 Point Laser Calibration",
            display
        )


        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # SPACE = SAVE
        # ====================================================

        if key == ord(' '):

            if (
                selected is not None
                and
                len(stable_points) >= STABILITY_FRAMES
            ):

                avg_x = int(
                    np.mean(
                        [p[0] for p in stable_points]
                    )
                )

                avg_y = int(
                    np.mean(
                        [p[1] for p in stable_points]
                    )
                )


                camera_points.append(
                    (avg_x, avg_y)
                )

                screen_points.append(
                    (screen_x, screen_y)
                )


                print()
                print(
                    f"POINT {target_index + 1} SAVED"
                )

                print(
                    f"Camera: {avg_x}, {avg_y}"
                )

                print(
                    f"Screen: {screen_x}, {screen_y}"
                )


                saved = True


        # ====================================================
        # Q = QUIT
        # ====================================================

        elif key == ord('q'):

            print("Calibration cancelled.")

            cap.release()
            cv2.destroyAllWindows()

            raise SystemExit


# ============================================================
# SAVE 9 POINTS
# ============================================================

print()
print("==============================================")
print("9-POINT CALIBRATION COMPLETE")
print("==============================================")


with open(
    CALIBRATION_FILE,
    "w"
) as f:

    for i in range(9):

        cx, cy = camera_points[i]

        sx, sy = screen_points[i]

        f.write(
            f"{cx},{cy},{sx},{sy}\n"
        )


print()
print(
    "Saved:",
    CALIBRATION_FILE
)


# ============================================================
# PRINT POINTS
# ============================================================

print()
print("Calibration points:")

for i in range(9):

    print(
        f"{i + 1}. "
        f"Camera {camera_points[i]} "
        f"-> "
        f"Screen {screen_points[i]}"
    )


# ============================================================
# CALCULATE HOMOGRAPHY
# ============================================================

camera_array = np.array(
    camera_points,
    dtype=np.float32
)

screen_array = np.array(
    screen_points,
    dtype=np.float32
)


H, status = cv2.findHomography(
    camera_array,
    screen_array,
    method=0
)


if H is None:

    print()
    print("ERROR: Homography calculation failed.")

else:

    print()
    print("Homography calculated successfully.")

    np.savetxt(
        "laser_homography.txt",
        H
    )

    print(
        "Saved: laser_homography.txt"
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Calibration finished successfully.")
print()
print("Next step: use the 9-point homography")
print("for accurate cursor mapping.")

