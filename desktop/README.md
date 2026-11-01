# AIRWRITE Desktop Studio — v8

The desktop entry point is **`b.py`**. It imports the preserved laser detector from **`air_writer.py`**.

## Camera profiles

- Laser mode uses its independent `camera_profile.txt` / legacy `droidcam_camera.txt` profile, retaining the existing DroidCam setup.
- Hand mode has its own `hand_camera_profile.txt` profile and automatically prefers the built-in laptop camera.
- External USB UVC webcams are supported. If the camera name is ambiguous, use the live preview selector; AIRWRITE will not silently switch devices.

## Laser workflow

1. Confirm the selected live camera.
2. Keep the laser OFF during the warm-up and 30-frame background capture.
3. Complete the five full-screen laser targets. Each target requires **one continuous second** of fresh, stable laser detection; if the spot moves or is lost, the stability timer restarts.
4. After all five points are accepted and the mapping passes validation, the full-screen Smart Board opens automatically.

The existing laser candidate detector in `air_writer.py` is preserved.

## Hand Gesture Studio — no spatial calibration

Hand mode intentionally has **no point-calibration wizard**. It opens the built-in laptop camera by preference and starts the Gesture Academy. The Academy recognises one visible hand in any natural pose, stores only non-image setup metadata locally, then teaches these poses in order: index pointer, thumb-index pinch, V-sign pan, thumb-middle zoom and three-finger menu. Each lesson advances only after the pose is recognised steadily.

After the lessons, the Smart Board opens in a split layout: drawing canvas on the left; mirrored live hand camera, current gesture status and quick guide on the right. The index fingertip is the only pointer. The hand point is normalized from the camera frame directly into the board canvas; no hand homography file is read or written.

- Index finger: move pointer
- Thumb + index pinch: draw / click / select
- Index + middle V-sign: pan (temporal consensus)
- Thumb + middle pinch: zoom
- Index + middle + ring raised: open tools menu
- No open-palm / closed-palm gate is required

MediaPipe Hands processes one primary hand using 21 landmarks locally. The camera stream is not sent to the website or a remote service. Room lighting, hand distance and camera quality affect performance, so actual physical testing remains necessary.

## Retained Smart Board functions

Cursor, drag, pen, Smart Write, swipe typing, eraser, shapes, scroll, color, brush size, pages, undo/redo, save PNG, export PDF and clear page remain accessible through the toolbar / radial tools menu.

## Known limits

- Optional YOLO `best.pt` was not in the supplied ZIP; OpenCV laser candidate detection remains available.
- Tesseract OCR requires a separate Windows installation for OCR functions.
- Actual camera, laser calibration, gesture accuracy, click reliability and drawing quality must be tested on the target laptop before commercial deployment.
