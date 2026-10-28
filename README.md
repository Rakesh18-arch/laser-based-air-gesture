# Air Writer Smart Board — integrated build

## What is included
- `air_writer_app.py`: desktop home screen, strict DroidCam connection, live preview, background capture, 9-point calibration wizard, and Smart Board.
- `air_writer.py`: the supplied laser spot candidate detector; reused by the app.
- `air_writer_yolo.py`, `click_operations.py`, `cursor_reference.py`, `drawing_reference.py`, `laser_calibration.py`: original project modules retained for reference/advanced workflows.
- Calibration data files from the supplied project.

## Start
1. Install Python 3.12 (64-bit) if not already installed.
2. On the phone, open DroidCam. On Windows, open the DroidCam client and connect the phone.
3. Run `INSTALL_AND_RUN.bat`, or run `python -m pip install -r requirements.txt` and then `python air_writer_app.py`.
4. Click **Refresh devices**. Select a device whose Windows name contains **DroidCam**, then click **Connect**.
5. Confirm the live preview is from the phone. The app intentionally will not switch to the HP built-in camera.
6. With the laser OFF, click **Capture background**.
7. Run **9-point screen calibration**. Point the laser at each full-screen target, wait for a green verified spot, and press Space. Press Enter after all nine targets to save.
8. Open Smart Board. Select Cursor, Drag, Pencil, Eraser, Undo, Redo, New page, Save PNG/PostScript, colors, and size from the toolbar. Draw a circle with the laser to open the radial menu; hover on an option for about 0.75 s to select it.

## Important limitations / reliability notes
- Camera connection is strict: device enumeration uses Windows DirectShow names via `pygrabber`. If the app cannot enumerate the device, it reports that problem instead of guessing an index and opening the laptop camera.
- OpenCV backend retries (DirectShow, Media Foundation, automatic) are attempted for the selected device only.
- The detector is the supplied OpenCV candidate pipeline. No detector can guarantee zero false positives in all lighting conditions; check the green candidate marker before enabling any system control.
- The app defaults to mouse control disabled. Enable it only after verifying calibration. Mouse control requires `pyautogui` and Windows camera permissions.
- Calibration targets are recorded in camera coordinates and screen positions. Current app uses a global homography for mapping; the supplied project also contains a piecewise 9-point mapper in `cursor_reference.py` for future integration where a global transform is not accurate enough.
- Board Save uses Tk Canvas PostScript (`.ps`) so no extra image conversion package is needed. For a clean PNG, use a screen capture or export enhancement in a later build.
- The supplied YOLO weights file (`best.pt`) was not included among the uploaded files. The original YOLO modules are retained, but this integrated UI does not load/retrain the model automatically.

## DroidCam troubleshooting
- Phone and PC client must both be open and connected; allow camera permission in Windows Settings > Privacy & security > Camera.
- If Wi-Fi discovery fails, add the phone IP shown in the phone app manually in the DroidCam client. USB is generally more stable.
- If there is a blank/green feed, exit and relaunch the Windows client. Close other apps that may hold the camera, then refresh devices.
- For unstable video, try 1280×720 at 30 FPS or USB.

## Research references used for connection design
- Official DroidCam Windows setup: https://droidcam.app/windows/
- Official DroidCam troubleshooting and USB/Wi-Fi guidance: https://droidcam.app/help/
- OpenCV Video I/O backends: https://docs.opencv.org/4.x/d0/da7/videoio_overview.html
- Microsoft camera device/background: https://learn.microsoft.com/en-us/windows/win32/directshow/about-video-capture-devices
- Related air-writing / virtual whiteboard patterns: https://github.com/amiraasalah/virtual-whiteboard-air-writing and https://github.com/ritwik-999/virtual-drawing-board
