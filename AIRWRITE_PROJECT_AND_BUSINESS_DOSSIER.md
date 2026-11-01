# AIRWRITE — Technical & Business Dossier

**Product:** AIRWRITE — Laser-Based Air Gesture Smart Board  
**Project entry point:** `desktop/b.py`  
**Protected detector module:** `desktop/air_writer.py`  
**Website runtime:** Local Python portal + SQLite + browser UI  
**Package type:** One-click Windows launch bundle

## 1. Product concept

AIRWRITE turns a compatible display into a touchless workspace. A Windows laptop runs the computer-vision application, a phone camera connected through DroidCam observes the laser spot, and the desktop application maps the detected laser position to the screen. After calibration, the Smart Board offers cursor interaction, writing, drawing, gestures and editing tools.

AIRWRITE is a local computer-vision / human-computer-interaction product. It is not an IoT system. The physical laser is operated manually; the website animation does not switch on a laser.

## 2. Verified source structure

The release uses `b.py` as the only desktop entry point. It imports `air_writer.py` directly. The detector file in this release was byte-compared with the corresponding file in the uploaded project archive and is identical.

Included desktop modules:

- `b.py` — main desktop UI, camera connection, background capture, calibration, cursor and Smart Board workflows.
- `air_writer.py` — supplied OpenCV laser spot detection pipeline.
- `air_writer_yolo.py` — optional YOLO helper module retained for compatibility.
- `laser_calibration.py` — calibration utility retained from the project.
- `click_operations.py`, `cursor_reference.py`, `drawing_reference.py` — supporting/reference modules retained from the supplied project.
- `droidcam_camera.txt` — saved tested camera profile (`MSMF`, index `0`).
- `laser_calibration_9points.txt` — legacy filename retained because `b.py` reads this exact path. The supplied profile contains five rows; this build uses a fresh five-target calibration before enabling the cursor.
- `drawings/` — supplied drawing image retained as user project material.

Intentionally excluded from the clean release: old `air_writer_app.py` launcher, old camera test scripts, generated `__pycache__`, runtime log, and `laser_homography.txt` (the current `b.py` path does not load that file). This avoids shipping stale alternatives or machine-specific debug output.

## 3. Desktop startup workflow

1. `START_AIRWRITE.bat` checks for Python 3.12 (64-bit); if unavailable it asks the Windows Python Install Manager to install it. If that path cannot be used, it opens the Python downloads page with an explanation.
2. The launcher creates a private `.venv` folder in the package and installs `requirements.txt` once.
3. It starts `server.py`, which opens the browser to `http://127.0.0.1:8765/`.
4. A user registers or signs in.
5. The signed-in user clicks **Launch AIRWRITE Desktop**. The local portal launches `desktop/b.py` using the same virtual environment.
6. `b.py` uses the saved DroidCam profile and auto-connects to the profile. It captures a fresh laser-OFF background after warm-up, then shows the five calibration targets.
7. A fresh and stable laser spot is required to capture each target. After successful mapping, the desktop app automatically opens the Smart Board.

The browser button launches the real Python desktop process; it does not implement a fake browser-only camera demo. The laser graphics on the website are decorative interface animations and clearly labelled simulated readouts.

## 4. Account database and owner access

The local portal uses the Python standard-library `sqlite3` database at:

`data/airwrite_users.sqlite3`

Tables:

- `users` — registered user name, email, organisation, role, creation time, last login and login count.
- `pilot_enquiries` — submitted pilot-enquiry details linked to the signed-in user.
- `activity_log` — registration, login and failed-login event timestamps.

Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes, not plain text. Session cookies are HttpOnly and SameSite=Lax. The website binds only to `127.0.0.1`, so it is accessible on the same PC and is not a public hosted registration server.

On first run, the portal creates a private Owner account. Its generated password is printed once in the launcher window and saved in `data/OWNER_LOGIN.txt`. The user signs in with username `owner`; the file contains the password. The Owner Dashboard can list registrations and pilot enquiries and export the registrations as CSV. Normal users cannot access those endpoints.

Keep the same extracted package folder to preserve the database. Back up the `data` folder. Extracting a new copy creates a separate database. Do not share `OWNER_LOGIN.txt` or the SQLite database publicly.

## 5. Website interface

The website includes:

- animated laser intro before login / registration;
- login and account registration forms with validation;
- authenticated workspace and desktop-launch button;
- local portal, camera-profile, calibration-profile and desktop status indicators;
- capability sections for camera, laser tracking, calibration, gestures, Smart Board and Smart Write;
- proposed Starter, Classroom and Campus deployment models;
- pilot-enquiry form with required business-use fields;
- owner-only dashboard for registrations and enquiries;
- CSV export of registrations.

Pricing is intentionally quote-based at this prototype stage; no unsupported market price is invented. The local pilot-enquiry form records data on this computer.

## 6. Business model hypothesis

Potential customer segments:

- schools, colleges and coaching centres;
- trainers and presentation / meeting spaces;
- accessibility and human-computer-interaction evaluation groups;
- technology integration partners.

Potential product packages:

- **AIRWRITE Starter:** single-user evaluation on one Windows system.
- **AIRWRITE Classroom:** one-room pilot, setup guidance and educator onboarding.
- **AIRWRITE Campus:** multi-room discovery, deployment planning, training and ongoing support.

Potential revenue streams: software licensing, hardware/accessory kits, installation and onboarding, institutional deployment and annual support/updates. These are hypotheses to validate through customer interviews and pilots; they are not yet proven revenue.

## 7. Dependencies and known limitations

The one-click package is intended for Windows 10/11 and Python 3.12 64-bit. `requirements.txt` installs NumPy, OpenCV, Pillow, PyAutoGUI, pygrabber, comtypes and the pytesseract Python wrapper.

Known items not included in the uploaded archive:

- `best.pt` — the trained YOLO weight file was absent. The supplied OpenCV detector remains the primary detector; optional YOLO verification cannot load without its trained weights.
- Tesseract OCR Windows executable — `pytesseract` is a Python wrapper only. Install the Tesseract Windows application separately to enable the OCR functions; see `OCR_SETUP.txt`.

A web-only browser cannot directly use the phone's DroidCam or run a local Python process. Therefore this package intentionally runs a localhost bridge and starts the actual desktop app on the same PC. A public multi-customer website with remote registrations requires a separately hosted backend, proper account security, privacy policy, backup and deployment setup; this local SQLite version is for local evaluation and demos.

## 8. Validation performed for this package

- Python syntax compilation for the local portal and desktop Python modules.
- HTML parser validation and checks that all literal JavaScript element-ID references exist in the page.
- Static checks that website anchor links resolve.
- Local HTTP tests for the status endpoint, 5-point profile readout, owner authentication, account registration, pilot-enquiry storage, Owner Dashboard data, CSV export and role-based access denial.
- Archive integrity checks after packaging.
- Byte comparison confirming `air_writer.py` matches the uploaded project detector.

Not physically validated in this environment: DroidCam hardware access, laser spot behaviour in the user's room lighting, Windows pointer control, calibration precision on the actual display, and OCR executable availability. Those require testing on the user's Windows laptop with the phone camera and laser hardware connected. No software can promise zero errors under all hardware, lighting and operating conditions.

## 9. First-run instructions

1. Extract the ZIP once.
2. Double-click `START_AIRWRITE.bat` at the package root.
3. Keep the console window open.
4. Register a normal account or sign in. The first launch also creates a private local Owner account; its credentials are in `data/OWNER_LOGIN.txt` and are used only for the Owner Dashboard.
5. Click **Launch AIRWRITE Desktop**. The real desktop entry is `desktop/b.py`.
6. Select and visually confirm the built-in laptop camera, external USB webcam or DroidCam. For a USB webcam, plug it in before opening Camera Studio.
7. Keep the laser OFF during warm-up/background capture, then follow the five full-screen laser calibration targets.
8. The full-screen Smart Board opens automatically. For hand-only operation, return to the portal and click **Start Hand Gesture Studio**. It opens the selected camera, maps five fingertip targets separately, and opens the full-screen Board in Pen mode.

The same folder must remain in place so that the `.venv`, database, camera profile and calibration data remain together.

## 10. Commercial-readiness checklist

Before charging customers or distributing this as a finished commercial product:

- Run repeated field tests on the exact supported Windows versions, DroidCam versions, camera placements, projector/monitor types and lighting conditions.
- Measure calibration repeatability, pointer jitter, false detections, latency, and click/menu reliability; define acceptable limits and a recovery path.
- Complete a laser-safety review for the exact diode/module and intended use, including warnings, safe operating instructions and any local product requirements.
- Confirm ownership and permission to commercially distribute every source file; the supplied project ZIP did not include a top-level project licence or contributor/third-party rights statement.
- Review and include required notices for each distributed third-party Python package and runtime component.
- Publish a clear privacy notice, data retention policy, account deletion method, backup/restore process and support contact before collecting real customer details.
- For public remote registrations, replace the local SQLite/localhost service with a properly hosted backend, TLS, production authentication and protected database backups. Do not expose this local development server directly to the internet.
- Pilot with a small number of classrooms first, capture measured outcomes and support issues, then finalise the hardware bundle and quote-based pricing.

The current bundle is a local prototype / demonstration release, not a certified or commercially validated product.

## AIRWRITE interaction workflow update (v8)

### Laser Writer — one-second stable target capture
- Laser Writer remains on its separate existing camera profile (DroidCam or the confirmed USB camera); Hand Gesture Studio does not overwrite it.
- Laser warm-up and the 30-frame background capture occur before the target sequence.
- Each of the five full-screen calibration targets requires one continuous second of fresh, stable laser detections (minimum 12 samples). If the detected point jumps, becomes stale, or exceeds the configured stability range, the one-second progress restarts. The next target appears only after the current point is accepted.
- A short 0.16-second transition is shown after a target is accepted. Camera warm-up/background capture and final mapping validation are additional time beyond the one-second target dwell.

### Hand Gesture Studio — no spatial calibration
- Hand mode does not load, prompt for or save a hand-to-screen calibration/homography. It uses normalized camera coordinates to map the index fingertip directly into the left-side Smart Board canvas.
- Hand mode automatically prefers the built-in laptop webcam. It has a dedicated hand camera profile separate from the laser camera profile. If Windows cannot confidently match the integrated camera using DirectShow, the app opens a live preview selector instead of silently selecting DroidCam or the wrong camera.
- The live camera preview is visible on the right side during the Gesture Academy and during the Smart Board session. The Smart Board drawing canvas is on the left.
- The first-use Gesture Academy asks the user to show one hand naturally, with no palm-open, palm-closed or palm-facing-camera gate. On stable detection it stores local setup metadata (camera name/type and handedness hint) only; it does not store photos, video or raw landmark frames.
- The Academy then teaches, one step at a time: index pointer; thumb-index pinch for draw/select; index-middle V-sign for pan; thumb-middle pinch for zoom; and three raised fingers for the tools menu. Once each pose is recognised for the short lesson hold, the next instruction is shown. After all lessons, the full Smart Board opens automatically.
- MediaPipe Hands supplies 21 landmarks. Only the index fingertip controls the pointer; other fingertips are used only to classify gestures. No second fingertip can independently move the pointer.
- PIP joint angles are used for finger extension so the app is less dependent on a finger pointing vertically in the image. Pinch/zoom have hysteresis. Pan and menu gestures use a 3-of-5 frame consensus to reduce one-frame false activations. Pointer movement uses a One Euro filter to reduce jitter.
- Gestures: index = pointer; thumb-index pinch = draw/select; V-sign = pan; thumb-middle pinch = zoom; index-middle-ring raised with pinky folded = open tools menu; open palm is not required as a status gate.
- Camera frames are processed locally. The local hand profile stores non-image setup metadata only.

### Review of the four reference repositories

These repositories were reviewed for design and algorithmic ideas; their code was not copied into AIRWRITE:

1. [yyyerica/HandGestureClassify](https://github.com/yyyerica/HandGestureClassify) demonstrates 21-joint hand pose estimation, a hand tracker and optional Kalman filtering. It relies on an older TensorFlow model stack and external weights. AIRWRITE uses MediaPipe's 21 landmarks and its own pointer/gesture logic instead of importing those legacy model dependencies.
2. [guiyang882/HandGesture](https://github.com/guiyang882/HandGesture) is an older C++/OpenCV image-processing project built around OpenCV 3.2, Qt 5.6 and CMake. Its segmentation/background-processing concepts were informative, but its stack is not a drop-in module for the current Python/Tkinter runtime.
3. [eaglesky/HandGestureApp](https://github.com/eaglesky/HandGestureApp) is an Android application using OpenCV for Android, the NDK and an SVM workflow with presampling. Its staged gesture sampling and confirmation ideas are useful, but it is not a Windows webcam module. Its repository declares the MIT License; no code from it was copied.
4. [whatifif/handgesture](https://github.com/whatifif/handgesture) and its companion [whatifif/handgesturecode](https://github.com/whatifif/handgesturecode) discuss controlling a computer with hand gestures and avoiding unstable single-frame predictions. AIRWRITE adapts the temporal-consensus principle for pan/menu poses. The companion code repository declares the MIT License; no code or model was copied.

The first two repositories use older stacks or do not expose a clear root-level licence in the inspected files. They are references only, not redistributed components. AIRWRITE's `hand_gesture_engine.py` is an original implementation using MediaPipe landmarks and geometric rules.

### Validation performed for v8
- Python syntax compilation for the local portal, desktop application, hand gesture engine and regression tests.
- Twelve hardware-independent synthetic tests covering index-only pointing, thumb-index pinch without an open-palm requirement, V-sign pan, three-finger menu, thumb-middle zoom pointer mapping, rotated-hand finger extension, stable primary-hand selection, 3-of-5 temporal consensus, pointer jitter smoothing, normalized canvas center mapping, mirrored camera edges and canvas-bound clamping.
- The existing `desktop/air_writer.py` laser detector is retained unchanged.
- Laser calibration dwell, no-hand-calibration flow and the split-view tutorial/board are statically checked; live operation still requires testing on the user's Windows laptop.

### Hardware validation still required
No physical webcam, laser or hand was available in this build environment. Therefore actual USB/laptop camera drivers, live MediaPipe detection under the user's room lighting, laser target capture repeatability, hand gesture false-positive rate, pointer jitter, click/menu reliability and end-to-end cursor behaviour have **not** been physically verified. The 1-second laser dwell is a software acceptance threshold, not a guarantee of one-second end-to-end calibration. These checks must be performed on the target Windows laptop before commercial deployment. The optional YOLO `best.pt` weights were not present in the supplied archive, and Tesseract remains a separate optional Windows installation.

The current bundle remains a local prototype / demonstration release, not a certified or commercially validated product.
