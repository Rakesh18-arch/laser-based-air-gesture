
import cv2
import sys
import time

backends = [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
]

for backend_name, backend in backends:
    for index in range(5):
        print(f"\nTesting {backend_name}, camera index {index}...")

        cap = None

        try:
            cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                print("Cannot open; skipping.")
                continue

            frame = None
            deadline = time.time() + 3

            while time.time() < deadline:
                ok, image = cap.read()

                if ok and image is not None:
                    frame = image
                    break

                time.sleep(0.1)

            if frame is None:
                print("No video frame; skipping.")
                continue

            window = (
                f"{backend_name} index {index} "
                "| P=phone, N=next, Q=quit"
            )

            print("Preview opened. Inspect the video window.")

            while True:
                ok, image = cap.read()

                if ok and image is not None:
                    frame = image

                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"{backend_name} index {index} | "
                    "P=phone, N=next, Q=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("p"):
                    with open("droidcam_camera.txt", "w") as file:
                        file.write(f"backend={backend_name}\n")
                        file.write(f"index={index}\n")

                    print(
                        f"\nPHONE VIDEO FOUND: "
                        f"backend={backend_name}, index={index}"
                    )
                    print("Saved in droidcam_camera.txt")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

                if key == ord("n"):
                    break

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

        except Exception as error:
            print("Camera test error:", error)

        finally:
            if cap is not None:
                cap.release()

            cv2.destroyAllWindows()

print("\nFinished. No camera was identified as the phone.")
