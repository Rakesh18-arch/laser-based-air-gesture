
import cv2

camera = cv2.VideoCapture(2, cv2.CAP_MSMF)

print("Camera opened:", camera.isOpened())

if camera.isOpened():
    while True:
        success, frame = camera.read()

        if not success:
            print("No video frame received")
            break

        cv2.imshow("Camera Test - Press Q to close", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
else:
    print("Camera could not open")

camera.release()
cv2.destroyAllWindows()
