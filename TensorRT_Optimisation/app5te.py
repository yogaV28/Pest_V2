import cv2
import imutils
from yoloDet import YoloTRT

# Use path for library and engine file
model = YoloTRT(library="yolov8/build/libmyplugins.so", engine="yolov8/build/yolov8-tiny.engine", conf=0.5, yolo_ver="v7")

# Open the CSI camera (you may need to adjust the index)
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t = model.Inference(frame)

    cv2.imshow("Output", frame)

    # Press 'q' to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

