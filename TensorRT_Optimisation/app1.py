import cv2
import imutils
from yoloDet import YoloTRT

# Use path for library and engine file
model = YoloTRT(library="yolov8/build/libmyplugins.so", engine="JetsonYoloV8-TensorRT\best.engine", conf=0.5, yolo_ver="v8")

# Open the webcam (you can specify the webcam index, usually 0 for the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t = model.Inference(frame)

    cv2.imshow("Output", frame)

    # Press 'q' to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

