import sys
import cv2
import imutils
from yoloDet import YoloTRT

# Assuming you have functions to calculate precision and recall
def calculate_precision_recall(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall

# use path for library and engine file
model = YoloTRT(library="yolov8/build/libmyplugins.so", engine="JetsonYoloV8-TensorRT\best.engine", conf=0.5, yolo_ver="v8")

cap = cv2.VideoCapture("videos/testvideo1.mp4")

true_positives = 0
false_positives = 0
false_negatives = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)

    # Perform inference
    detections, t = model.Inference(frame)

    # Assuming you have ground truth bounding boxes (replace with your actual ground truth)
    ground_truth_boxes = [...]

    # Match predicted boxes with ground truth
    for obj in detections:
        # obj['box'] contains the bounding box coordinates (x, y, width, height)
        # Implement your logic to check for matches with ground truth boxes
        match_found = False  # Placeholder for your matching logic
        if match_found:
            true_positives += 1
        else:
            false_positives += 1

    # Count false negatives (missed ground truth boxes)
    false_negatives += len(ground_truth_boxes) - true_positives

    # Display the output
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Calculate precision and recall
precision, recall = calculate_precision_recall(true_positives, false_positives, false_negatives)
print("Precision: {:.2f}, Recall: {:.2f}".format(precision, recall))

cap.release()
cv2.destroyAllWindows()

