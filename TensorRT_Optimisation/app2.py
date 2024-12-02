import cv2
import imutils
from yoloDet import YoloTRT
import os

# Use path for library and engine file
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/yolov7-tiny.engine", conf=0.5, yolo_ver="v7")

# Path to the test images folder
test_folder = "Test/images/"

# Create a result folder if it doesn't exist
result_folder = "result"
os.makedirs(result_folder, exist_ok=True)

# Get the list of image files in the test folder
image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Read the image from the test folder
    image_path = os.path.join(test_folder, image_file)
    frame = cv2.imread(image_path)

    # Resize the frame
    frame = imutils.resize(frame, width=600)

    # Perform object detection
    detections, t = model.Inference(frame)

    # Draw bounding boxes on the frame
    for detection in detections:
        # Extract information from detection (e.g., bounding box coordinates)
        x, y, w, h, confidence, class_id = detection

        # Draw bounding box on the frame
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Save the result image to the result folder
    result_path = os.path.join(result_folder, f"result_{image_file}")
    cv2.imwrite(result_path, frame)

    # Display the result (optional)
    cv2.imshow("Output", frame)
    cv2.waitKey(0)  # Wait for a key press before moving to the next image

# Destroy any OpenCV windows
cv2.destroyAllWindows()

