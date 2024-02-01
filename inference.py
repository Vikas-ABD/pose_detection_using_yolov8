from ultralytics import YOLO
import cv2
import numpy as np

model_path = r'runs\runs\pose\train2\weights\best.pt'
image_path = r'test.jpg'

img = cv2.imread(image_path)
model = YOLO(model_path)

# Run inference on the image
result = model(image_path)[0]

# Get the keypoints tensor
keypoints_tensor = result.keypoints

# Convert Keypoints to NumPy array and then to list
keypoints_list = keypoints_tensor.xy

# Iterate through keypoints and draw them on the image
for person_idx, person_keypoints in enumerate(keypoints_list):
    for keypoint_idx, keypoint in enumerate(person_keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.putText(img, f'Keypoint {keypoint_idx + 1}', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

# Display the image
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
