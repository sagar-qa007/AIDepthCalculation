import torch
import torchvision.transforms as T
import cv2
import numpy as np

# Load MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load YOLOv5s model for object detection
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo.to("cpu").eval()

cap = cv2.VideoCapture(0)

# Define approximate focal length (calibrate if needed)
FOCAL_LENGTH = 500  # adjust based on your camera calibration

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_batch = transform(frame).to("cpu")

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_MAGMA)

    # Run YOLOv5 detection
    results = yolo(frame)
    boxes = results.xyxy[0].numpy()

    # Pick first 2 detected objects
    if len(boxes) >= 2:
        obj1 = boxes[0]
        obj2 = boxes[1]

        x1, y1, x2, y2 = map(int, obj1[:4])
        cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2

        x3, y3, x4, y4 = map(int, obj2[:4])
        cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2

        # Draw bounding boxes and centroids
        cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(depth_colored, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(depth_colored, (cx1, cy1), 5, (0, 255, 0), -1)
        cv2.circle(depth_colored, (cx2, cy2), 5, (0, 255, 0), -1)

        # Draw line between objects
        cv2.line(depth_colored, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

        # Estimate relative depth from depth map
        z1 = depth_map[cy1, cx1]
        z2 = depth_map[cy2, cx2]
        avg_depth = (z1 + z2) / 2

        # Compute pixel distance between centroids
        pixel_distance = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

        # Estimate real-world distance (in inches) using focal length
        distance_in_inches = (pixel_distance * avg_depth) / FOCAL_LENGTH

        cv2.putText(depth_colored, f"Distance: {distance_in_inches:.2f} in", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Depth with Distance", depth_colored)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()