import cv2
import torch
import time
import numpy as np

# ─────────────────────────────────────────────────────────────
# Model Configuration
# 📏 Step 1: Calibrate the Camera
# 	1.	Select a Reference Object: Choose an object with a known width (e.g., a credit card, which is typically 3.37 inches wide).
# 	2.	Capture a Reference Image: Place the object at a known distance (e.g., 24 inches) from the camera and capture an image.
# 	3.	Measure the Object’s Width in Pixels: In the captured image, measure the width of the object in pixels.
# 	4.	Calculate the Focal Length:
# \text{Focal Length} = \frac{\text{Pixel Width} \times \text{Known Distance}}{\text{Real Width}}
# For example, if the object appears 200 pixels wide at 24 inches:
# \text{Focal Length} = \frac{200 \times 24}{3.37} \approx 1424.33

# 📐 Step 2: Real-Time Distance Measurement

# With the focal length calculated, you can now measure distances in real-time:
# 	1.	Detect the Object: Use OpenCV to detect the object in each frame.
# 	2.	Measure the Object’s Width in Pixels: Determine the width of the detected object in pixels.
# 	3.	Calculate the Distance:
# \text{Distance} = \frac{\text{Real Width} \times \text{Focal Length}}{\text{Pixel Width}}
# This will give you the distance in inches.
# ─────────────────────────────────────────────────────────────

MODEL_VARIANT = "MiDaS_small"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("[INFO] Loading MiDaS model...")
depth_model = torch.hub.load("intel-isl/MiDaS", MODEL_VARIANT)
depth_model.to(DEVICE)
if DEVICE.type == "cuda":
    depth_model = depth_model.half()
depth_model.eval()
print(f"[INFO] Model '{MODEL_VARIANT}' loaded and moved to {DEVICE}")

print("[INFO] Loading input transforms...")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = (
    midas_transforms.dpt_transform
    if MODEL_VARIANT in ["DPT_Large", "DPT_Hybrid"]
    else midas_transforms.small_transform
)
print("[INFO] Transforms loaded.")

# ─────────────────────────────────────────────────────────────
# Initialize Webcam
# ─────────────────────────────────────────────────────────────

print("[INFO] Initializing webcam...")
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam successfully initialized.")

# ─────────────────────────────────────────────────────────────
# Distance Calibration Constants
# ─────────────────────────────────────────────────────────────
KNOWN_DISTANCE = 24.0  # inches
KNOWN_WIDTH = 3.37     # inches (width of object like credit card)
PIXEL_WIDTH_AT_KNOWN_DISTANCE = 200  # manually measured pixel width during calibration

FOCAL_LENGTH = (PIXEL_WIDTH_AT_KNOWN_DISTANCE * KNOWN_DISTANCE) / KNOWN_WIDTH

def calculate_distance(focal_length, real_width, pixel_width):
    if pixel_width == 0:
        return None
    return (real_width * focal_length) / pixel_width

# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────

frame_count = 0

while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        print("[WARNING] Frame not received. Exiting...")
        break

    frame_count += 1
    start_time = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_frame).to(DEVICE)
    if DEVICE.type == "cuda":
        input_tensor = input_tensor.half()

    with torch.no_grad():
        prediction = depth_model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map_normalized = cv2.normalize(
        depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F
    )

    # Find the closest object's distance (smallest depth value)
    min_depth = np.min(depth_map)
    closest_location = np.unravel_index(np.argmin(depth_map), depth_map.shape)

    # NEW: Basic object detection using contours for bounding box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 35, 125)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        pixel_width = max(int(rect[1][0]), int(rect[1][1]))
        distance_in_inches = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, pixel_width)

        if distance_in_inches:
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Distance: {distance_in_inches:.2f} in",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

    # FPS display
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show output
    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()