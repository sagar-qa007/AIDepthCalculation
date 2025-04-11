import cv2
import torch
import time
import numpy as np

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────

MODEL_VARIANT = "DPT_Hybrid"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not webcam.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam successfully initialized.")

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

    # Highlight the closest point in the frame
    cv2.circle(frame, (closest_location[1], closest_location[0]), 10, (0, 0, 255), 2)

    elapsed = time.time() - start_time
    fps = 1 / elapsed if elapsed > 0 else 0

    colored_depth = (depth_map_normalized * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_MAGMA)

    # Add FPS and distance info on frame
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Nearest Obj (rel): {min_depth:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sagar - Live Feed", frame)
    cv2.imshow("Sagar - Depth Map", colored_depth)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
print("[INFO] Webcam released and windows closed.")