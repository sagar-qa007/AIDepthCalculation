# 📷 Real-Time Depth Estimation using MiDaS and OpenCV

This project demonstrates real-time depth estimation using a webcam feed, leveraging [Intel's MiDaS](https://github.com/intel-isl/MiDaS) model for monocular depth prediction. It also highlights the closest object in view and displays frame rate (FPS) for performance monitoring.

---

## 🔍 Features

- Live webcam stream with depth estimation
- Visualization of depth map using OpenCV
- Identifies and highlights the nearest object in view
- Displays frames-per-second (FPS) on screen
- Supports CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon)
- Includes functionality to:
  - Calculate relative distance of the closest object
  - Calculate distance between two user-selected points/objects

---

## 📦 Requirements

### 🔧 Setup Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ For Apple Silicon (M1/M2), ensure PyTorch with MPS support is installed:
```bash
pip install torch torchvision torchaudio
```

---

## 🚀 How It Works

### `MainDepthHeatMap.py`
1. **Load MiDaS Model** – Uses `torch.hub` to load a pre-trained depth estimation model.
2. **Initialize Webcam** – Opens webcam for real-time video input.
3. **Process Frames** – Performs depth estimation for each frame:
   - Applies model-specific transforms
   - Computes and normalizes depth map
   - Highlights closest object
   - Overlays FPS and distance info
4. **Display** – Shows both RGB feed and depth heatmap.

### `CalculateDistanceFromCamera.py`
- Computes relative depth (distance) from the camera to the nearest object.
- Helpful for real-time safety systems or interaction triggers.

### `CalculateDistanceBetweenObject.py`
- Allows user to click on two points in the frame.
- Calculates the relative depth difference (approximate distance) between two objects.

---

## 🧠 Supported MiDaS Models

You can modify the model by changing the `MODEL_VARIANT` in the script:

- `DPT_Large` – High accuracy, slower
- `DPT_Hybrid` – Balanced performance (default)
- `MiDaS_small` – Lightweight and fast

---

## ⚙️ Running the App

Run any of the Python scripts:

```bash
python MainDepthHeatMap.py
python CalculateDistanceFromCamera.py
python CalculateDistanceBetweenObject.py
```

Press `ESC` to exit webcam view.

---

## 💻 Device Support

The script will automatically choose the best available device:

- `CUDA` if available (NVIDIA GPU)
- `MPS` if on macOS with Apple Silicon
- Fallback to `CPU`

---

## 🧪 Notes

- Depth values are **relative**, not absolute (i.e., closer/farther but not in real-world units).
- Real-time performance depends on the selected model and your hardware.

---

## 👨‍💻 Author

Made with ❤️ by Sagar Khalasi