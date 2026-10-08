# SpatialTrack Architecture & Mathematical Foundations

SpatialTrack is an open-source, CPU-optimized computer vision engine engineered for real-time spatial telemetry, multi-object tracking, and metric speed estimation.

---

## 1. System Pipeline Overview

```
Video Stream (MP4/RTSP)
       │
       ▼
1. Threaded Video Decoder (VideoReader)
       │
       ▼
2. INT8 ONNX Detector (YOLOv10n)
       │
       ▼
3. Two-Stage ByteTrack Tracker (Kalman Filter + Hungarian Algorithm)
       │
       ▼
4. Planar Homography Mapper (Pixel Centroids -> World Meters)
       │
       ▼
5. Analytics Engine (EMA Speed, Zone Monitor, Spatial Heatmap)
       │
       ▼
6. Compositor & Sinks (Dual-Pane BEV Canvas, CSV, JSONL)
```

---

## 2. Component Deep Dive

### Stage 1: Threaded Video Ingestion
To eliminate I/O blocking during inference, `VideoReader` decodes frames in an independent daemon thread buffered by a bounded thread-safe queue. This decouples hardware disk read latency from pipeline computation.

### Stage 2: Quantized INT8 Detection
- **Architecture:** YOLOv10n anchor-free nano model (2.3M parameters).
- **Quantization:** Dynamic INT8 static-weight quantization via ONNX Runtime shrinks model weight footprint from 9.4 MB to 2.65 MB.
- **Latency:** ~70 ms per frame on standard modern laptop CPUs without GPU hardware.
- **Preprocessing:** Vectorized letterbox padding maintaining aspect ratio without memory reallocation.

### Stage 3: Multi-Object Tracking (ByteTrack + 2D Kalman Filter)
State tracking represents each subject as an 8-dimensional state vector:

$$\mathbf{x} = [c_x, c_y, a, h, v_{cx}, v_{cy}, v_a, v_h]^T$$

Where:
- $(c_x, c_y)$ is the 2D bounding box center coordinate.
- $a = w / h$ is aspect ratio, and $h$ is bounding box height.
- $(v_{cx}, v_{cy}, v_a, v_h)$ represent velocity components.

#### Two-Stage Bipartite Association
1. **Stage 1 (High-Confidence):** Matches track predictions with detections above `high_threshold` (0.35) using the Hungarian algorithm on pairwise IoU costs.
2. **Stage 2 (Occlusion Recovery):** Matches remaining unconfirmed tracks with low-confidence detections (`low_threshold` 0.1 to 0.35). This allows the tracker to maintain ID persistence through heavy occlusions or motion blur without requiring expensive Re-ID deep neural networks.

### Stage 4: Planar Homography Projection
Vehicles are physical 3D objects touching the ground plane. Measuring velocity from the bounding box center introduces severe projection errors due to vehicle height.

SpatialTrack anchors vehicle positions at the **bottom-center ground contact point**:

$$u = \frac{x_{min} + x_{max}}{2}, \quad v = y_{max}$$

The ground contact point is transformed to metric world coordinates $(X, Y)$ in meters via the calibrated homography matrix $\mathbf{H}$.

### Stage 5: Velocity Estimation & Smoothing

#### Instantaneous Metric Displacement
Between consecutive frames $t-1$ and $t$ separated by $\Delta t = 1 / \text{FPS}$:

$$\Delta d = \sqrt{(X_t - X_{t-1})^2 + (Y_t - Y_{t-1})^2}$$

#### Micro-Jitter Suppression
Bounding box edge fluctuations produce artificial 1 to 2 pixel jitter. If $\Delta d < 0.05$ meters (5 cm), displacement is clamped to 0.0, completely eliminating stationary object speed noise.

#### Exponential Moving Average (EMA)
Raw frame-to-frame velocity fluctuates due to tracking centroid variance. We apply EMA smoothing with factor $\alpha = 0.3$:

$$v_{\text{smooth}, t} = \alpha \cdot v_{\text{instant}, t} + (1 - \alpha) \cdot v_{\text{smooth}, t-1}$$

### Stage 6: Spatial Density Heatmaps
Traffic hotspots are accumulated using a 2D Gaussian density kernel on a discrete metric grid:

$$G(x, y) = \exp\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2 \sigma^2}\right)$$

The density grid is normalized, color-mapped with OpenCV's `COLORMAP_JET`, and alpha-blended over the Bird's-Eye-View (BEV) road canvas.
