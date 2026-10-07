# Camera Calibration & Planar Homography Guide

This guide explains how to calibrate camera perspectives in SpatialTrack to map 2D image coordinates (pixels) to metric ground coordinates (meters).

---

## 1. Projective Geometry Overview

A camera lens projects a 3D real-world scene onto a 2D digital image plane through perspective distortion:
- Distant objects appear smaller than nearby objects.
- Parallel road lanes converge towards a vanishing point.
- Pixel displacement does not scale linearly with real-world distance.

When tracking vehicles on a relatively planar road surface, we assume all vehicle contact points lie on the ground plane ($Z = 0$). Under this planar assumption, the transformation between camera image pixels $(u, v)$ and metric coordinates $(X, Y)$ is governed by a **2D Projective Transformation (Homography)**:

$$\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

$$X = \frac{x'}{w}, \quad Y = \frac{y'}{w}$$

Where $\mathbf{H}$ is a $3 \times 3$ non-singular matrix with 8 degrees of freedom (up to a scale factor). Exactly **4 non-collinear point correspondences** are necessary and sufficient to solve $\mathbf{H}$ uniquely.

---

## 2. Choosing Calibration Reference Points

To achieve high metric accuracy (typically within 1 to 3 km/h of radar guns), select 4 points on the road surface with known real-world distances:

1. **Standard Highway Lane Markings:**
   - In most countries, highway skip lines have standardized lengths (e.g., 3.0 meters of paint, 9.0 meters of gap).
   - Standard highway lane width is typically 3.5 to 3.7 meters.
2. **Pedestrian Crosswalks:**
   - Standard continental crosswalk stripes are usually 0.5m wide with 0.5m gaps and 3.0m or 4.0m lengths.
3. **Physical Landmarks:**
   - Curb intersections, lamp posts, or physical cones placed during camera setup.

### Point Ordering Convention
Choose points in consistent clockwise order starting from the bottom-left:
- **Point 1:** Near Left (e.g., $(0.0, 0.0)$ meters)
- **Point 2:** Far Left (e.g., $(0.0, 40.0)$ meters)
- **Point 3:** Far Right (e.g., $(12.0, 40.0)$ meters)
- **Point 4:** Near Right (e.g., $(12.0, 0.0)$ meters)

---

## 3. Interactive Calibration Tool

SpatialTrack includes a built-in calibration wizard.

### Interactive GUI Mode
Run the command pointing to your video file or static snapshot:

```bash
spatialtrack calibrate --source sample_traffic.mp4 --output configs/my_calibration.json
```

1. An OpenCV window opens displaying the first frame.
2. Click the 4 reference points on the road plane in clockwise order.
3. For each clicked point, the console prompts for the corresponding world coordinates in meters.
4. The tool computes the homography matrix $\mathbf{H}$, verifies its condition number, and writes the JSON calibration file.

### Headless / Automated Mode
In server environments without a graphical display, supply coordinate pairs directly via CLI flags:

```bash
spatialtrack calibrate \
  --source sample_traffic.mp4 \
  --output configs/my_calibration.json \
  --src-points "523,412; 758,412; 980,680; 300,680" \
  --dst-points "0,0; 12,0; 12,40; 0,40" \
  --description "Highway Section A"
```

---

## 4. Calibration File Format

The resulting JSON file adheres to the following structured format:

```json
{
  "source_points_px": [
    [523.0, 412.0],
    [758.0, 412.0],
    [980.0, 680.0],
    [300.0, 680.0]
  ],
  "target_points_m": [
    [0.0, 0.0],
    [12.0, 0.0],
    [12.0, 40.0],
    [0.0, 40.0]
  ],
  "frame_width": 1280,
  "frame_height": 720,
  "description": "Highway Section A"
}
```

---

## 5. Verification & Quality Checks

SpatialTrack runs automated validation checks on calibration data:

1. **Non-Collinearity:** Points must form a non-degenerate convex quadrilateral. Three collinear points produce an unsolvable linear system.
2. **Matrix Conditioning:** The condition number $\kappa(\mathbf{H}) = \|\mathbf{H}\| \|\mathbf{H}^{-1}\|$ measures sensitivity to numerical noise. A condition number under 500 indicates a well-posed transformation.
3. **Reprojection Round-Trip:** The mapper tests $\mathbf{H}^{-1}(\mathbf{H}(p)) \approx p$ ensuring round-trip numerical error is $< 10^{-4}$ meters.
