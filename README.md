# BoxMOT Standalone Package

This is a minimal standalone version of BoxMOT containing only the essential modules needed to run the tracking system.

## Contents

- `boxmot/` - Core tracking modules
  - `trackers/botsort/` - BoTSORT tracker implementation
  - `appearance/` - ReID models and backends  
  - `motion/` - Kalman filters and camera motion compensation
  - `utils/` - Utility functions
  - `qdrant/` - Long-term feature storage
  - `configs/` - Configuration files
- `yolov7/` - YOLOv7 detection model
- `yolov7_track_pending.py` - Main tracking script
- `test_standalone.py` - Test script to verify package works
- Weight files:
  - `best.pt` - YOLOv7 detection weights
  - `osnet_x1_0_market1501.pt` - ReID model weights

## Requirements

```bash
pip install torch torchvision opencv-python numpy scipy loguru qdrant-client pyyaml
```

## Usage

1. Test the package:
```bash
python test_standalone.py
```

2. Run tracking:
```bash
python yolov7_track_pending.py
```

## Multi-Camera Demo

This repo includes a simple multi-camera demo that runs BoT-SORT per camera, then assigns global IDs across cameras and draws track positions on a shared map.

### Prerequisites
- Weights in repo root:
  - `best.pt` (YOLOv7 detector)
  - `osnet_x1_0_market1501.pt` (ReID model)
- Assets for the Warehouse_012 sample:
  - Map and calibration: `homography-computation/PhysicalAI-SmartSpaces_py/MTMC_Tracking_2025/train/Warehouse_012/{map.png, calibration.json}`
  - Videos: `homography-computation/PhysicalAI-SmartSpaces_py/MTMC_Tracking_2025/train/Warehouse_012/videos/Camera_00.mp4`, `Camera_02.mp4`
- GPU recommended. Set `device` to `cpu` if no CUDA.

### Run
```bash
python multicam_demo.py
```
- Output video: `mc_demo_1.avi` (two camera panels on the left, map panel on the right)
- Default device: CUDA (`cuda:0`). Adjust if needed.

### Configure
- Edit camera sources in `multicam_demo.py` (list `camera_configs` around `multicam_demo.py:353`). Add more cameras by appending `CameraConfig` items.
- Core demo settings (see `DemoConfig` in `multicam_demo.py:45`):
  - `weights`: YOLOv7 weights path
  - `tracker_type`: default `botsort`
  - `detect_class`: COCO class id (2 = person in this setup)
  - `conf_thres`, `iou_thres`: detector thresholds
  - `device`: e.g. `cuda:0` or `cpu`
  - `out_path`: output video path
  - `img_size`, `cam_display_width`: inference size and display width
  - Map options: `map_image`, `calibration_json`, `map_rotation_deg`, `map_flip_x`, `map_flip_y`
- Global ID matching threshold (optional): change `GlobalIDManager(sim_threshold=...)` where it’s created (`multicam_demo.py:380`) or in `boxmot/multicam/global_id.py` default.

### What It Does
- Per camera: YOLOv7 detect → BoT-SORT tracking → features extracted for ReID → local tracks.
- Cross-camera: `GlobalIDManager` assigns global IDs using cosine similarity over per-entity prototypes.
- Map overlay: homography projects each bbox center onto the shared map and displays `LID→GID` with a distance indicator.

### Optional: Qdrant
- If `qdrant-client` is installed and a server is running at `localhost:6333`, the demo will clear the `long_term_reid` collection at start (best-effort).
- Long-term ReID storage is optional for this demo; basic global ID uses in-memory prototypes.

### Troubleshooting
- "Unable to open video source": verify the `Camera_XX.mp4` paths exist.
- CUDA errors: set `DemoConfig.device` to a valid GPU index or `cpu`.
- `ModuleNotFoundError: yolov7.*`: ensure the `yolov7/` folder exists in repo root (it’s included here).
- Output video issues: the `XVID` codec is used; ensure OpenCV build supports it, or switch codec in `multicam_demo.py`.

## Key Features

- BoTSORT tracking with pending management
- IDSD (ID Switch Detection) logic integrated
- Long-term ReID with Qdrant vector database
- YOLOv7 object detection
- Real-time visualization with track history

## Configuration

Edit the `TrackingConfig` class in `yolov7_track_pending.py` to change:
- Input video source
- Detection confidence thresholds
- Output video path
- Model weights

## Notes

This standalone package includes only the minimal dependencies needed for tracking. The full BoxMOT repository contains additional trackers and utilities not included here.
