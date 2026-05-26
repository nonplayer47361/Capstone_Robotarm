# External Dataset Notes

`external_datasets/` is a local-only folder for downloaded open-source datasets.
The folder is ignored by Git because these files can be large and may have their
own licenses or redistribution limits.

## Prepared For Future Tests

- `external_datasets/bottle cap.v1i.yolov8.zip`
  - Purpose: future bottle-cap object detection and A4 coordinate generalization test
  - Format: Roboflow YOLOv8 export
  - Status: downloaded locally, not used in the current pill-cap-only experiment

- `external_datasets/NASA Rockyard.v1i.yolov8.zip`
  - Purpose: future irregular-shape rock/stone detection test
  - Format: Roboflow YOLOv8 export
  - Status: downloaded locally, not used in the current pill-cap-only experiment

## Current Scope

The active A4 coordinate experiment remains focused on the pill-cap model. These
external datasets are reserved for later expansion after the pill-cap pipeline is
validated.
