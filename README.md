# Basketball AI Match Analysis

A starter Streamlit application for basketball video analysis using YOLO object detection and ByteTrack tracking.

## Features currently implemented

- Upload MP4, MOV, AVI or MKV match videos.
- Detect people and the COCO sports-ball class with a pretrained YOLO model.
- Track detected objects with ByteTrack IDs.
- Manually assign tracker IDs to Team A or Team B and choose team colours.
- Draw basketball detections with a red bounding box.
- Enter team names and a manual scoreboard.
- Export an annotated MP4 and player tracking summary CSV.

## Run locally

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run may download pretrained YOLO weights. Processing speed depends on video length and hardware.

## Current limitations — please read

This repository is an MVP starter, not a fully trained basketball analytics system.

- The pretrained COCO model detects the general person class and sports-ball class. It does not guarantee reliable basketball-only ball detection.
- Hoop/court detection, jersey-number OCR, coach/referee classification, automatic basket counting, action recognition, assists, and verified player ratings are **not implemented**.
- Team assignment is manual using tracker IDs.
- Scores are entered manually.
- A tracker ID can switch or disappear when players overlap or leave the frame.
- Detected-frame counts are tracking activity, not player performance scores.

These advanced features need inspected, labelled datasets, dedicated training, and validation on match footage.

## Dataset information

The three ZIP archives visible in the user's screenshot have not been uploaded to this repository or inspected. Their displayed names and approximate sizes are recorded in [dataset_instructions.md](dataset_instructions.md). Do not assume the archive labels or formats are compatible until inspected.

## Suggested next development steps

1. Inspect and document the supplied datasets and their licences.
2. Train and validate dedicated player, ball and hoop detectors.
3. Add robust jersey-number recognition and editable player profiles.
4. Implement court calibration and event detection.
5. Validate basket counting and statistics against manually labelled match clips.

## Responsible use

Only use match footage and datasets that you have permission to use. Do not commit secrets, private tokens, or personal data.
