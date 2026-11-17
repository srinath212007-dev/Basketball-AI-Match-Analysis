# Dataset instructions

Three dataset ZIP files were shown in a Windows File Explorer screenshot:

1. **Main dataset – Players, ball, hoop and court** — displayed size approximately 1,756 KB.
2. **Jersey numbers and basketball actions** — displayed size approximately 86,039 KB.
3. **Additional ball, player and hoop datasets** — displayed size approximately 26,320 KB.

These names and sizes are transcribed from the screenshot only. The ZIP archives themselves were not attached to this GitHub upload, so their contents, labels, annotations, class maps, licences, and formats have not been inspected or verified.

## Before using the archives

1. Keep original copies of all ZIP files.
2. Extract each archive and inspect its folder structure.
3. Read the dataset licence and usage restrictions.
4. Identify annotation format and exact class names.
5. Check sample images and labels for correctness.
6. Split into training, validation, and test sets without leakage.
7. Document preprocessing and class mappings before training.

## Storage guidance

Do not commit large raw datasets or model weights unless permitted and appropriate. Store large files in an approved cloud dataset location and document the download process. Never commit credentials, access tokens, private links, or personal data.

## Current application

The current Streamlit MVP uses pretrained Ultralytics YOLO weights for general person and sports-ball detection. It does not yet train on the three custom archives.
