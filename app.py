import os
import tempfile
from collections import Counter

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Basketball AI Match Analysis", page_icon="🏀", layout="wide")

st.markdown("""
<style>
.stApp {background: #0b1220; color: #edf2f7;}
[data-testid="stSidebar"] {background: #111c2e;}
.block-container {padding-top: 1.5rem;}
.metric-card {background:#15243a;padding:16px;border-radius:14px;border:1px solid #263b59;}
</style>
""", unsafe_allow_html=True)

st.title("🏀 Basketball AI Match Analysis")
st.caption("Video-based player detection and tracking powered by YOLO and ByteTrack")

with st.sidebar:
    st.header("⚙️ Match settings")
    team_a = st.text_input("Team A name", "Team A")
    team_b = st.text_input("Team B name", "Team B")
    color_a = st.color_picker("Team A box colour", "#00C8FF")
    color_b = st.color_picker("Team B box colour", "#FFD43B")
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    model_choice = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt"], index=0)
    st.info("Assign teams by tracker ID after a preview run. Team assignments are manual.")

uploaded = st.file_uploader("Upload a basketball match video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    st.video(uploaded)
    st.subheader("Player team assignment")
    st.caption("After processing, enter tracker IDs separated by commas. Unassigned players remain neutral.")
    col_a, col_b = st.columns(2)
    with col_a:
        ids_a_text = st.text_input(f"{team_a} player IDs", "")
    with col_b:
        ids_b_text = st.text_input(f"{team_b} player IDs", "")

    score1, score2 = st.columns(2)
    with score1:
        points_a = st.number_input(f"{team_a} score (manual)", min_value=0, step=1, value=0)
    with score2:
        points_b = st.number_input(f"{team_b} score (manual)", min_value=0, step=1, value=0)

    if st.button("▶️ Analyse match video", type="primary", use_container_width=True):
        with st.spinner("Loading YOLO model and analysing video. The first run may download model weights..."):
            try:
                ids_a = {int(x.strip()) for x in ids_a_text.split(",") if x.strip()}
                ids_b = {int(x.strip()) for x in ids_b_text.split(",") if x.strip()}
                if ids_a & ids_b:
                    st.error("A tracker ID cannot belong to both teams. Please correct the ID lists.")
                    st.stop()

                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
                    src.write(uploaded.getbuffer())
                    input_path = src.name

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    st.error("Could not open this video. Try an MP4 video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    cap.release()
                    st.error("Could not create output video on this environment.")
                    st.stop()

                model = YOLO(model_choice)
                player_frames = Counter()
                frame_count = 0
                progress = st.progress(0.0)
                status = st.empty()
                preview = st.empty()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                           conf=conf, classes=[0, 32], verbose=False)
                    annotated = frame.copy()
                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes
                        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
                        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
                        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
                        for box, class_id, track_id in zip(xyxy, cls, track_ids):
                            x1, y1, x2, y2 = map(int, box)
                            if class_id == 32:
                                colour = (0, 0, 255)
                                label = "BASKETBALL"
                            elif class_id == 0:
                                player_frames[track_id] += 1
                                if track_id in ids_a:
                                    colour, label = tuple(int(color_a[i:i+2], 16) for i in (5,3,1)), f"{team_a} | ID {track_id}"
                                elif track_id in ids_b:
                                    colour, label = tuple(int(color_b[i:i+2], 16) for i in (5,3,1)), f"{team_b} | ID {track_id}"
                                else:
                                    colour, label = (190, 190, 190), f"Player | ID {track_id}"
                            else:
                                continue
                            cv2.rectangle(annotated, (x1,y1), (x2,y2), colour, 2)
                            cv2.putText(annotated, label, (x1, max(22,y1-8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
                    writer.write(annotated)
                    frame_count += 1
                    if frame_count % 10 == 0 or frame_count == total_frames:
                        progress.progress(min(frame_count / max(total_frames,1), 1.0))
                        status.write(f"Processed {frame_count:,} frames")
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Live annotated preview")
                cap.release()
                writer.release()
                try:
                    os.remove(input_path)
                except OSError:
                    pass

                st.session_state["analysis"] = {
                    "output": output_path,
                    "counts": dict(player_frames),
                    "frames": frame_count,
                    "fps": fps,
                    "score_a": points_a,
                    "score_b": points_b,
                    "team_a": team_a,
                    "team_b": team_b
                }
                st.success("Video analysis completed.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

if "analysis" in st.session_state:
    data = st.session_state["analysis"]
    st.divider()
    st.header("📊 Match report")
    m1, m2, m3 = st.columns(3)
    m1.metric("Frames analysed", f"{data['frames']:,}")
    m2.metric("Tracked player IDs", len(data["counts"]))
    m3.metric("Video duration", f"{data['frames']/max(data['fps'],1):.1f} sec")
    left, right = st.columns(2)
    left.metric(data["team_a"], data["score_a"])
    right.metric(data["team_b"], data["score_b"])
    if data["score_a"] > data["score_b"]:
        st.success(f"Manual scoreboard result: {data['team_a']} leads.")
    elif data["score_b"] > data["score_a"]:
        st.success(f"Manual scoreboard result: {data['team_b']} leads.")
    else:
        st.info("Manual scoreboard is tied.")
    rows = [{"Tracker ID": k, "Detected frames": v} for k,v in sorted(data["counts"].items())]
    df = pd.DataFrame(rows)
    st.subheader("Player tracking summary")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download player statistics CSV", df.to_csv(index=False),
                       file_name="basketball_player_stats.csv", mime="text/csv")
    with open(data["output"], "rb") as f:
        st.download_button("⬇️ Download annotated match video", f.read(),
                           file_name="basketball_annotated.mp4", mime="video/mp4")
    st.caption("Detected-frame counts are tracking activity, not verified player performance. Scores are entered manually.")

st.divider()
st.caption("MVP limitations: this starter detects COCO person and sports-ball classes. Hoop detection, jersey OCR, automatic basket/score detection, referee/coach classification, actions and true player ratings require dedicated labelled data and trained models.")
