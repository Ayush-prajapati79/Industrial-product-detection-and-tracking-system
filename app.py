"""
Industrial Product Detection and Tracking System
YOLO (detection) + ByteTrack (tracking) + virtual line counter, in a Streamlit UI.

Run:  streamlit run app.py
"""
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Industrial Product Tracking", layout="wide")
st.title("Industrial Product Detection and Tracking System")

# ---------------- Sidebar settings ----------------
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("YOLO model file", "yolov8n.pt",
                                   help="Auto-downloads on first run. Use your own best.pt for a custom model.")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
tracker_name = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
orientation = st.sidebar.radio("Counting line", ["Horizontal", "Vertical"])
line_pos = st.sidebar.slider("Line position (%)", 10, 90, 55)
source_type = st.sidebar.radio("Video source", ["Upload video", "Webcam"])

# Load once to list class names for the filter
try:
    names = YOLO(model_path).names
except Exception as e:
    st.error(f"Could not load model '{model_path}': {e}")
    st.stop()

default_cls = [n for n in ("bottle", "cup", "box", "cell phone") if n in names.values()]
selected = st.sidebar.multiselect("Classes to detect", list(names.values()),
                                  default=default_cls or list(names.values())[:3])
class_ids = [i for i, n in names.items() if n in selected] or None

video_path = None
if source_type == "Upload video":
    up = st.file_uploader("Upload a conveyor / production line video", type=["mp4", "avi", "mov", "mkv"])
    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(up.read())
        video_path = tmp.name

c1, c2 = st.columns(2)
start = c1.button("▶ Start", type="primary")
c2.button("⏹ Stop")  # pressing any button reruns the script, which ends the loop

frame_box = st.empty()
m1, m2, m3, m4 = st.columns(4)
total_ph, fps_ph, active_ph, class_ph = m1.empty(), m2.empty(), m3.empty(), m4.empty()
table_ph = st.empty()

# ---------------- Main processing ----------------
if start:
    if source_type == "Upload video" and not video_path:
        st.warning("Please upload a video first.")
        st.stop()

    model = YOLO(model_path)  # fresh instance = fresh tracker state
    cap = cv2.VideoCapture(0 if source_type == "Webcam" else video_path)

    prev_pos = {}                 # track id -> previous center coordinate along counting axis
    counted = set()               # ids already counted
    class_counts = defaultdict(int)
    log = []
    prev_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        line = int(h * line_pos / 100) if orientation == "Horizontal" else int(w * line_pos / 100)

        res = model.track(frame, persist=True, conf=conf, tracker=tracker_name,
                          classes=class_ids, verbose=False)[0]
        active = 0

        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            clss = res.boxes.cls.int().cpu().tolist()
            active = len(ids)

            for (x1, y1, x2, y2), tid, c in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                pos = cy if orientation == "Horizontal" else cx
                label = names[c]

                # Line crossing: center moved from one side of the line to the other
                if tid in prev_pos and tid not in counted:
                    if (prev_pos[tid] - line) * (pos - line) < 0:
                        counted.add(tid)
                        class_counts[label] += 1
                        log.append({"time": datetime.now().strftime("%H:%M:%S"),
                                    "track_id": tid, "class": label})
                prev_pos[tid] = pos

                color = (0, 200, 0) if tid in counted else (255, 160, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} #{tid}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Draw counting line and total
        if orientation == "Horizontal":
            cv2.line(frame, (0, line), (w, line), (0, 0, 255), 2)
        else:
            cv2.line(frame, (line, 0), (line, h), (0, 0, 255), 2)
        cv2.putText(frame, f"Count: {len(counted)}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        frame_box.image(frame, channels="BGR", use_container_width=True)
        total_ph.metric("Total counted", len(counted))
        fps_ph.metric("FPS", f"{fps:.1f}")
        active_ph.metric("Objects in view", active)
        class_ph.metric("Per class", ", ".join(f"{k}: {v}" for k, v in class_counts.items()) or "-")

    cap.release()
    st.success(f"Finished. Total products counted: {len(counted)}")

    if log:
        df = pd.DataFrame(log)
        table_ph.dataframe(df, use_container_width=True)
        st.download_button("Download count log (CSV)", df.to_csv(index=False).encode(),
                           "count_log.csv", "text/csv")
