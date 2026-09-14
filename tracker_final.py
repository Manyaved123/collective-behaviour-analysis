"""
LIVE / VIDEO COLLECTIVE BEHAVIOUR TRACKER -- FINAL FLEXIBLE VERSION
======================================================================
Lets you choose, every time you run it:
  1. WHERE the footage comes from: your live webcam, OR a saved video file
  2. WHAT to track: person, or one of several common animals YOLO already
     knows (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)

NOTE ON FISH: the AI model this uses (YOLO) was never trained to recognize
fish, so it cannot detect fish -- this is a real, honest limitation, not a
bug. For fish specifically, you'd go back to the TRex tool from earlier
(which doesn't "know" species at all, it just detects anything that moves
against a still background, which works fine for fish).

HOW TO RUN:
    python tracker_final.py

It will ask you questions in the terminal (no need to edit the code).
"""

import sys
import cv2
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from ultralytics import YOLO

# COCO class IDs (what YOLO's pretrained model already knows) for common subjects
CLASS_MAP = {
    "person": 0,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
}

TRAIL_LENGTH = 15
INTERACTION_DISTANCE = 250
PREDICTION_FRAMES_AHEAD = 10
MODEL_NAME = "yolov8n.pt"

# ---- ASK THE USER WHAT THEY WANT, EACH RUN ----
print("What do you want to track?")
print("Options:", ", ".join(CLASS_MAP.keys()))
target = input("Type one option: ").strip().lower()
while target not in CLASS_MAP:
    print("Not recognized. Please type exactly one of:", ", ".join(CLASS_MAP.keys()))
    target = input("Type one option: ").strip().lower()
target_class_id = CLASS_MAP[target]

print("\nWhere is the footage coming from?")
print("  1 = live webcam")
print("  2 = a saved video file")
choice = input("Type 1 or 2: ").strip()
if choice == "2":
    source = input("Paste the full path to your video file: ").strip().strip('"')
else:
    source = 0

print(f"\nTracking: {target}   |   Source: {'webcam' if source == 0 else source}")
print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
all_records = []

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("ERROR: Could not open that video source. Check the path or webcam index.")
    sys.exit()

print("Running. Press 'q' in the video window to stop and save results.")

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended (or webcam disconnected).")
        break

    results = model.track(frame, persist=True, classes=[target_class_id], verbose=False)

    centers = {}
    directions = {}

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, pid in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers[pid] = (cx, cy)
            trails[pid].append((cx, cy))
            all_records.append({"frame": frame_number, "individual_id": int(pid), "x": cx, "y": cy})

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            cv2.putText(frame, f"ID {pid}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            pts = list(trails[pid])
            for j in range(1, len(pts)):
                cv2.line(frame, (int(pts[j-1][0]), int(pts[j-1][1])),
                          (int(pts[j][0]), int(pts[j][1])), (0, 200, 255), 2)

            if len(pts) >= 2:
                dx = pts[-1][0] - pts[0][0]
                dy = pts[-1][1] - pts[0][1]
                mag = np.hypot(dx, dy)
                if mag > 1:
                    directions[pid] = (dx / mag, dy / mag)
                    velocity_per_frame = np.array([dx, dy]) / (len(pts) - 1)
                    predicted_point = np.array(pts[-1]) + velocity_per_frame * PREDICTION_FRAMES_AHEAD
                    start = np.array(pts[-1])
                    end = predicted_point
                    n_dashes = 8
                    for k in range(n_dashes):
                        if k % 2 == 0:
                            seg_start = start + (end - start) * (k / n_dashes)
                            seg_end = start + (end - start) * ((k + 1) / n_dashes)
                            cv2.line(frame, (int(seg_start[0]), int(seg_start[1])),
                                      (int(seg_end[0]), int(seg_end[1])), (0, 0, 255), 2)
                    cv2.circle(frame, (int(end[0]), int(end[1])), 5, (0, 0, 255), -1)

    ids_list = list(centers.keys())
    for i in range(len(ids_list)):
        for j in range(i + 1, len(ids_list)):
            p1, p2 = centers[ids_list[i]], centers[ids_list[j]]
            dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if dist < INTERACTION_DISTANCE:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

    n_individuals = len(centers)
    polarization_text = "N/A"
    cohesion_text = "N/A"

    if len(directions) >= 2:
        dirs = np.array(list(directions.values()))
        avg_dir = dirs.mean(axis=0)
        polarization_text = f"{np.linalg.norm(avg_dir):.2f}"

    if n_individuals >= 2:
        pts_arr = np.array(list(centers.values()))
        nearest_dists = []
        for i in range(len(pts_arr)):
            diffs = pts_arr - pts_arr[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nearest_dists.append(dists.min())
        cohesion_text = f"{np.mean(nearest_dists):.0f} px"

    overlay_lines = [
        f"Tracking: {target}   Frame: {frame_number}   Detected: {n_individuals}",
        f"Polarization: {polarization_text}   Avg. nearest-neighbor dist: {cohesion_text}",
        "Red dashed line = predicted future position",
    ]
    for idx, line in enumerate(overlay_lines):
        cv2.putText(frame, line, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Collective Behaviour Tracker (press q to stop & save)", frame)
    frame_number += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nSaving results...")
if len(all_records) == 0:
    print(f"No '{target}' was detected during this run -- nothing to save. Try a clearer video or a different target.")
    sys.exit()

df = pd.DataFrame(all_records)
df.to_csv("trajectories.csv", index=False)
print(f"Saved trajectories.csv ({len(df)} rows)")

metric_rows = []
for frame_num, group in df.groupby("frame"):
    positions = group[["x", "y"]].values
    if len(positions) >= 2:
        nearest_dists = []
        for i in range(len(positions)):
            diffs = positions - positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nearest_dists.append(dists.min())
        cohesion = np.mean(nearest_dists)
    else:
        cohesion = np.nan
    metric_rows.append({"frame": frame_num, "n_individuals": len(positions), "avg_nearest_neighbor_dist": cohesion})

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv("metrics_over_time.csv", index=False)
print("Saved metrics_over_time.csv")

plt.figure(figsize=(10, 4))
plt.plot(metrics_df["frame"], metrics_df["avg_nearest_neighbor_dist"], color="steelblue")
plt.xlabel("Frame")
plt.ylabel("Avg. nearest-neighbor distance (px)")
plt.title(f"Group cohesion over time ({target})")
plt.tight_layout()
plt.savefig("metrics_plot.png", dpi=150)
print("Saved metrics_plot.png")

last_frame_num = df["frame"].max()
snapshot = df[df["frame"] == last_frame_num].reset_index(drop=True)

G = nx.Graph()
for i, row in snapshot.iterrows():
    G.add_node(row["individual_id"], pos=(row["x"], row["y"]))
for i in range(len(snapshot)):
    for j in range(i + 1, len(snapshot)):
        dist = np.linalg.norm(snapshot.loc[i, ["x", "y"]].values - snapshot.loc[j, ["x", "y"]].values)
        if dist < INTERACTION_DISTANCE:
            G.add_edge(snapshot.loc[i, "individual_id"], snapshot.loc[j, "individual_id"])

pos = nx.get_node_attributes(G, "pos")
plt.figure(figsize=(6, 6))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, edge_color="gray")
plt.title(f"Interaction network -- {target} (last frame: {last_frame_num})")
plt.gca().invert_yaxis()
plt.savefig("final_network.png", dpi=150)
print("Saved final_network.png")

max_simultaneous = df.groupby("frame")["individual_id"].nunique().max()
print(f"\nSummary:")
print(f"  Total ID numbers ever assigned: {df['individual_id'].nunique()} "
      f"(inflated by tracking briefly losing/re-finding individuals -- not the true count)")
print(f"  MAX individuals seen at the same time in one frame: {max_simultaneous} "
      f"(this is the honest headcount of the group)")
print(f"  Frames processed: {df['frame'].nunique()}")
print("All done! Check this folder for your 4 result files.")
