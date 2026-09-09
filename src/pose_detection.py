import cv2
import mediapipe as mp
import pandas as pd
import os


# -----------------------------
# Configuration
# -----------------------------

VIDEO_PATH = "input/dance.mp4"
OUTPUT_VIDEO = "output/annotated/dance_pose.mp4"
OUTPUT_CSV = "output/data/landmarks.csv"


# -----------------------------
# Create output directories
# -----------------------------

os.makedirs("output/annotated", exist_ok=True)
os.makedirs("output/data", exist_ok=True)


# -----------------------------
# Initialize MediaPipe
# -----------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Open video
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# -----------------------------
# Video writer
# -----------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# -----------------------------
# Store landmark data
# -----------------------------

landmark_data = []

frame_number = 0


# -----------------------------
# Process video
# -----------------------------

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb_frame)


    # -------------------------
    # If pose detected
    # -------------------------

    if results.pose_landmarks:

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # Save landmarks
        for landmark_id, landmark in enumerate(
            results.pose_landmarks.landmark
        ):

            landmark_data.append({
                "frame": frame_number,
                "landmark_id": landmark_id,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })


    # Write annotated frame
    out.write(frame)


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
out.release()
pose.close()


# -----------------------------
# Save CSV
# -----------------------------

df = pd.DataFrame(landmark_data)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


print("Pose detection completed!")
print(f"Annotated video: {OUTPUT_VIDEO}")
print(f"Landmark data: {OUTPUT_CSV}")
print(f"Total frames processed: {frame_number}")
print(f"Landmark records: {len(df)}")