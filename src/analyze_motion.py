import pandas as pd
import numpy as np
import os


# ==========================================
# File paths
# ==========================================

INPUT_CSV = "output/data/landmarks.csv"
OUTPUT_CSV = "output/data/motion_features.csv"


# ==========================================
# Load landmark data
# ==========================================

print("Loading landmark data...")

df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} landmark records.")
print(f"Frames found: {df['frame'].nunique()}")


# ==========================================
# MediaPipe landmark IDs
# ==========================================

landmarks = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24
}


# ==========================================
# Create output dataframe
# ==========================================

frames = sorted(df["frame"].unique())

features = pd.DataFrame({
    "frame": frames
})


# ==========================================
# Calculate movement for each body part
# ==========================================

for name, landmark_id in landmarks.items():

    print(f"Analyzing {name}...")

    data = df[
        df["landmark_id"] == landmark_id
    ].copy()

    data = data.sort_values("frame")

    # Calculate movement between consecutive frames
    dx = data["x"].diff()
    dy = data["y"].diff()

    distance = np.sqrt(
        dx ** 2 + dy ** 2
    )

    distance = distance.fillna(0)

    features[f"{name}_movement"] = distance.values


# ==========================================
# Calculate total movement
# ==========================================

movement_columns = [
    column
    for column in features.columns
    if column.endswith("_movement")
]

features["total_movement"] = (
    features[movement_columns].sum(axis=1)
)


# ==========================================
# Save results
# ==========================================

os.makedirs(
    "output/data",
    exist_ok=True
)

features.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==========================================
# Print summary
# ==========================================

print()
print("==========================================")
print("Motion analysis completed successfully!")
print("==========================================")

print(f"Output file: {OUTPUT_CSV}")
print(f"Frames analyzed: {len(features)}")

print(
    f"Total movement: "
    f"{features['total_movement'].sum():.4f}"
)