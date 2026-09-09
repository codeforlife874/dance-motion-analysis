import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


INPUT_CSV = "output/data/landmarks.csv"

os.makedirs("output/plots", exist_ok=True)


df = pd.read_csv(INPUT_CSV)


def get_landmark(frame_data, landmark_id):

    row = frame_data[
        frame_data["landmark_id"] == landmark_id
    ]

    if row.empty:
        return None

    row = row.iloc[0]

    return np.array([
        row["x"],
        row["y"]
    ])


def calculate_angle(a, b, c):

    # vectors BA and BC
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle


angles = []


for frame in sorted(df["frame"].unique()):

    frame_data = df[
        df["frame"] == frame
    ]


    # Left elbow:
    # shoulder - elbow - wrist

    shoulder = get_landmark(
        frame_data,
        11
    )

    elbow = get_landmark(
        frame_data,
        13
    )

    wrist = get_landmark(
        frame_data,
        15
    )


    if (
        shoulder is not None
        and elbow is not None
        and wrist is not None
    ):

        angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        angles.append({
            "frame": frame,
            "left_elbow_angle": angle
        })


angles_df = pd.DataFrame(angles)


# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    angles_df["frame"],
    angles_df["left_elbow_angle"]
)

plt.xlabel("Frame")

plt.ylabel("Angle (degrees)")

plt.title(
    "Left Elbow Angle Over Time"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/plots/left_elbow_angle.png"
)

plt.close()


print(
    "Joint angle analysis completed!"
)