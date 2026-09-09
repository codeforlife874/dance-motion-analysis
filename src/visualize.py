import pandas as pd
import matplotlib.pyplot as plt
import os


LANDMARKS_CSV = "output/data/landmarks.csv"
MOTION_CSV = "output/data/motion_features.csv"

os.makedirs("output/plots", exist_ok=True)


# ==========================================
# Load data
# ==========================================

landmarks = pd.read_csv(LANDMARKS_CSV)
motion = pd.read_csv(MOTION_CSV)


# ==========================================
# Helper function
# ==========================================

def plot_trajectory(landmark_id, name):

    data = landmarks[
        landmarks["landmark_id"] == landmark_id
    ].sort_values("frame")


    plt.figure(figsize=(8, 6))

    plt.plot(
        data["x"],
        data["y"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("X position")
    plt.ylabel("Y position")

    plt.title(
        f"{name} Movement Trajectory"
    )

    plt.grid(True)

    plt.tight_layout()

    filename = (
        f"output/plots/{name}_trajectory.png"
    )

    plt.savefig(filename)

    plt.close()

    print(f"Saved {filename}")


# ==========================================
# Generate trajectories
# ==========================================

plot_trajectory(
    15,
    "left_wrist"
)

plot_trajectory(
    16,
    "right_wrist"
)

plot_trajectory(
    27,
    "left_ankle"
)

plot_trajectory(
    28,
    "right_ankle"
)


# ==========================================
# Movement intensity
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    motion["frame"],
    motion["total_movement"]
)

plt.xlabel("Frame")

plt.ylabel("Movement intensity")

plt.title(
    "Dance Movement Intensity Over Time"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/plots/movement_intensity.png"
)

plt.close()


print(
    "Movement intensity plot saved."
)