import pandas as pd
import numpy as np
import os


# ==========================================
# Configuration
# ==========================================

INPUT_CSV = "output/data/landmarks.csv"
OUTPUT_CSV = "output/data/detected_moves.csv"


# ==========================================
# MediaPipe landmark IDs
# ==========================================

LANDMARKS = {
    "nose": 0,

    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

    "left_hip": 23,
    "right_hip": 24,

    "left_knee": 25,
    "right_knee": 26,

    "left_ankle": 27,
    "right_ankle": 28
}


# ==========================================
# Load landmark data
# ==========================================

print("Loading landmark data...")

df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} landmark records.")
print(f"Frames: {df['frame'].nunique()}")


# ==========================================
# Convert dataframe into frame-based structure
# ==========================================

pivot = df.pivot_table(
    index="frame",
    columns="landmark_id",
    values=["x", "y"],
    aggfunc="first"
)


# ==========================================
# Helper function
# ==========================================

def get_point(frame, landmark_id):

    try:

        x = pivot.loc[frame, ("x", landmark_id)]
        y = pivot.loc[frame, ("y", landmark_id)]

        if pd.isna(x) or pd.isna(y):
            return None

        return np.array([x, y])

    except KeyError:

        return None


# ==========================================
# Calculate angle between 3 points
# ==========================================

def calculate_angle(a, b, c):

    if a is None or b is None or c is None:
        return None

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine = np.dot(ba, bc) / (
        norm_ba * norm_bc
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(
        np.arccos(cosine)
    )


# ==========================================
# Detect movement for each frame
# ==========================================

results = []


frames = sorted(
    pivot.index.tolist()
)


previous_positions = {}


for frame in frames:

    # --------------------------------------
    # Get important points
    # --------------------------------------

    left_shoulder = get_point(
        frame,
        LANDMARKS["left_shoulder"]
    )

    right_shoulder = get_point(
        frame,
        LANDMARKS["right_shoulder"]
    )

    left_wrist = get_point(
        frame,
        LANDMARKS["left_wrist"]
    )

    right_wrist = get_point(
        frame,
        LANDMARKS["right_wrist"]
    )

    left_hip = get_point(
        frame,
        LANDMARKS["left_hip"]
    )

    right_hip = get_point(
        frame,
        LANDMARKS["right_hip"]
    )

    left_knee = get_point(
        frame,
        LANDMARKS["left_knee"]
    )

    right_knee = get_point(
        frame,
        LANDMARKS["right_knee"]
    )

    left_ankle = get_point(
        frame,
        LANDMARKS["left_ankle"]
    )

    right_ankle = get_point(
        frame,
        LANDMARKS["right_ankle"]
    )


    # ======================================
    # HAND RAISE DETECTION
    # ======================================

    left_hand_raised = False
    right_hand_raised = False


    if (
        left_wrist is not None
        and left_shoulder is not None
    ):

        if left_wrist[1] < left_shoulder[1]:

            left_hand_raised = True


    if (
        right_wrist is not None
        and right_shoulder is not None
    ):

        if right_wrist[1] < right_shoulder[1]:

            right_hand_raised = True


    # ======================================
    # ELBOW ANGLES
    # ======================================

    left_elbow = get_point(
        frame,
        LANDMARKS["left_elbow"]
    )

    right_elbow = get_point(
        frame,
        LANDMARKS["right_elbow"]
    )


    left_elbow_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )


    right_elbow_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )


    # ======================================
    # KNEE ANGLES
    # ======================================

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )


    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )


    # ======================================
    # SQUAT DETECTION
    # ======================================

    squat = False

    valid_knee_angles = []

    if left_knee_angle is not None:
        valid_knee_angles.append(left_knee_angle)

    if right_knee_angle is not None:
        valid_knee_angles.append(right_knee_angle)


    if len(valid_knee_angles) > 0:

        average_knee_angle = np.mean(
            valid_knee_angles
        )

        # Bent knees indicate a squat-like pose
        if average_knee_angle < 120:
            squat = True

    else:

        average_knee_angle = None


    # ======================================
    # LEG LIFT DETECTION
    # ======================================

    leg_lift = False

    if (
        left_ankle is not None
        and right_ankle is not None
        and left_knee is not None
        and right_knee is not None
    ):

        ankle_height_difference = abs(
            left_ankle[1] - right_ankle[1]
        )

        if ankle_height_difference > 0.12:

            leg_lift = True


    # ======================================
    # ARM MOVEMENT DETECTION
    # ======================================

    arm_movement = False

    current_positions = {
        "left_wrist": left_wrist,
        "right_wrist": right_wrist
    }


    movement_values = []


    for name, current in current_positions.items():

        previous = previous_positions.get(name)

        if (
            current is not None
            and previous is not None
        ):

            movement = np.linalg.norm(
                current - previous
            )

            movement_values.append(
                movement
            )


    if movement_values:

        average_arm_movement = np.mean(
            movement_values
        )

        if average_arm_movement > 0.015:

            arm_movement = True

    else:

        average_arm_movement = 0


    previous_positions = current_positions


    # ======================================
    # Determine primary movement
    # ======================================

    detected_move = "Standing / Neutral"


    if squat:

        detected_move = "Squat"

    elif leg_lift:

        detected_move = "Leg Lift"

    elif left_hand_raised or right_hand_raised:

        detected_move = "Hand Raise"

    elif arm_movement:

        detected_move = "Arm Movement"


    # ======================================
    # Store result
    # ======================================

    results.append({

        "frame": frame,

        "left_hand_raised":
            left_hand_raised,

        "right_hand_raised":
            right_hand_raised,

        "squat":
            squat,

        "leg_lift":
            leg_lift,

        "arm_movement":
            arm_movement,

        "left_elbow_angle":
            left_elbow_angle,

        "right_elbow_angle":
            right_elbow_angle,

        "average_knee_angle":
            average_knee_angle,

        "arm_movement_amount":
            average_arm_movement,

        "detected_move":
            detected_move
    })


# ==========================================
# Save results
# ==========================================

results_df = pd.DataFrame(results)


os.makedirs(
    "output/data",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==========================================
# Print summary
# ==========================================

print()
print("==========================================")
print("Dance movement detection completed!")
print("==========================================")

print(f"Output: {OUTPUT_CSV}")

print()
print("Detected movement counts:")
print(
    results_df["detected_move"]
    .value_counts()
)

print()
print("Done!")