import pandas as pd
import os


INPUT_CSV = "output/data/detected_moves.csv"
OUTPUT_CSV = "output/data/movement_segments.csv"


df = pd.read_csv(INPUT_CSV)


# ==========================================
# Group consecutive frames with same move
# ==========================================

segments = []


current_move = None
start_frame = None
previous_frame = None


for _, row in df.iterrows():

    frame = int(row["frame"])
    move = row["detected_move"]


    if current_move is None:

        current_move = move
        start_frame = frame


    elif move != current_move:

        segments.append({

            "start_frame": start_frame,

            "end_frame": previous_frame,

            "move": current_move
        })


        current_move = move
        start_frame = frame


    previous_frame = frame


# Add final segment

if current_move is not None:

    segments.append({

        "start_frame": start_frame,

        "end_frame": previous_frame,

        "move": current_move
    })


segments_df = pd.DataFrame(segments)


# ==========================================
# Remove extremely short segments
# ==========================================

segments_df["duration_frames"] = (
    segments_df["end_frame"]
    - segments_df["start_frame"]
    + 1
)


segments_df = segments_df[
    segments_df["duration_frames"] >= 5
]


# ==========================================
# Convert frames to seconds
# ==========================================

FPS = 30

segments_df["start_time"] = (
    segments_df["start_frame"] / FPS
)

segments_df["end_time"] = (
    segments_df["end_frame"] / FPS
)


# ==========================================
# Save
# ==========================================

os.makedirs(
    "output/data",
    exist_ok=True
)


segments_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==========================================
# Print readable output
# ==========================================

print()
print("==========================================")
print("DANCE MOVEMENT TIMELINE")
print("==========================================")

for _, row in segments_df.iterrows():

    print(
        f"{row['start_time']:.1f}s - "
        f"{row['end_time']:.1f}s  →  "
        f"{row['move']}"
    )


print()
print(f"Saved to: {OUTPUT_CSV}")