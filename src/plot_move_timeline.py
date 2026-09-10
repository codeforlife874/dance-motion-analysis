import pandas as pd
import matplotlib.pyplot as plt
import os


INPUT_CSV = "output/data/movement_segments.csv"

os.makedirs(
    "output/plots",
    exist_ok=True
)


df = pd.read_csv(INPUT_CSV)


# ==========================================
# Give each movement a number
# ==========================================

movement_types = sorted(
    df["move"].unique()
)


movement_to_number = {
    move: i
    for i, move in enumerate(
        movement_types
    )
}


df["movement_number"] = (
    df["move"]
    .map(movement_to_number)
)


# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(12, 6))


for _, row in df.iterrows():

    plt.plot(
        [
            row["start_time"],
            row["end_time"]
        ],
        [
            row["movement_number"],
            row["movement_number"]
        ],
        linewidth=8
    )


plt.yticks(
    range(len(movement_types)),
    movement_types
)


plt.xlabel("Time (seconds)")

plt.ylabel("Detected movement")

plt.title(
    "Dance Movement Timeline"
)

plt.grid(
    axis="x"
)

plt.tight_layout()


plt.savefig(
    "output/plots/dance_movement_timeline.png",
    dpi=200
)


plt.close()


print(
    "Dance movement timeline saved!"
)