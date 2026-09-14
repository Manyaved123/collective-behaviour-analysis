"""
STEP 2: VISUALIZE THE TRAJECTORIES
====================================
This draws the path each fish took over the whole simulation (like leaving
a trail behind them), plus a snapshot of where they all are at the very end
with an arrow showing which way each is facing. This is the first check that
your simulation actually shows "collective" behaviour -- the paths should
look like they're moving together, not randomly scattered.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv("/home/claude/project/trajectories.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- LEFT PLOT: full trails of every fish ----
ax = axes[0]
colors = cm.viridis(df["fish_id"].unique() / df["fish_id"].max())
for fish_id, color in zip(df["fish_id"].unique(), colors):
    fish_data = df[df["fish_id"] == fish_id]
    ax.plot(fish_data["x"], fish_data["y"], alpha=0.6, linewidth=1, color=color)
ax.set_title("Full movement trails of every fish")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# ---- RIGHT PLOT: snapshot of the final frame, with direction arrows ----
ax = axes[1]
last_frame = df["frame"].max()
snapshot = df[df["frame"] == last_frame]
ax.quiver(snapshot["x"], snapshot["y"], snapshot["vx"], snapshot["vy"],
          angles="xy", scale_units="xy", scale=0.3, width=0.006, color="steelblue")
ax.set_title(f"Snapshot at frame {last_frame} (arrows = direction)")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("/home/claude/project/trails_and_snapshot.png", dpi=150)
print("Saved plot to trails_and_snapshot.png")
