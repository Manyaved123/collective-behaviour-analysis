"""
STEP 3: NETWORK ANALYSIS + COLLECTIVE BEHAVIOUR METRICS
===========================================================
Now we turn the trajectories into the two things your project is actually
about:

  A) AN INTERACTION NETWORK
     For a chosen frame, we connect two fish with an edge if they are
     close enough to "interact" (within INTERACTION_DISTANCE). This turns
     positions into a graph -- who's connected to whom.

  B) COLLECTIVE BEHAVIOUR METRICS (computed for every frame, over time)
     - POLARIZATION: how aligned everyone's direction is (0 = totally
       random directions, 1 = everyone facing exactly the same way)
     - COHESION: average distance to nearest neighbor (small = tightly
       packed group, large = spread out)

Output:
  - metrics_over_time.csv  (polarization & cohesion per frame)
  - metrics_plot.png       (how these change over the simulation)
  - network_plot.png       (the interaction network at the final frame)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("trajectories.csv")
INTERACTION_DISTANCE = 15  # same radius the fish used to "notice" each other

# ---------------------------------------------------------------
# PART A: compute POLARIZATION and COHESION for every frame
# ---------------------------------------------------------------
metric_rows = []
for frame, group in df.groupby("frame"):
    velocities = group[["vx", "vy"]].values
    # normalize each fish's velocity to just its direction (unit vector)
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    directions = velocities / speeds

    # POLARIZATION: magnitude of the average direction vector.
    # If everyone points the same way, the vectors add up to a long vector (~1).
    # If directions are random, they cancel out to near zero.
    avg_direction = directions.mean(axis=0)
    polarization = np.linalg.norm(avg_direction)

    # COHESION: for each fish, distance to its single nearest neighbor,
    # then average across all fish. Smaller = more tightly grouped.
    positions = group[["x", "y"]].values
    nearest_dists = []
    for i in range(len(positions)):
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf  # ignore distance to self
        nearest_dists.append(dists.min())
    cohesion = np.mean(nearest_dists)

    metric_rows.append({"frame": frame, "polarization": polarization, "avg_nearest_neighbor_dist": cohesion})

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv("metrics_over_time.csv", index=False)

print("=== Collective behaviour summary ===")
print(f"Average polarization across whole simulation: {metrics_df['polarization'].mean():.3f} (0=random, 1=perfectly aligned)")
print(f"Average nearest-neighbor distance: {metrics_df['avg_nearest_neighbor_dist'].mean():.2f} units")
print(f"Polarization went from {metrics_df['polarization'].iloc[0]:.3f} (frame 0) to {metrics_df['polarization'].iloc[-1]:.3f} (final frame)")

# plot metrics over time
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes[0].plot(metrics_df["frame"], metrics_df["polarization"], color="darkorange")
axes[0].set_ylabel("Polarization\n(0=random, 1=aligned)")
axes[0].set_title("How coordinated is the group over time?")
axes[0].set_ylim(0, 1.05)

axes[1].plot(metrics_df["frame"], metrics_df["avg_nearest_neighbor_dist"], color="steelblue")
axes[1].set_ylabel("Avg. nearest-neighbor\ndistance")
axes[1].set_xlabel("Frame")
axes[1].set_title("How tightly grouped is the group over time?")

plt.tight_layout()
plt.savefig("metrics_plot.png", dpi=150)
print("\nSaved metrics_plot.png")

# ---------------------------------------------------------------
# PART B: build and draw the interaction NETWORK at the final frame
# ---------------------------------------------------------------
last_frame = df["frame"].max()
snapshot = df[df["frame"] == last_frame].reset_index(drop=True)

G = nx.Graph()
for i, row in snapshot.iterrows():
    G.add_node(row["fish_id"], pos=(row["x"], row["y"]))

for i in range(len(snapshot)):
    for j in range(i + 1, len(snapshot)):
        dist = np.linalg.norm(snapshot.loc[i, ["x", "y"]].values - snapshot.loc[j, ["x", "y"]].values)
        if dist < INTERACTION_DISTANCE:
            G.add_edge(snapshot.loc[i, "fish_id"], snapshot.loc[j, "fish_id"], weight=dist)

print(f"\n=== Network summary (final frame) ===")
print(f"Nodes (fish): {G.number_of_nodes()}")
print(f"Edges (interactions): {G.number_of_edges()}")
print(f"Average degree (avg. # of neighbors per fish): {2*G.number_of_edges()/G.number_of_nodes():.2f}")
density = nx.density(G)
print(f"Network density: {density:.3f}")
if nx.is_connected(G):
    print("The whole group forms ONE connected network (everyone is linked, directly or indirectly).")
else:
    n_components = nx.number_connected_components(G)
    print(f"The group is split into {n_components} separate sub-groups (not fully connected).")

# draw the network using real x,y positions as node layout
pos = nx.get_node_attributes(G, "pos")
plt.figure(figsize=(7, 7))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500,
        edge_color="gray", font_size=9)
plt.title(f"Interaction network at frame {last_frame}\n(edge = fish within {INTERACTION_DISTANCE} units of each other)")
plt.savefig("network_plot.png", dpi=150)
print("Saved network_plot.png")
