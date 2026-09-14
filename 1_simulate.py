"""
STEP 1: SIMULATE A GROUP OF "FISH" (BOIDS MODEL)
===================================================
This creates N fake fish, each starting at a random position with a random
direction. Every fish follows 3 simple rules, checked against its nearby
neighbors only:

  1. SEPARATION - steer away from neighbors that are too close (avoid crashing)
  2. ALIGNMENT  - steer toward the average direction of nearby neighbors
  3. COHESION   - steer toward the average position of nearby neighbors

No fish is "in charge." Yet when you run this, the group starts moving
together in organized ways -- that IS collective behaviour, emerging from
individual rules. This is the same core idea Craig Reynolds used in 1986
to create "Boids," and the same idea behind real fish schooling.

Output: trajectories.csv -- one row per fish per frame, with its x,y position
"""

import numpy as np
import pandas as pd

# ---- SETTINGS you can play with ----
N_FISH = 20          # how many individuals
N_FRAMES = 300        # how many time steps to simulate
WORLD_SIZE = 100      # the fish move within a 100x100 square
NEIGHBOR_RADIUS = 15  # fish only "notice" other fish within this distance
MAX_SPEED = 1.5
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
SEPARATION_DISTANCE = 5  # if closer than this, actively push apart

np.random.seed(42)  # so results are reproducible

# ---- Initialize fish with random positions and random directions ----
positions = np.random.uniform(0, WORLD_SIZE, size=(N_FISH, 2))
angles = np.random.uniform(0, 2 * np.pi, size=N_FISH)
velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1) * MAX_SPEED

records = []  # we'll collect every fish's position at every frame here

for frame in range(N_FRAMES):
    new_velocities = velocities.copy()

    for i in range(N_FISH):
        # find neighbors: other fish within NEIGHBOR_RADIUS of fish i
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        neighbor_mask = (dists > 0) & (dists < NEIGHBOR_RADIUS)

        if np.any(neighbor_mask):
            # RULE 1: SEPARATION - push away from very-close neighbors
            close_mask = neighbor_mask & (dists < SEPARATION_DISTANCE)
            if np.any(close_mask):
                separation = -diffs[close_mask].sum(axis=0)
            else:
                separation = np.zeros(2)

            # RULE 2: ALIGNMENT - match average direction of neighbors
            alignment = velocities[neighbor_mask].mean(axis=0)

            # RULE 3: COHESION - move toward average position of neighbors
            center_of_neighbors = positions[neighbor_mask].mean(axis=0)
            cohesion = center_of_neighbors - positions[i]

            # combine all 3 rules
            steer = (SEPARATION_WEIGHT * separation
                     + ALIGNMENT_WEIGHT * alignment
                     + COHESION_WEIGHT * cohesion)

            new_velocities[i] = velocities[i] + 0.1 * steer

        # keep speed constant (like real fish swimming at roughly steady pace)
        speed = np.linalg.norm(new_velocities[i])
        if speed > 0:
            new_velocities[i] = new_velocities[i] / speed * MAX_SPEED

    velocities = new_velocities
    positions = positions + velocities

    # wrap around the edges of the world (like a torus) so fish don't just leave
    positions = positions % WORLD_SIZE

    # record this frame for every fish
    for i in range(N_FISH):
        records.append({
            "frame": frame,
            "fish_id": i,
            "x": positions[i, 0],
            "y": positions[i, 1],
            "vx": velocities[i, 0],
            "vy": velocities[i, 1],
        })

df = pd.DataFrame(records)
df.to_csv("/home/claude/project/trajectories.csv", index=False)
print(f"Done! Simulated {N_FISH} fish over {N_FRAMES} frames.")
print(f"Saved {len(df)} rows to trajectories.csv")
print(df.head(10))
