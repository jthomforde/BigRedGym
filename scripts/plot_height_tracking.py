"""Plot commanded vs. actual base height from logs/height_tracking.csv."""

import matplotlib.pyplot as plt
import numpy as np

time, commanded, actual = np.loadtxt(
    "logs/height_tracking.csv", delimiter=",", skiprows=1, unpack=True
)
error_cm = (actual - commanded) * 100

fig, (ax_height, ax_error) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
ax_height.plot(time, commanded, color="#898781", linestyle="--", label="commanded")
ax_height.plot(time, actual, color="#2a78d6", label="actual")
ax_height.set_ylabel("base height [m]")
ax_height.legend(loc="upper right")

ax_error.plot(time, error_cm, color="#2a78d6")
ax_error.axhline(0.0, color="#898781", linewidth=0.8)
ax_error.set_ylabel("error [cm]")
ax_error.set_xlabel("time [s]")

fig.tight_layout()
fig.savefig("logs/height_tracking.pdf")