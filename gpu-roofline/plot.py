#!/usr/bin/env python3


import os
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import sys

sys.path.append("..")
from device_order import *


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 9))
# fig2, ax2 = plt.subplots(figsize=(8, 4))
# fig3, ax3 = plt.subplots(figsize=(8, 4))


filenames =["Tesla-T4.txt", "Tesla-T4-freq450.txt", "Tesla-T4-freq600.txt", "Tesla-T4-freq750.txt", "Tesla-T4-freq900.txt", "Tesla-T4-freq1050.txt", "Tesla-T4-freq1200.txt", "Tesla-T4-freq1350.txt", "Tesla-T4-freq1500.txt"] #["h200.txt", "alex_a100_40.txt", "genoa_l40.txt"]

# Light green for original, then light blue to deep blue gradient for frequency series
# Dark green for Tesla-T4.txt (original)
light_green = "#006400"

# Create gradient from light blue to deep blue for 8 frequency series
# Using matplotlib's colormap to generate smooth gradient
blue_colors = plt.cm.Blues(np.linspace(0.3, 0.95, 8))  # From light (0.3) to deep (0.95)
blue_hex = [mcolors.rgb2hex(c) for c in blue_colors]

colors = [light_green] + blue_hex

c = 0
for filename in filenames:
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)

        datapoints = [[]]

        for row in csvreader:
            print(row)
            if len(row) == 0:
                datapoints.append([])

            elif len(row) == 16:
                # Extract temperature (row[15] has format like "57°C")
                temp_str = row[15].replace('°C', '').strip()
                datapoints[-1].append(
                    [float(row[5]), float(row[9]), float(row[13]), float(row[11]), float(temp_str)]
                )

        print(datapoints)
        print()

        for i in range(min(1, len(datapoints[1]))):
            print([d[i][1] for d in datapoints if len(d) > 0])
            ax1.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][1] / 1000 for d in datapoints if len(d) > 0],
                "-",
                color=colors[c],
                label=filename
            )

            ax2.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][2] for d in datapoints if len(d) > 0],
                "-",
                color=colors[c],
            )

            ax3.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][3] / 1000 for d in datapoints if len(d) > 0],
                "-",
                color=colors[c],
            )

            ax4.plot(
                [d[i][0] for d in datapoints if len(d) > 0],
                [d[i][4] for d in datapoints if len(d) > 0],
                "-",
                color=colors[c],
            )
            c += 1


ax1.legend()

ax4.set_xlabel("Arithmetic Intensity, Flop/B")
ax1.set_ylabel("TFlop/s")
ax2.set_ylabel("Power, W")
ax3.set_ylabel("Clock, GHz")
ax4.set_ylabel("Temperature, °C")


ax1.set_ylim([0, ax1.get_ylim()[1]])
ax1.set_xlim([0, ax1.get_xlim()[1]])

ax2.set_ylim([0, ax2.get_ylim()[1]])
ax2.set_xlim([0, ax2.get_xlim()[1]])

ax3.set_ylim([0, ax3.get_ylim()[1]])
ax3.set_xlim([0, ax3.get_xlim()[1]])

ax4.set_ylim([0, ax4.get_ylim()[1]])
ax4.set_xlim([0, ax4.get_xlim()[1]])

# ax.set_xscale("log")
# ax2.set_xscale("log")

# ax.set_yscale("log")
# ax2.set_yscale("log")


fig.tight_layout()

#plt.savefig("L40_plot.pdf", dpi=4000)
plt.savefig("roofline_comparison_T4_original_freq450-1500.png")
plt.show()
