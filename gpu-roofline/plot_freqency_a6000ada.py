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


filenames =["A6000ada.txt", "A6000ada-freq450.txt", "A6000ada-freq600.txt", "A6000ada-freq750.txt", "A6000ada-freq900.txt", "A6000ada-freq1050.txt", "A6000ada-freq1200.txt","A6000ada-freq1350.txt", "A6000ada-freq1500.txt", "A6000ada-freq1650.txt", "A6000ada-freq1800.txt", "A6000ada-freq1950.txt", "A6000ada-freq2100.txt", "A6000ada-freq2250.txt", "A6000ada-freq2400.txt","A6000ada-freq2550.txt", "A6000ada-freq2700.txt", "A6000ada-freq2850.txt", "A6000ada-freq3000.txt"] # , "A6000ada-freq900.txt", "A6000ada-freq1050.txt", "A6000ada-freq1200.txt", "A6000ada-freq1350.txt", "A6000ada-freq1500.txt", "A6000ada-freq1650.txt", "A6000ada-freq1800.txt", "A6000ada-freq1950.txt", "A6000ada-freq2100.txt", "A6000ada-freq2250.txt", "A6000ada-freq2400.txt", "A6000ada-freq2550.txt", "A6000ada-freq2700.txt", "A6000ada-freq2850.txt", "A6000ada-freq3000.txt"

# Light green for original, then light blue to deep blue gradient for frequency series
# Dark green for Tesla-T4.txt (original)
light_green = "#006400"

# Create gradient from light blue to deep blue for frequency series
# Using matplotlib's colormap to generate smooth gradient
num_freq_files = len(filenames) - 1  # Exclude the first file (A6000ada.txt)
blue_colors = plt.cm.Blues(np.linspace(0.3, 0.95, num_freq_files))  # From light (0.3) to deep (0.95)
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

        # Find sections with data
        sections_with_data = [d for d in datapoints if len(d) > 0]
        
        if len(sections_with_data) == 0:
            print(f"Skipping {filename}: no data found")
            continue
        
        # Plot data from the first section (or aggregate across sections)
        # Get unique arithmetic intensity values (x-axis) from all sections
        x_values = []
        y1_values = []  # TFlop/s
        y2_values = []  # Power
        y3_values = []  # Clock
        y4_values = []  # Temperature
        
        # Collect data from all sections, using the first data point of each section
        for section in sections_with_data:
            if len(section) > 0:
                # Use the first data point from each section
                x_values.append(section[0][0])  # Arithmetic Intensity
                y1_values.append(section[0][1] / 1000)  # TFlop/s
                y2_values.append(section[0][2])  # Power
                y3_values.append(section[0][3] / 1000)  # Clock
                y4_values.append(section[0][4])  # Temperature
        
        if len(x_values) > 0:
            ax1.plot(x_values, y1_values, "-", color=colors[c], label=filename)
            ax2.plot(x_values, y2_values, "-", color=colors[c])
            ax3.plot(x_values, y3_values, "-", color=colors[c])
            ax4.plot(x_values, y4_values, "-", color=colors[c])
        
        c += 1


ax1.legend(fontsize=6, ncol=2, frameon=True)

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
plt.savefig("roofline_comparison_A6000ada_original_freq450-3000.png")
plt.show()
