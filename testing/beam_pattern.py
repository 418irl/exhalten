import numpy as np
import matplotlib.pyplot as plt
import os

path = r"C:\.....\name.csv"
assert os.path.exists(path), f"File not found: {path}" # for testing only since path broke previously

angle_deg = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0)
voltage   = np.loadtxt(path, delimiter=",", skiprows=1, usecols=2)  # column 2, not 3

angle_shifted = angle_deg - 90   # remove this line if your boresight really is at 0°

voltage_norm = voltage / np.max(voltage) #normalised voltage bc how much weaker each angle is compared to the strongest direction is imp not the absolute voltage
level_db = 20 * np.log10(np.abs(voltage_norm)) 
level_db_clipped = np.clip(level_db, -40, 0)
angle_rad = np.deg2rad(angle_shifted)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
ax.plot(angle_rad, level_db_clipped, linewidth=2, marker='o')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1) #for plotting in clockwise direction
ax.set_rlim(-40, 0)
ax.set_rlabel_position(135) #all labels plotted at 135 degrees
ax.set_title("Echosounder Beam Pattern", va='bottom')
plt.tight_layout()
plt.show()
