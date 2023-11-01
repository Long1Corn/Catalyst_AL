import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Example data
descriptor = [-0.61, -0.16, 0.18, 0.33, 0.17, -0.17, 0.08, 0.39, 0.04, -0.13, 0.15, -0.23]
C2H4_selectivity = [-220, -120, 100, 55, -15, -145, 70, 65, 95, -135, 85, 40]
conversion = [	100,	100,	100,	50,	100,	100,	100,	60,	30,	100,	100,	40]

C2H4_selectivity = np.array(C2H4_selectivity)*np.array(conversion)/100

catalysts = ["Ni", "Ni$_{2}$In", "NiIn", "Ni$_{2}$In$_{3}$", "Ni$_{13}$In$_{9}$", "Ni$_{5}$Ga$_{3}$", "NiGa",
             "Ni$_{2}$Ga$_{3}$", "NiSb", "Ni$_{3}$Sn", "Ni$_{3}$Sn$_{2}$",
             "Ni$_{3}$Sn$_{4}$", ]

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.linewidth'] = 2

fig, ax = plt.subplots(dpi=1200)
# Normalize the y values to range between 0 and 1
norm_y = (C2H4_selectivity - np.min(C2H4_selectivity)) / (np.max(C2H4_selectivity) - np.min(C2H4_selectivity))

# Use a colormap to get a color for each y value
colors = plt.cm.viridis(norm_y)

# Plot the scatter points
ax.scatter(descriptor, C2H4_selectivity, s=200, color=colors, alpha=0.7)  # s is the size of the scatter points

# Label each point
for i, label in enumerate(catalysts):
    ax.text(descriptor[i], C2H4_selectivity[i], label, ha='right', va='bottom', weight='bold', size=11,
            color="black", )

z = np.polyfit(descriptor, C2H4_selectivity, 1)
p = np.poly1d(z)
ax.plot([min(descriptor), max(descriptor)], [p(min(descriptor)), p(max(descriptor))],
        "--", color=[0.25, 0.25, 0.25], linewidth=3, alpha=0.7)

plt.xlabel('Descriptor value (eV)')
plt.ylabel('C$_{2}$H$_{4}$ selectivity (%)')
plt.title("Experiment validated performance")
plt.tight_layout()

plt.show()
