import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data_path = r'Results/Round_1/R1_mean_std.csv'
data = pd.read_csv(data_path)

mean = data['M'].values
std = data['std'].values

x = np.arange(len(mean))
ub = mean + 2 * std
lb = mean - 2 * std

# change font to arial
matplotlib.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.fill_between(x, ub, lb, alpha=0.4, label="95% confident level")
ax.plot(x, mean, linewidth=3, label='Mean value')
ax.set_ylim(-0.6, 0.4)
ax.set_xlim(0, 1500)
ax.xaxis.set_ticks([0, 500, 1000, 1500])
ax.plot([0, 3000], [0.08, 0.08], '--', alpha=0.4, label="Reference performance")

plt.legend(loc='lower left')

ax.patch.set_linewidth(5)
plt.show(dpi=1200)
