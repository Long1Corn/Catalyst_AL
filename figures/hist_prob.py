import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data_path = r'Results/Round_2/R2_prob.csv'
data = pd.read_csv(data_path)

prob = data['CDF'].values
prob = prob[prob > 0.01]

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

x1 = ((0.01 < prob) & (prob < 0.5)).sum()
x2 = ((0.5 < prob) & (prob < 0.8)).sum()
x3 = ((0.8 < prob) & (prob < 1.0)).sum()

x = [x1, x2, x3]

x_label = ['1%', '50%', '80%']

ax.bar(x_label, height=x, color='silver')
ax.set_xticklabels(x_label, fontsize=18)
ax.set_yticklabels([], fontsize=15)

for i in range(3):
    plt.text(i, x[i] - 4, x[i], ha='center', fontsize=20)

plt.show(dpi=600)
