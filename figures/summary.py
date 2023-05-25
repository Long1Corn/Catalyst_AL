import string

import matplotlib
import numpy as np
import pandas as pd
from chemformula import ChemFormula
from matplotlib import pyplot as plt

data = pd.read_csv(r"Results/Round_2/predictions.csv")
data = data[['Crystals', 'M']].values

result_lst = []

for item in data:
    item[0] = item[0].split(sep='_')[0]

all_comp = np.unique(data[:, 0])

for comp in all_comp:
    index = comp == data[:, 0]
    val = np.max(data[:, 1][index])

    chemical_formula = ChemFormula(comp)
    elements = chemical_formula.element
    num_atom = 0
    if len(elements.keys()) == 1:
        continue

    for item in elements.keys():
        num_atom = num_atom + elements[item]
        if item != 'Ni':
            atom = item
    percent = elements[atom] / num_atom

    result_lst.append([atom, percent, val])

result_lst = np.array(result_lst)

all_atom = np.unique(result_lst[:, 0])

x = np.array([np.where(all_atom == atom) for atom in result_lst[:, 0]]).flatten()
y = result_lst[:, 1].astype(np.float)
c = result_lst[:, 2].astype(np.float)

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams['axes.linewidth'] = 3

plt.figure(figsize=(20, 3), dpi=600)
plt.scatter(x, y, s=np.exp((c + 5.5)), c=c, cmap='rainbow', alpha=0.7, vmin=-2, vmax=0.5)
plt.xticks(ticks=np.arange(len(all_atom)), labels=all_atom)
plt.xlim(-1,36)
plt.xlabel('Elements')
plt.ylabel('Composition')

cb = plt.colorbar(ticks=[-2, -1.5, -1, -0.5, 0, 0.5],shrink=0.75)
cb.outline.set_linewidth(0)

plt.ylim(0, 1)
plt.yticks(ticks=[0, 0.5, 1])
plt.plot([6, 6], [0, 1], '--', c='gray', linewidth=1)
plt.plot([12, 12], [0, 1], '--', c='gray', linewidth=1)
plt.plot([18, 18], [0, 1], '--', c='gray', linewidth=1)
plt.plot([24, 24], [0, 1], '--', c='gray', linewidth=1)
plt.plot([30, 30], [0, 1], '--', c='gray', linewidth=1)



plt.show()
