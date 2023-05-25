import os
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

data_pth = r'AL/Cat_Data/Slab_All'
data_lst = []

for file in os.listdir(data_pth):
    if file.endswith(".cif"):
        data_lst.append(file)

atom_lst = []
miller_lst = []

for data in data_lst:
    x = re.split('_|\\.', data)
    formula = x[0]
    mpid = x[1]
    miller = x[3]

    atom = 'Ni'
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    for element in elements:
        if element[0] != 'Ni':
            atom = element[0]
            break
    atom_lst.append(atom)

    miller_2 = re.findall(r'-*[0-9]', miller)
    miller_2 = [int(idx) for idx in miller_2]
    miller = sum(np.abs(miller_2))

    miller_lst.append(miller)

labels, counts = np.unique(atom_lst, return_counts=True)
ticks = range(len(counts))

figure(figsize=(10, 4), dpi=600)
plt.bar(ticks, counts, align='center')
plt.xticks(ticks, labels)
plt.title("Design Space Summary: Composition")
plt.show()

labels, counts = np.unique(miller_lst, return_counts=True)
ticks = range(len(counts))

figure(figsize=(5, 2), dpi=600)
plt.bar(ticks, counts, align='center')
plt.xticks(ticks, labels)
plt.title("Design Space Summary: Sum of Miller Index")
plt.show()
