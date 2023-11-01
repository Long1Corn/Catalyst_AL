from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.cif import CifWriter
import os
import pandas as pd

source_path = r'Cat_Data/Raw_Crystal_Data_All'
save_path = r'Cat_Data/Slab_All_Prediction'

CIF_files = [f for f in os.listdir(source_path) if f.endswith('.cif')]
num_files = len(CIF_files)
data_list = []

index = 0
slab_info = []

# create save_pth if not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i, CIF_file in enumerate(CIF_files):
    print("Converting {}/{}".format(i + 1, num_files))

    crystal = Structure.from_file(os.path.join(source_path, CIF_file))
    name = os.path.splitext(CIF_file)[0]
    slabs = generate_all_slabs(crystal, max_index=2, min_slab_size=10.0, min_vacuum_size=15.0)

    for slab in slabs:
        miller_index = slab.miller_index
        shift = slab.shift

        c = CifWriter(slab)
        c.write_file(os.path.join(save_path, f"{index}.cif"))

        index = index + 1

        data_list.append([f"{index}.cif", 0])
        slab_info.append([index, name, miller_index, shift])


data_list = pd.DataFrame(data_list)
data_list.to_csv(os.path.join(save_path, "id_prop.csv"), header=False, index=False)

slab_info = pd.DataFrame(slab_info)
slab_info.to_csv(os.path.join(save_path, "slab_info.csv"), header=["index", "name", "miller_index", "shift"],
                 index=False)
