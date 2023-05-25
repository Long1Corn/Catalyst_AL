from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matplotlib import pyplot as plt
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp.inputs import Poscar
import os
import pandas as pd

source_path = r'Cat_Data/Raw_Crystal_Data_All'
save_path = r'Cat_Data/Slab_All'

CIF_files = [f for f in os.listdir(source_path) if f.endswith('.cif')]
num_files = len(CIF_files)
data_list = []

for i, CIF_file in enumerate(CIF_files):
    print("Converting {}/{}".format(i + 1, num_files))

    crystal = Structure.from_file(os.path.join(source_path, CIF_file))
    name = os.path.splitext(CIF_file)[0]
    slabs = generate_all_slabs(crystal, max_index=2, min_slab_size=10.0, min_vacuum_size=15.0)

    index_store = []

    for slab in slabs:
        miller_index = slab.miller_index
        index = ''.join(map(str, miller_index))

        if index not in index_store:
            index_store.append(index)
        else:
            continue

        slab.make_supercell([[3, 0, 0], [0, 3, 0], [0, 0, 1]])

        # crystal_facets = [slab for slab in slabs if slab.miller_index == (1, 1, 1)][0]

        c = CifWriter(slab)

        file_name = r"{}_{}".format(name, index)
        c.write_file(os.path.join(save_path, file_name + ".cif"))

        data_list.append([file_name+".cif", 0])

data_list = pd.DataFrame(data_list)
data_list.to_csv(os.path.join(save_path, "id_prop.csv"), header=False, index=False)
