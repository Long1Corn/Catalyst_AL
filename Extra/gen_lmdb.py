import concurrent
import os
import pdb
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool

import lmdb
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from torch_geometric.data import Data
from tqdm import tqdm


def time_it(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed: {end - start}")
        return result

    return wrapper


def get_CIF_data(pth_enegy_csv_lst, large_dif_id=None):
    energy_csv_lst = []
    for pth_energy_csv in pth_enegy_csv_lst:
        energy_csv_lst.append(pd.read_csv(pth_energy_csv))

    energy_csv = pd.concat(energy_csv_lst, axis=0, ignore_index=True)

    cif_id = energy_csv["cif_id"].values
    prime_struct = energy_csv["prime_struct"].values

    miller_str = [cif_id.tolist()[i].split('_')[1:] for i in range(len(cif_id))]
    miller_str = [''.join(miller_str[i]) for i in range(len(cif_id))]

    slab_cif_pth = [os.path.join(CIF_pth, f"{prime_struct[i]}_{miller_str[i]}.cif") for i in range(len(cif_id))]

    energy_csv["CIF_pth"] = slab_cif_pth

    return energy_csv


def process_item(i: int, dataset):
    """
    Process a dataset item given its index and return a PyTorch Geometric Data object
    or None if there was an error.

    Args:
        i (int): Index of the item in the dataset.
        dataset (Dataset): The dataset containing the item.

    Returns:
        Union[Data, None]: A PyTorch Geometric Data object containing the item's
                           information or None if there was an error.
    """

    try:
        # Retrieve features, neighbor features, neighbor feature indices, and target
        data = dataset
        CIF_pth = data["CIF_pth"]
        y = float(data["Energy_barrier"])
        sid = data["cif_id"]
        frame_number = 0

        # Read the CIF file
        crystal = Structure.from_file(CIF_pth)

        natoms = len(crystal)

        pos = crystal.cart_coords
        pos = torch.Tensor(pos.tolist())

        # indices of atoms in the top layer
        atom_pooling_index = []
        atom_coords_z = crystal.frac_coords[:, 2]
        max1_z = np.max(atom_coords_z)

        max2_z = 1
        try:
            max2_z = np.max(atom_coords_z[atom_coords_z < (max1_z * 0.98)])
        except:
            pass

        max_z = min(max1_z, max2_z)

        for i, coord in enumerate(atom_coords_z):
            if coord >= max_z * 0.95:
                atom_pooling_index.append(i)

        tags = torch.zeros(natoms)
        tags[atom_pooling_index] = 1

        fixed = torch.ones(natoms)
        fixed[atom_pooling_index] = 0

        lattice = crystal.lattice.matrix
        lattice = torch.Tensor([lattice.tolist()])

        # Calculate atom features
        atom_fea = list(crystal.atomic_numbers)
        atom_fea = torch.tensor(atom_fea).squeeze()

        force = torch.zeros_like(pos)

        data = Data(y=y, pos=pos, cell=lattice, atomic_numbers=atom_fea, natoms=natoms, tags=tags,
                    fixed=fixed, sid=sid, fid=int(frame_number)
                    , force=force)

    except Exception as e:
        print(f"Error processing item {i}: {e}")
        return None

    # Create a PyTorch Geometric Data object from the retrieved information
    return data


def process_item_wrapper(args):
    index, dataset = args

    return process_item(index, dataset)


def gen_lmdb(dataset, DB_path: str, map_size: int = 10 ** 9):
    print(f"save to {DB_path}")

    qty = len(dataset)
    db_syn_freq = 10000

    # Open LMDB environment
    db = lmdb.open(DB_path, map_size=map_size, subdir=False, meminit=False, map_async=True)
    txn = db.begin(write=True)

    db_idx = 0  # This counter will ensure continuous indices in the LMDB database

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all the tasks and get a list of futures

        futures = [executor.submit(process_item_wrapper, (i, dataset.iloc[i])) for i in range(qty)]

        # Use tqdm to create a progress bar
        with tqdm(total=qty, desc="Processing items", smoothing=0.1) as pbar:
            for future in concurrent.futures.as_completed(futures):
                data_value = future.result()

                if data_value is not None:
                    txn.put(key=f"{db_idx}".encode("ascii"), value=pickle.dumps(data_value, protocol=-1))
                    db_idx += 1

                if db_idx % db_syn_freq == 0:
                    txn.commit()
                    db.sync()
                    txn = db.begin(write=True)

                # Update the progress bar
                pbar.update(1)

    # Finalize the database
    txn.commit()
    db.sync()
    txn = db.begin(write=True)
    txn.put(key='length'.encode("ascii"), value=pickle.dumps(db_idx, protocol=-1))
    txn.commit()
    db.sync()
    db.close()


if __name__ == "__main__":
    pth_energy_lst = [r"Extra\R2_modified.csv"
                      ]

    DB_pth = r'Extra\dataset\R2_barrier\R2_barrier.lmdb'
    CIF_pth = r"Cat_Data\Slab_All"

    CIF_data = get_CIF_data(pth_energy_lst)
    gen_lmdb(CIF_data, DB_pth, map_size=1 * 10 ** 7)
