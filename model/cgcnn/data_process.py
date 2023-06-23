from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_data_loader(dataset, collate_fn=default_collate,
                    batch_size=64, num_workers=1, pin_memory=False, ):
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=collate_fn, pin_memory=pin_memory)

    return loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    atom_pooling_index_list = []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, atom_pooling_index), target, cif_id) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)

        atom_pooling_index_list.append(torch.LongTensor(atom_pooling_index + base_idx))

        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            atom_pooling_index_list), \
        torch.stack(batch_target, dim=0), \
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(self, label_dir=None, properties=[], struct_dir=None,
                 aug=False, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):

        self.label_dir = label_dir
        self.properties = properties
        self.struct_dir = struct_dir
        self.aug = aug

        self.max_num_nbr, self.radius = max_num_nbr, radius
        # wd = os.getcwd()

        assert os.path.exists(label_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.label_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

        self.id_prop_data = pd.read_csv(id_prop_file)

        self.id_prop_data = self.id_prop_data.sample(frac=1, random_state=23)

        atom_init_file = os.path.join(self.label_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=1000)  # Cache loaded structures
    def get_strcut(self, path):
        crystal = Structure.from_file(path)
        return crystal

    def __getitem__(self, idx):
        # cif_id, target, prime_struct = self.id_prop_data[idx]
        cif_id = self.id_prop_data["cif_id"].values[idx]
        prime_struct = self.id_prop_data["prime_struct"].values[idx]
        targets = self.id_prop_data[self.properties].values[idx, :]

        struct_path = os.path.join(self.struct_dir, prime_struct)
        crystal = self.get_strcut(struct_path + ".cif")

        if self.aug:
            miller_str = cif_id.split('_')[1:]
            miller_idx = [int(a) for a in miller_str]

            slab_gen = SlabGenerator(crystal, miller_index=miller_idx, min_slab_size=10.0, min_vacuum_size=15.0)
            slab = slab_gen.get_slab()

            aug_x = 3
            aug_y = 3
            aug_z = np.random.randint(1, 3)
            slab.make_supercell([[aug_x, 0, 0], [0, aug_y, 0], [0, 0, aug_z]])
        else:
            slab = crystal

        atom_pooling_index = []
        atom_coords_z = slab.frac_coords[:, 2]
        max1_z = np.max(atom_coords_z)

        max2_z = 1
        try:
            max2_z = np.max(atom_coords_z[atom_coords_z < (max1_z * 0.98)])
        except:
            print("{} has only one layer".format(cif_id))

        max_z = min(max1_z, max2_z)
        # max_z = max1_z

        for i, coord in enumerate(atom_coords_z):
            if coord >= max_z * 0.95:
                atom_pooling_index.append(i)

        atom_pooling_index = np.array(atom_pooling_index)

        atom_fea = np.vstack([self.ari.get_atom_fea(slab[i].specie.number)
                              for i in range(len(slab))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = slab.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x.index,
                                            nbr[:self.max_num_nbr])))

                nbr_fea.append(list(map(lambda x: x.nn_distance,
                                        nbr[:self.max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        # nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea).unsqueeze(-1)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.tensor(targets, dtype=torch.float32)
        return (atom_fea, nbr_fea, nbr_fea_idx, atom_pooling_index), target, cif_id
