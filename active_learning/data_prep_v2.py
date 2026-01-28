
from typing import Any
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections import OrderedDict
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import torch
import h5py
from config import ROOT_DIR
from active_learning.utils import molecular_graph_featurizer as smiles_to_graph, smiles_to_ecfp, get_tanimoto_matrix, check_featurizability, to_torch_dataloader


allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def resolve_column_name(df: pd.DataFrame, column: str, path: str) -> str:
    if column in df.columns:
        return column
    matches = [col for col in df.columns if col.lower() == column.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Column '{column}' is ambiguous in {path}. Matches: {matches}")
    raise KeyError(f"Column '{column}' not found in {path}. Available columns: {list(df.columns)}")


def canonicalize(smiles: str, sanitize: bool = True):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=sanitize))


def get_data(random_state: int = 42, dataset: str = 'ALDH1'):

    # read smiles from file and canonicalize them
    with open(os.path.join(ROOT_DIR, f'data/{dataset}/original/inactives.smi')) as f:
        inactives = [canonicalize(smi.strip().split()[0]) for smi in f.readlines()]
    with open(os.path.join(ROOT_DIR, f'data/{dataset}/original/actives.smi')) as f:
        actives = [canonicalize(smi.strip().split()[0]) for smi in f.readlines()]

    # remove duplicates:
    inactives = list(set(inactives))
    actives = list(set(actives))

    # remove intersecting molecules:
    intersecting_mols = np.intersect1d(inactives, actives)
    inactives = [smi for smi in inactives if smi not in intersecting_mols]
    actives = [smi for smi in actives if smi not in intersecting_mols]

    # remove molecules that have scaffolds that cannot be kekulized or featurized
    inactives_, actives_ = [], []
    for smi in tqdm(actives):
        try:
            if Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False)) is not None:
                if check_featurizability(smi):
                    actives_.append(smi)
        except:
            pass
    for smi in tqdm(inactives):
        try:
            if Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False)) is not None:
                if check_featurizability(smi):
                    inactives_.append(smi)
        except:
            pass

    # add to df
    df = pd.DataFrame({'smiles': inactives_ + actives_,
                       'y': [0] * len(inactives_) + [1] * len(actives_)})

    # shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame, random_state: int = 42, screen_size: int = 50000, test_size: int = 10000,
               dataset: str = 'ALDH1') -> (pd.DataFrame, pd.DataFrame):

    from sklearn.model_selection import train_test_split
    df_screen, df_test = train_test_split(df, stratify=df['y'].tolist(), train_size=screen_size, test_size=test_size,
                                          random_state=random_state)

    # write to csv
    df_screen.to_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/screen.csv'), index=False)
    df_test.to_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/test.csv'), index=False)

    return df_screen, df_test

def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    # conformer = mol.GetConformers()[0]
    # positions = conformer.GetPositions()
    # positions = torch.Tensor(positions)

    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return x, edge_index, edge_attr

class MasterDataset2Labeled: # Test가 아닌경우
    """Test가 아닌경우"""
    """ Dataset that holds all data in an indexable way """
    def __init__(self, name: str, representation: str = 'ecfp', feature = '',
                 input='./data/input.csv', assay_active = None, assay_inactive = None, input_val_col='y', input_smiles_col='smiles', is_reverse=False) -> None:

        assert representation in ['ecfp', 'graph', 'scaffold'], f"'representation' must be 'ecfp' or 'graph', not {representation}"
        self.mode = name
        self.representation = representation
        self.pth = input
        self.assay_active = assay_active
        self.assay_inactive = assay_inactive
        self.input_val_col = input_val_col
        self.input_smiles_col = input_smiles_col
        self.is_reverse = is_reverse

        self.smiles, self.x, self.y, self.graph = self.load()

    
    def load(self):

        print('Loading data ... ', flush=True, file=sys.stderr)

        csv = pd.read_csv(self.pth)
        smiles_col = resolve_column_name(csv, self.input_smiles_col, self.pth)
        smiles = np.array(csv[smiles_col])
        x = smiles_to_ecfp(smiles, silent=False)
        if self.mode != 'test' and self.assay_active is not None:
            csv.loc[csv[self.input_val_col].isin(self.assay_active), self.input_val_col] = 1
            csv.loc[csv[self.input_val_col].isin(self.assay_inactive), self.input_val_col] = 0
            csv[self.input_val_col] = csv[self.input_val_col].astype(int)
        
        elif self.mode != 'test' and self.assay_active is None:
            csv[self.input_val_col] = csv[self.input_val_col].replace([np.inf, -np.inf], np.nan)
            mu = csv[self.input_val_col].mean()
            sigma = csv[self.input_val_col].std(ddof=0)   # population std (권장)

            csv[self.input_val_col] = (csv[self.input_val_col] - mu) / sigma
            if self.is_reverse:
                csv[self.input_val_col] = -csv[self.input_val_col]

        y = np.array(csv[self.input_val_col])

        # graph feature 추출
        graphs = [smiles_to_graph(smi, y=y) for smi, y in tqdm(zip(smiles, y))]
        graph_list = []

        for i, graph in tqdm(enumerate(graphs)):
            graph.fp = torch.tensor([x[i]], dtype=torch.float32)

            mol = Chem.MolFromSmiles(graph.smiles, sanitize=True)
            xp, edgep_index, edgep_attr = mol_to_graph_data_obj_simple_3D(mol)
            graph.xp = xp
            graph.edgep_index = edgep_index
            graph.edgep_attr = edgep_attr

            graph_list.append(graph)

        return smiles, x, y, graph_list
    
    def get_dataloader(self, batch_size=64, shuffle=False, pin_memory=True):
        if self.representation == 'ecfp':
            return to_torch_dataloader(self.x, self.y, self.assay_active is not None, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        else:
            return to_torch_dataloader(self.graph, self.y, self.assay_active is not None, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    def __len__(self) -> int:
        return len(self.smiles)

    def all(self):
        return self[range(len(self.smiles))]

    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]
        if self.representation == 'ecfp':
            return self.x[idx], self.y[idx], self.smiles[idx], self.smiles[idx]
        elif self.representation == 'graph':
            return self.x[idx], self.y[idx], self.smiles[idx], self.graph[idx]
        
    def get_data(self, idx):
        if type(idx) is int:
            idx = [idx]
        if self.representation == 'ecfp':
            return self.fp[idx], self.y[idx], self.smiles[idx]
        elif self.representation == 'graph':
            return [self.graph[i] for i in idx], self.y[idx], self.smiles[idx]


class MasterDataset2Unlabeled: # Test인 경우
    """Test인 경우"""
    """ Dataset that holds all data in an indexable way """
    def __init__(self, name: str, representation: str = 'ecfp', feature = '',
                 input='./data/input.csv', assay_active = None, assay_inactive = None, input_unlabel_val_col='score', input_unlabel_smiles_col='smiles') -> None:

        assert representation in ['ecfp', 'graph', 'scaffold'], f"'representation' must be 'ecfp' or 'graph', not {representation}"
        self.mode = name
        self.representation = representation
        self.pth = input
        self.input_unlabel_val_col = input_unlabel_val_col
        self.input_unlabel_smiles_col = input_unlabel_smiles_col
        self.assay_active = assay_active
        self.assay_inactive = assay_inactive

        self.smiles, self.x, self.y, self.graph = self.load()
    
    def load(self):
        csv = pd.read_csv(self.pth)
        smiles_col = resolve_column_name(csv, self.input_unlabel_smiles_col, self.pth)
        smiles = np.array(csv[smiles_col])

        x = smiles_to_ecfp(smiles, silent=False)
        if self.mode != 'test' and self.assay_active is not None:
            csv.loc[csv[self.input_val_col].isin(self.assay_active), self.input_val_col] = 1
            csv.loc[csv[self.input_val_col].isin(self.assay_inactive), self.input_val_col] = 0
            csv[self.input_val_col] = csv[self.input_val_col].astype(int)
        y = np.zeros(len(csv), dtype=float)

        # graph feature 추출
        graphs = [smiles_to_graph(smi, y=y) for smi, y in tqdm(zip(smiles, y))]
        graph_list = []

        for i, graph in tqdm(enumerate(graphs)):
            graph.fp = torch.tensor([x[i]], dtype=torch.float32)

            mol = Chem.MolFromSmiles(graph.smiles, sanitize=True)
            xp, edgep_index, edgep_attr = mol_to_graph_data_obj_simple_3D(mol)
            graph.xp = xp
            graph.edgep_index = edgep_index
            graph.edgep_attr = edgep_attr

            graph_list.append(graph)

        return smiles, x, y, graph_list
    
    def get_dataloader(self, batch_size=64, shuffle=False, pin_memory=True):
        if self.representation == 'ecfp':
            return to_torch_dataloader(self.x, self.y, self.assay_active is not None, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
        else:
            return to_torch_dataloader(self.graph, self.y, self.assay_active is not None, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    def __len__(self) -> int:
        return len(self.smiles)

    def all(self):
        return self[range(len(self.smiles))]

    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]
        if self.representation == 'ecfp':
            return self.x[idx], self.y[idx], self.smiles[idx], self.smiles[idx]
        elif self.representation == 'graph':
            return self.x[idx], self.y[idx], self.smiles[idx], self.graph[idx]

def smi_to_scaff(smiles: str, includeChirality: bool = False):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=includeChirality)


def similarity_vectors(df_screen, df_test, root: str = 'data', dataset: str = 'ALDH1'):

    print("Computing Tanimoto matrix for all test molecules")
    S = get_tanimoto_matrix(df_test['smiles'].tolist(), verbose=True, scaffolds=False, zero_diag=True, as_vector=True)
    save_hdf5(1-S, f'{ROOT_DIR}/{root}/{dataset}/test/tanimoto_distance_vector')
    del S

    print("Computing Tanimoto matrix for all screen molecules")
    S = get_tanimoto_matrix(df_screen['smiles'].tolist(), verbose=True, scaffolds=False, zero_diag=True, as_vector=True)
    save_hdf5(1 - S, f'{ROOT_DIR}/{root}/{dataset}/screen/tanimoto_distance_vector')
    del S


def save_hdf5(obj: Any, filename: str):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('obj', data=obj)
    hf.close()


def load_hdf5(filename: str) -> Any:
    hf = h5py.File(filename, 'r')
    obj = np.array(hf.get('obj'))
    hf.close()

    return obj
