
import os
import warnings
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from collections import Counter
from scipy.cluster import hierarchy
from data_prep import MasterDataset, load_hdf5, get_data, split_data, similarity_vectors
warnings.simplefilter(action='ignore', category=FutureWarning)







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






ROOT_DIR = ".."


if __name__ == '__main__':
    os.makedirs(ROOT_DIR+"/data/ALDH1/screen", exist_ok=True)
    os.makedirs(ROOT_DIR+"/data/ALDH1/test", exist_ok=True)
    os.makedirs(ROOT_DIR+"/data/PKM2/screen", exist_ok=True)
    os.makedirs(ROOT_DIR+"/data/PKM2/test", exist_ok=True)
    os.makedirs(ROOT_DIR+"/data/VDR/screen", exist_ok=True)
    os.makedirs(ROOT_DIR+"/data/VDR/test", exist_ok=True)

    # Process the data
    for dataset in ['ALDH1', 'PKM2', 'VDR']:

        df = get_data(dataset=dataset)
        df_screen, df_test = split_data(df, screen_size=100000, test_size=20000, dataset=dataset)

        MasterDataset(name='screen', df=df_screen, overwrite=True, dataset=dataset)
        MasterDataset(name='test', df=df_test, overwrite=True, dataset=dataset)




    for dataset in ["ALDH1", "PKM2", "VDR"]:
        for usage in ["screen", "test"]:

            fp = torch.load(os.path.join(ROOT_DIR, "data", dataset, usage, "x"))
            graph_list = torch.load(os.path.join(ROOT_DIR, "data", dataset, usage, "graphs"))
            graph2_list = []


            for i, graph in tqdm(enumerate(graph_list)):
                graph.fp = torch.tensor([fp[i]], dtype=torch.float32)

                mol = Chem.MolFromSmiles(graph.smiles, sanitize=True)
                xp, edgep_index, edgep_attr = mol_to_graph_data_obj_simple_3D(mol)
                graph.xp = xp
                graph.edgep_index = edgep_index
                graph.edgep_attr = edgep_attr

                graph2_list.append(graph)


            torch.save(graph2_list, os.path.join(ROOT_DIR, "data", dataset, usage, "graphs2"), pickle_protocol=4)















        # df_screen = pd.read_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/screen.csv'))
        # df_test = pd.read_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/test.csv'))
        # similarity_vectors(df_screen, df_test, dataset=dataset)

    # # Perform clustering for each dataset
    # for dataset, tani_cutoffs in zip(['PKM2', 'VDR', 'ALDH1'], [[0.80, 0.61], [0.80, 0.70], [0.80, 0.60]]):
    #     ds_screen = MasterDataset('screen', representation='ecfp', dataset=dataset)
    #     x_screen, y_screen, smiles_screen = ds_screen.all()
    #     smiles_index = torch.load(f'data/{dataset}/screen/smiles_index')
    #     min_supercluster_size = 128
    #     min_subcluster_size = 64
    #     n = len(smiles_screen)
    #     subcluster_mu = np.mean(y_screen.tolist()) * min_subcluster_size
    #     subcluster_sigma = np.std(y_screen.tolist()) * np.sqrt(min_subcluster_size)

    #     D = load_hdf5(f'data/{dataset}/screen/tanimoto_distance_vector')

    #     # Average clustering -> works probably slightly better than complete, as it gives much larger clusters
    #     linkage = hierarchy.average(D)
    #     del D

    #     # Cut the tree in clusters where the average intra-cluster Tanimoto distance is 0.8, 0.6
    #     cut_clusters = hierarchy.cut_tree(linkage, height=tani_cutoffs)
    #     cut = np.concatenate((np.array([range(n)]).T, cut_clusters), axis=1)

    #     # Find the big superclusters
    #     super_clusters = [clust for clust, cnt in Counter(cut[:, 1]).items() if cnt >= min_supercluster_size]
    #     cut = cut[[True if i in super_clusters else False for i in cut[:, 1]]]

    #     # find the subclusters
    #     sub_clusters = [clust for clust, cnt in Counter(cut[:, -1]).items() if cnt >= min_subcluster_size]


    #     # put the subclusters and superclusters together
    #     cluster_smiles = []
    #     for sub_clust in sub_clusters:
    #         super_clust = cut[:, 1][cut[:, 2] == sub_clust][0]
    #         cluster_smiles.append([
    #             smiles_screen[cut[:, 0][np.where(cut[:, 2] == sub_clust)]],
    #             smiles_screen[cut[:, 0][np.where(cut[:, 1] == super_clust)]]
    #         ])
    #     cluster_smiles = np.array(cluster_smiles, dtype=object)  # len = 46  len(cluster_smiles)

    #     # Only keep the clusters where the subcluster actually contains a hit
    #     cluster_smiles_with_hits = []
    #     for subset, superset in cluster_smiles:
    #         y_subset = y_screen[np.array([smiles_index[smi] for smi in subset])]
    #         # print(sum(y_subset))
    #         if sum(y_subset) > subcluster_mu + subcluster_sigma:
    #             cluster_smiles_with_hits.append([subset, superset])
    #     cluster_smiles_with_hits = np.array(cluster_smiles_with_hits, dtype=object)   # len(cluster_smiles_with_hits)

    #     for i in cluster_smiles_with_hits:
    #         print(len(i[0]), len(i[1]), len(i[0]) / len(i[1]))

    #     only_child = []
    #     for i in range(len(cluster_smiles_with_hits)):
    #         supercluster = cluster_smiles_with_hits[i][1]
    #         subcluster = cluster_smiles_with_hits[i][0]

    #         contains = 0
    #         if len(np.intersect1d(supercluster, subcluster)) > 0:
    #             contains = 1

    #         for j in range(len(cluster_smiles_with_hits)):
    #             if i != j and len(np.intersect1d(cluster_smiles_with_hits[j][1], subcluster)) > 0:
    #                 contains += 1

    #         only_child.append(contains)

    #     cluster_smiles_with_hits = cluster_smiles_with_hits[np.where(np.array(only_child) == 1)]
    #     print(len(cluster_smiles_with_hits))

    #     torch.save(cluster_smiles_with_hits, f'data/{dataset}/screen/starting_clusters')

    # Process three other datasets with very few actives as an extra case-study
    # for dataset in ['IDH1', 'ADRB2', 'OPRK1', 'GBA', 'KAT2A', 'FEN1']:

    #     df = get_data(dataset=dataset)
    #     df_screen, df_test = split_data(df, screen_size=100000, test_size=20000, dataset=dataset)

    #     MasterDataset(name='screen', df=df_screen, overwrite=True, dataset=dataset)
    #     MasterDataset(name='test', df=df_test, overwrite=True, dataset=dataset)

    #     df_screen = pd.read_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/screen.csv'))
    #     df_test = pd.read_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/test.csv'))
