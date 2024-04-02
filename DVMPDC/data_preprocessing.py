import random
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, explicit_H=True, use_chirality=False):
    """
    用于生成单个原子特征表示的函数。该函数接受一个 RDKit 中的原子对象作为输入，并返回一个包含该原子特征表示的 NumPy 数组。
    原子的元素符号（one-of-k 编码）；44
    原子的度数归一化值（GetDegree() 函数除以10）；1
    原子的隐式价、形式电荷和自由基电子数；3
    原子的杂化轨道类型（one-of-k 编码）；5
    原子是否为芳香环上的原子；1
    原子的显式氢数（如果 explicit_H 参数为 True）；1
    原子的立体异构信息（如果 use_chirality 参数为 True）。没有使用
    """
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(),
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    return undirected_edge_list.T, n_features


def get_bipartite_graph(mol_graph_1, mol_graph_2):
    x1 = np.arange(0, len(mol_graph_1.GetAtoms()))
    x2 = np.arange(0, len(mol_graph_2.GetAtoms()))
    edge_list = torch.LongTensor(np.meshgrid(x1, x2))
    edge_list = torch.stack([edge_list[0].reshape(-1), edge_list[1].reshape(-1)])
    return edge_list


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def get_cell_feature(cellfile, threshold, delimiter=','):
    cell_df = pd.read_csv(cellfile, sep=delimiter)
    cell_line = cell_df.iloc[:, 0]
    gene_expr = cell_df.iloc[:, 1:]
    variances = gene_expr.var()
    selected_columns = variances[variances >= threshold].index
    cell_features = pd.concat([cell_line, gene_expr[selected_columns]], axis=1)
    return cell_features

cell_features = get_cell_feature(cellfile='data/oneil_cell.csv', threshold=0.8, delimiter=',')

df_drugs_smiles = pd.read_csv('data/oneil_drug_two_smiles.csv')

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_name'], df_drugs_smiles['smiles'])]

drug_to_mol_graph = {id:Chem.MolFromSmiles(smiles.strip()) for id, smiles in zip(df_drugs_smiles['drug_name'], df_drugs_smiles['smiles'])}

ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])

MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol)
                                for drug_id, mol in drug_id_mol_graph_tup}

MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

class DrugDataset(Dataset):
    def __init__(self, tri_list, ratio=1.0, shuffle=True):
        self.tri_list = []
        self.ratio = ratio

        for da, db, cell, label, *_ in tri_list:
            if ((da in MOL_EDGE_LIST_FEAT_MTX) and (db in MOL_EDGE_LIST_FEAT_MTX)):
                self.tri_list.append((da, db, cell, label))

        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_cell = []
        pos_h_samples = []
        pos_t_samples = []
        pos_b_samples = []

        neg_cell = []
        neg_h_samples = []
        neg_t_samples = []
        neg_b_samples = []

        for da, db, cell, label in batch:
            if label == 1:
                cell_data = cell_features.loc[
                    cell_features['cell_line_name'] == cell, cell_features.columns != 'cell_line_name']
                cell_data = cell_data.values
                pos_cell.append(cell_data)
                h_data = self.__create_graph_data(da)
                t_data = self.__create_graph_data(db)
                h_graph = drug_to_mol_graph[da]
                t_graph = drug_to_mol_graph[db]

                pos_b_graph = self._create_b_graph(get_bipartite_graph(h_graph, t_graph), h_data.x, t_data.x)

                pos_h_samples.append(h_data)
                pos_t_samples.append(t_data)
                pos_b_samples.append(pos_b_graph)

            if label == 0:
                cell_data = cell_features.loc[
                    cell_features['cell_line_name'] == cell, cell_features.columns != 'cell_line_name']
                cell_data = cell_data.values
                neg_cell.append(cell_data)
                h_data = self.__create_graph_data(da)
                t_data = self.__create_graph_data(db)
                h_graph = drug_to_mol_graph[da]
                t_graph = drug_to_mol_graph[db]

                neg_b_graph = self._create_b_graph(get_bipartite_graph(h_graph, t_graph), h_data.x, t_data.x)

                neg_h_samples.append(h_data)
                neg_t_samples.append(t_data)
                neg_b_samples.append(neg_b_graph)

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_b_samples = Batch.from_data_list(pos_b_samples)
        pos_cell = torch.tensor(pos_cell)

        pos_tri = (pos_h_samples, pos_t_samples, pos_cell, pos_b_samples)

        neg_h_samples = Batch.from_data_list(neg_h_samples)
        neg_t_samples = Batch.from_data_list(neg_t_samples)
        neg_b_samples = Batch.from_data_list(neg_b_samples)
        neg_cell = torch.tensor(neg_cell)

        neg_tri = (neg_h_samples, neg_t_samples, neg_cell, neg_b_samples)

        return pos_tri, neg_tri

    def __create_graph_data(self, id):
        edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
        n_features = MOL_EDGE_LIST_FEAT_MTX[id][1]
        return Data(x=n_features, edge_index=edge_index)

    def _create_b_graph(self, edge_index, x_s, x_t):
        return BipartiteData(edge_index, x_s, x_t)


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

