from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import copy
import difflib
import itertools
import os
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np




# from RG-MPNN (kong et al.)
#code from RG-MPNN source code 

fdefName = 'RG_BaseFeatures.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)



def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx()))
    return mol

def show_baseFeature(fdefName):
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    print(len(factory.GetFeatureFamilies()))
    print(len(factory.GetFeatureDefs()))
    print(factory.GetFeatureFamilies())

def show_fra(factory):
    family_df = pd.DataFrame(columns=['family', 'definition']) 
    family_names = factory.GetFeatureFamilies()
    dict_fra = factory.GetFeatureDefs()
    for k,v in dict_fra.items():
        for fam in family_names:
            if fam in k:
                family_df.loc[k] = [fam, v]
    return family_df

def get_pharm(m):
    atom_num = m.GetNumAtoms()
    feats = factory.GetFeaturesForMol(m)
    dict_feats = {}
    for f in feats:
        dict_feats[f.GetFamily()] = dict_feats.get(f.GetFamily(),[])+[list(f.GetAtomIds())]  


    return dict_feats, atom_num


def similar(s1, s2): 
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def rm_dupli(l):
    l_new = []
    for item in l:
        if item not in l_new:
            l_new.append(item)
    return l_new
    

def single_merge_list(l):
    if len(l)==0:
        return l
    for j in range(1,len(l)):
        if set(l[0]) & set(l[j]) != set():
            l[0] = list(set(l[0])|set(l[j]))
            del l[j]
            break
    else:
        new_l.append(l[0])
        del l[0]

    return l

def x_merge(feat_dict, group_name):

    if group_name not in feat_dict.keys():
        return feat_dict
    
    l = feat_dict[group_name]  
    l = rm_dupli(l)
    global new_l  
    new_l = []
    while True:
        l = single_merge_list(l)
        if len(l) == 0:
            break
    feat_dict[group_name] = new_l
    return feat_dict


def y_merge(feat_dict, high_name, low_name):

    if (high_name not in feat_dict.keys()) or (low_name not in feat_dict.keys()):
        return feat_dict

    high = feat_dict[high_name]
    low = feat_dict[low_name]

    for i,x in enumerate(high):
        for j,y in enumerate(low):
            if (set(x)&set(y)) != set():
                high[i] = list(set(x)|set(y))
                low[j] = []       
    return feat_dict

def rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, high_name, low_name, rg_name):

    if (high_name not in feat_dict.keys()) or (low_name not in feat_dict.keys()):
        return rg_dict, feat_hit_dict

    high = feat_dict[high_name]
    low = feat_dict[low_name]

    for x in high:
        for y in low:
            if (set(x)&set(y)) != set():
                feat_hit_dict[high_name].append(x)
                feat_hit_dict[low_name].append(y)

                rg_dict[rg_name].append(list(set(x)|set(y))) 
            
    return rg_dict, feat_hit_dict

def AD_merge(rg_dict, D_name, A_name, rg_name):

    if (D_name not in rg_dict.keys()) or (A_name not in rg_dict.keys()):
        return rg_dict
    D = rg_dict[D_name]
    A = rg_dict[A_name]
    for i,x in enumerate(D):
        for j,y in enumerate(A):
            if (set(x)&set(y)) != set():
                rg_dict[rg_name].append(list(set(x)|set(y)))
                D[i] = []
                A[j] = []
                
    return rg_dict

def non_hit_define(feat_dict, feat_hit_dict, rg_dict, feat_name, rg_name):

    if feat_name not in feat_dict.keys():
        return rg_dict

    rg_dict[rg_name].extend([item for item in feat_dict[feat_name] if item not in feat_hit_dict[feat_name]])

    return rg_dict


def reduce_graph(m):

    rg_dict ={
           'Mn':[],'Y':[],'Nb':[],
           'Fe':[],'Zr':[],'Mo':[],
           'Cr':[],'Re':[],'Cu':[],
           'Ti':[],'Ta':[],'Co':[],
           'V':[],'W':[],'Ni':[],
           'Sc':[],'Hf':[],'Zn':[]
           }

    rg_dict_new ={
           'Mn':[],'Y':[],'Nb':[],
           'Fe':[],'Zr':[],'Mo':[],
           'Cr':[],'Re':[],'Cu':[],
           'Ti':[],'Ta':[],'Co':[],
           'V':[],'W':[],'Ni':[],
           'Sc':[],'Hf':[],'Zn':[]
           }
           
    feat_dict, atom_num = get_pharm(m)
    feat_hit_dict = dict([(key, []) for key in list(feat_dict.keys())])  

    feat_dict = x_merge(feat_dict, 'CC')
    feat_dict = x_merge(feat_dict, 'Acceptor')
    feat_dict = x_merge(feat_dict, 'NegIonizable')
    feat_dict = x_merge(feat_dict, 'PosIonizable')
    feat_dict = x_merge(feat_dict, 'Donor')

    feat_dict = y_merge(feat_dict,'PosIonizable','NegIonizable')
    feat_dict = y_merge(feat_dict,'PosIonizable','Donor')
    feat_dict = y_merge(feat_dict,'PosIonizable','Acceptor')
    feat_dict = y_merge(feat_dict,'NegIonizable','Donor')
    feat_dict = y_merge(feat_dict,'NegIonizable','Acceptor')

    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'PosIonizable', 'Mn')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'NegIonizable', 'Fe')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'Donor', 'Ti')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aromatic', 'Acceptor', 'V')

    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'PosIonizable', 'Y')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'NegIonizable', 'Zr')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'Donor', 'Ta')
    rg_dict, feat_hit_dict = rg_define_for_rings(feat_dict, rg_dict, feat_hit_dict, 'Aliphatic', 'Acceptor', 'W')

    rg_dict = y_merge(rg_dict,'Mn','Fe')
    rg_dict = y_merge(rg_dict,'Mn','Ti')
    rg_dict = y_merge(rg_dict,'Mn','V')
    rg_dict = y_merge(rg_dict,'Fe','Ti')
    rg_dict = y_merge(rg_dict,'Fe','V')

    rg_dict = y_merge(rg_dict,'Y','Zr')
    rg_dict = y_merge(rg_dict,'Y','Ta')
    rg_dict = y_merge(rg_dict,'Y','W')
    rg_dict = y_merge(rg_dict,'Zr','Ta')
    rg_dict = y_merge(rg_dict,'Zr','W')

    for x in ['Mn','Fe','Ti','V','Y','Zr','Ta','W']:
        rg_dict = x_merge(rg_dict,x)

    rg_dict = AD_merge(rg_dict, 'Ti', 'V', 'Cr') 
    rg_dict = AD_merge(rg_dict, 'Ta', 'W', 'Re') 

    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Aromatic', 'Sc')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Aliphatic', 'Hf')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'PosIonizable', 'Nb')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'NegIonizable', 'Mo')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Donor', 'Co')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'Acceptor', 'Ni')
    rg_dict = non_hit_define(feat_dict, feat_hit_dict, rg_dict, 'CC', 'Zn')

    rg_dict = AD_merge(rg_dict, 'Co', 'Ni', 'Cu') 

    rg_dict = x_merge(rg_dict, 'Zn')

    hit_num = []
    for x in rg_dict.values():
        if x != []:
            for y in x:
                hit_num.extend(y)

    C_num = [x for x in list(range(atom_num)) if x not in list(set(hit_num))]
    for i in C_num:
        rg_dict['Zn'].append([i])

    for key,values in rg_dict.items():
        if values == []:
            continue
        for value in values:
            if value == []:
                continue
            if value not in rg_dict_new[key]:
                rg_dict_new[key].append(value)

    return rg_dict_new




#My code

def mol_to_graph(mol, sanitize=True):

    if sanitize==True: 
        Chem.SanitizeMol(mol) #for molanchor to work
   

    graph = nx.Graph()

    # Add nodes (atoms)
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atom=atom)

    # Add edges (bonds)
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        graph.add_edge(start_idx, end_idx, bond=bond)  

    return graph 


def graph_to_pyg(graph):
    # Atom features
    x = []
    for _, attr in graph.nodes(data=True):
        atom = attr['atom']  # RDKit atom object
        features = [
            atom.GetAtomicNum(),                    # atomic number
            atom.GetDegree(),                       # number of neighbors
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),           # RDKit enum
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(includeNeighbors=True)
        ]
        x.append(features)

    x = torch.tensor(x, dtype=torch.float)

    # Edge index and edge features
    edge_index = []
    edge_attr = []



    for u, v, attr in graph.edges(data=True):

        if type(attr['bond']) !=  float:
            bond = attr['bond']  # full RDKit bond object
            features = [
                bond.GetBondTypeAsDouble(),  # 1.0, 2.0, 3.0, 1.5 (aromatic)
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
        # for molanchor with single atom fragments.
        else:
            features = [0, 0, 0]
        # Add both directions for undirected graph
        edge_index.append([u, v])
        edge_index.append([v, u])
        edge_attr.append(features)
        edge_attr.append(features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # Common atomic numbers
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPE_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

def one_hot_encoding(x, allowable_set):
    """Simple one-hot encoder"""
    if x not in allowable_set:
        x = allowable_set[-1]  # Map to 'unknown' token if needed
    return [x == s for s in allowable_set]

def graph_to_pyg_oh(graph):
    # Atom features
    x = []
    for _, attr in graph.nodes(data=True):
        atom = attr['atom']  # RDKit atom object

        atom_features = (
            one_hot_encoding(atom.GetAtomicNum(), ATOM_LIST) +
            one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_LIST) +
            [atom.GetFormalCharge(),
             atom.GetDegree(),
             atom.GetTotalNumHs(includeNeighbors=True),
             int(atom.GetIsAromatic())]
        )
        x.append(atom_features)

    x = torch.tensor(x, dtype=torch.float)

    # Edge index and edge features
    edge_index = []
    edge_attr = []

    for u, v, attr in graph.edges(data=True):
        if type(attr['bond']) != float:
            bond = attr['bond']  # RDKit bond object
            bond_features = (
                one_hot_encoding(bond.GetBondType(), BOND_TYPE_LIST) +
                [int(bond.GetIsConjugated()), int(bond.IsInRing())]
            )
        else:
            bond_features = [0] * (len(BOND_TYPE_LIST) + 2)  # Pad with zeros

        edge_index.append([u, v])
        edge_index.append([v, u])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Reduce graph 
def label_nodes_with_pharmacophores(G, rg_dict):
    for pharmacophore, groups in rg_dict.items():
        for i, group in enumerate(groups):
            group_label = f"{pharmacophore}_{i+1}"
        #for group in groups:
            for atom_idx in group:
                if G.has_node(atom_idx):
                    G.nodes[atom_idx]['pharmacophore'] = pharmacophore
                    G.nodes[atom_idx]['group'] = group_label
    return G

def get_pooling_tensor(G):
    atom_indices = []
    group_labels = []

    for n, data in G.nodes(data=True):
        atom_indices.append(n)
        group_labels.append(data['group'])

    # Step 2: Map each unique group name to an integer index
    group_to_idx = {g: i for i, g in enumerate(sorted(set(group_labels)))}
    pharmacophore_indices = [group_to_idx[g] for g in group_labels]

    # Step 3: Create tensor
    index_tensor = torch.tensor([atom_indices, pharmacophore_indices], dtype=torch.long)

    return index_tensor



def get_atom_features(G): 
    features = []
    node_to_idx = {}  # Keep track of node ordering
    for i, n in enumerate(G.nodes()):
        atom = G.nodes[n]["atom"]
        
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        hybridization = int(atom.GetHybridization())
        is_aromatic = int(atom.GetIsAromatic())
        number_hs= atom.GetTotalNumHs(includeNeighbors=True) 

        features.append([atomic_num, degree, formal_charge, hybridization, is_aromatic, number_hs])
        node_to_idx[n] = i

    node_features = torch.tensor(features, dtype=torch.float)
    return node_features

def get_atom_features_oh(G): 
    features = []
    node_to_idx = {}  # Keep track of node ordering

    for i, n in enumerate(G.nodes()):
        atom = G.nodes[n]["atom"]
        
        atomic_num = atom.GetAtomicNum()
        atomic_num_oh = one_hot_encoding(atomic_num, ATOM_LIST)

        hybridization = atom.GetHybridization()  # not int() here, let one-hot handle enum directly
        hybridization_oh = one_hot_encoding(hybridization, HYBRIDIZATION_LIST)

        formal_charge = atom.GetFormalCharge()
        degree = atom.GetDegree()
        is_aromatic = int(atom.GetIsAromatic())
        number_hs = atom.GetTotalNumHs(includeNeighbors=True)

        # Flatten the full feature vector
        features.append(atomic_num_oh + [degree, formal_charge] + hybridization_oh + [is_aromatic, number_hs])
        node_to_idx[n] = i

    node_features = torch.tensor(features, dtype=torch.float)
    return node_features





def get_edge_counter(G):
    
    group_labels = [G.nodes[n]["group"] for n in G.nodes()]
    unique_groups = sorted(set(group_labels)) 

    # Create mapping from group names to reduced graph indices
    group_to_rg_idx = {group: i for i, group in enumerate(unique_groups)}

    # Map each atom node to its group index
    atom_to_group_idx = {}
    group_to_atoms = defaultdict(set)

    for n, data in G.nodes(data=True):
        group = data["group"]
        group_idx = group_to_rg_idx[group]
        atom_to_group_idx[n] = group_idx
        group_to_atoms[group_idx].add(n)

    # count inter-group atom connections (edges)
    edge_counter = defaultdict(int)
    atom_pair_seen = set()

    for u, v in G.edges():
        gu = atom_to_group_idx[u]
        gv = atom_to_group_idx[v]
        
        if gu == gv:
            continue  # Skip intra-group edges

        group_pair = tuple(sorted((gu, gv)))
        atom_pair = tuple(sorted((u, v)))

        if atom_pair not in atom_pair_seen:
            edge_counter[group_pair] += 1
            atom_pair_seen.add(atom_pair)
    return edge_counter

def rg_to_pyg(RG):
    # Node features
        x = []
        for n, data in RG.nodes(data=True):
            features = data['features']
            if isinstance(features, torch.Tensor):
                features = features.tolist()
            x.append([float(f) for f in features])  # Ensure flat float list
        x = torch.tensor(x, dtype=torch.float)

        # Edge index and edge features
        edge_index = []
        edge_attr = []

        for u, v, data in RG.edges(data=True):
            bond_type = data.get('bond', 1.0)
            features = [
                float(bond_type),  # Bond type
                0.0,               # is_conjugated 
                0.0                # is_in_ring
            ]
            edge_index.extend([[u, v], [v, u]])
            edge_attr.extend([features, features])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def rg_to_pyg_oh(RG):
    # Node features
        x = []
        for n, data in RG.nodes(data=True):
            features = data['features']
            if isinstance(features, torch.Tensor):
                features = features.tolist()
            x.append([float(f) for f in features])  # Ensure flat float list
        x = torch.tensor(x, dtype=torch.float)

        # Edge index and edge features
        edge_index = []
        edge_attr = []

        for u, v, data in RG.edges(data=True):
            bond_type = float(data.get('bond_type', 1.0))  
            bond_oh = one_hot_encoding(bond_type, BOND_TYPE_LIST)
            features = bond_oh + [0.0, 0.0]  #flatten 
            edge_index.extend([[u, v], [v, u]])
            edge_attr.extend([features, features])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def reduce_graph_from_mol(mol):
    #rg dict
    mol = mol_with_atom_index(mol)
    rg_dict_mol=reduce_graph(mol)

    #convert to networkx graph and label with pharmacophore groups 
    G = mol_to_graph(mol)
    G = label_nodes_with_pharmacophores(G, rg_dict_mol)

    # get pool tensor and node features
    pool_tensor = get_pooling_tensor(G)

    atom_indices = pool_tensor[0] 
    pharmacophore_indices = pool_tensor[1]

    node_features = get_atom_features(G)
    selected_features = node_features[atom_indices]


    #pooling

    pharmacophore_node_features = scatter_mean(
        selected_features,
        index=pharmacophore_indices,
        dim=0
    )

    #get edge counter for adding edges to RG 
    edge_counter = get_edge_counter(G)

    #create RG 
    RG = nx.Graph()

    #add nodes and edges to RG 
    group_labels = [G.nodes[n]["group"] for n in G.nodes()]
    unique_groups = sorted(set(group_labels)) 

    for i, group_name in enumerate(unique_groups):
        RG.add_node(i, group=group_name, features=pharmacophore_node_features[i])


    for (u, v), count in edge_counter.items():
        if count >= 2:
            RG.add_edge(u, v, bond_type='2.0')
        else:
            RG.add_edge(u, v, bond_type='1.0')

    #convert to pytoch geometric 
    rg_pyg = rg_to_pyg(RG)

    return rg_pyg


#with one hot encoding

def reduce_graph_from_mol_oh(mol):
    #get rg dict
    mol = mol_with_atom_index(mol)
    rg_dict_mol=reduce_graph(mol)

    #convert to networkx graph and label with pharmacophore groups 
    G = mol_to_graph(mol)
    G = label_nodes_with_pharmacophores(G, rg_dict_mol)

    #get pool tensor and node features
    pool_tensor = get_pooling_tensor(G)

    atom_indices = pool_tensor[0] 
    pharmacophore_indices = pool_tensor[1]

    node_features = get_atom_features_oh(G)
    selected_features = node_features[atom_indices]


    #pool 

    pharmacophore_node_features = scatter_mean(
        selected_features,
        index=pharmacophore_indices,
        dim=0
    )

    #get edge counter for addign edges to RG 
    edge_counter = get_edge_counter(G)

    #create RG 
    RG = nx.Graph()

    #add nodoes and edges to RG 
    group_labels = [G.nodes[n]["group"] for n in G.nodes()]
    unique_groups = sorted(set(group_labels)) 

    for i, group_name in enumerate(unique_groups):
        RG.add_node(i, group=group_name, features=pharmacophore_node_features[i])


    for (u, v), count in edge_counter.items():
        if count >= 2:
            RG.add_edge(u, v, bond_type='2.0')
        else:
            RG.add_edge(u, v, bond_type='1.0')

    #convert to pytoch geometric 
    rg_pyg = rg_to_pyg_oh(RG)

    return rg_pyg





def mol_to_pool_idx(mol):
    #get pharmaophore nodes 
    mol = mol_with_atom_index(mol)
    rg_dict_mol=reduce_graph(mol)
    G = mol_to_graph(mol)
    G = label_nodes_with_pharmacophores(G, rg_dict_mol)

    #get pooling tensor

    pool_tensor = get_pooling_tensor(G)
    atom_indices = pool_tensor[0] 
    pharmacophore_indices = pool_tensor[1]

    #add new_edge_index and new_edge_attr
    edge_counter = get_edge_counter(G)
    edge_list = list(edge_counter.keys())
    new_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.tensor(
        [[2.0] if edge_counter[e] >= 2 else [1.0] for e in edge_list],
        dtype=torch.float)
    

    return pharmacophore_indices, new_edge_index, new_edge_attr





def reduce_graph_from_mol_nx(mol):
    # Step 1: get rg dict
    mol = mol_with_atom_index(mol)
    rg_dict_mol = reduce_graph(mol)

    # Step 2: convert to networkx graph and label with pharmacophore groups 
    G = mol_to_graph(mol)
    G = label_nodes_with_pharmacophores(G, rg_dict_mol)

    # Step 3: get pool tensor and node features
    pool_tensor = get_pooling_tensor(G)
    atom_indices = pool_tensor[0] 
    pharmacophore_indices = pool_tensor[1]

    node_features = get_atom_features(G)
    selected_features = node_features[atom_indices]

    # Step 4: pool
    pharmacophore_node_features = scatter_mean(
        selected_features,
        index=pharmacophore_indices,
        dim=0
    )

    # Step 5: get edge counter for adding edges to RG 
    edge_counter = get_edge_counter(G)

    # Step 6: create RG 
    RG = nx.Graph()

    # Step 7: add nodes and edges to RG
    group_labels = [G.nodes[n]["group"] for n in G.nodes()]
    unique_groups = sorted(set(group_labels)) 

    for i, group_name in enumerate(unique_groups):
        # Find which atoms belong to this group
        node_atom_indices = [atom_idx for atom_idx, ph_idx in zip(atom_indices, pharmacophore_indices)
                             if ph_idx == i]
        
        RG.add_node(i,
                    group=group_name,
                    features=pharmacophore_node_features[i],
                    atom_indices=node_atom_indices)  # <--- store atom indices  (make sure RG stores atom_indices for each node!)

    # Add edges
    for (u, v), count in edge_counter.items():
        if count >= 2:
            RG.add_edge(u, v, bond_type='2.0')
        else:
            RG.add_edge(u, v, bond_type='1.0')

    return RG



def get_rg_edges(edge_index, pharma_index, device=None):
    if device is None:
        device = edge_index.device

    edge_counter = {}
    E = edge_index.size(1)

    seen_atom_pairs = set()

    for i in range(E):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u == v:
            continue

        # undirected atom–atom edge (deduplicate)
        atom_pair = tuple(sorted((u, v)))
        if atom_pair in seen_atom_pairs:
            continue
        seen_atom_pairs.add(atom_pair)

        gu, gv = pharma_index[u].item(), pharma_index[v].item()
        if gu == gv:
            continue  # skip intra-group

        gpair = tuple(sorted((gu, gv)))
        edge_counter[gpair] = edge_counter.get(gpair, 0) + 1

    edges, attrs = [], []
    for (gu, gv), count in edge_counter.items():
        edges.append([gu, gv])  # just one direction
        # 1 if exactly one bond, 2 if two or more
        attrs.append(1 if count == 1 else 2)

    new_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
    new_edge_attr = torch.tensor(attrs, dtype=torch.float, device=device).unsqueeze(-1)

    return new_edge_index, new_edge_attr



def get_rg_edges_vectorized(edge_index, pharma_index, device=None):
    if device is None:
        device = edge_index.device

    # Map atoms to pharmacophores
    gu = pharma_index[edge_index[0]]
    gv = pharma_index[edge_index[1]]

    # Ignore intra-pharmacophore edges
    mask = gu != gv
    u, v = edge_index[0, mask], edge_index[1, mask]
    gu, gv = gu[mask], gv[mask]

    # Deduplicate atom–atom edges (undirected)
    atom_pairs = torch.stack([u, v], dim=1)
    atom_pairs = torch.sort(atom_pairs, dim=1).values  # ensure undirected
    atom_pairs = torch.unique(atom_pairs, dim=0)

    # Map to pharmacophore pairs
    gu = pharma_index[atom_pairs[:, 0]]
    gv = pharma_index[atom_pairs[:, 1]]
    g_pairs = torch.sort(torch.stack([gu, gv], dim=1), dim=1).values

    # Count occurrences of each pharmacophore pair
    # Using a dictionary for simplicity
    edge_counter = {}
    for i in range(g_pairs.size(0)):
        gp = tuple(g_pairs[i].tolist())
        edge_counter[gp] = edge_counter.get(gp, 0) + 1

    # Build edges and attributes
    edges, attrs = [], []
    for (gu, gv), count in edge_counter.items():
        edges.append([gu, gv])
        attrs.append(1 if count == 1 else 2)

    new_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
    new_edge_attr = torch.tensor(attrs, dtype=torch.float, device=device).unsqueeze(-1)

    return new_edge_index, new_edge_attr


def get_rg_edges_with_mask(edge_index, pharma_index, edge_mask, device=None):
    if device is None:
        device = edge_index.device

    # Map atoms to pharmacophores
    gu = pharma_index[edge_index[0]]
    gv = pharma_index[edge_index[1]]

    # Ignore intra-pharmacophore edges
    mask = gu != gv
    u, v = edge_index[0, mask], edge_index[1, mask]
    gu, gv = gu[mask], gv[mask]
    edge_mask = edge_mask[mask]  # filter mask too

    # Deduplicate atom–atom edges (undirected)
    atom_pairs = torch.stack([u, v], dim=1)
    atom_pairs = torch.sort(atom_pairs, dim=1).values
    unique_pairs, inv = torch.unique(atom_pairs, dim=0, return_inverse=True)

    # Aggregate mask values per atom pair
    atom_pair_weights = torch.zeros(unique_pairs.size(0), device=device)
    atom_pair_weights.scatter_add_(0, inv, edge_mask)

    # Map to pharmacophore pairs
    gu = pharma_index[unique_pairs[:, 0]]
    gv = pharma_index[unique_pairs[:, 1]]
    g_pairs = torch.sort(torch.stack([gu, gv], dim=1), dim=1).values

    # Aggregate again at pharmacophore level
    g_unique, g_inv = torch.unique(g_pairs, dim=0, return_inverse=True)
    g_weights = torch.zeros(g_unique.size(0), device=device)
    g_weights.scatter_add_(0, g_inv, atom_pair_weights)

    # Build new edge_index and attributes
    new_edge_index = g_unique.t().contiguous()
    new_edge_attr = g_weights.unsqueeze(-1)  # explanation strength

    return new_edge_index, new_edge_attr




def plot_pharma_graph(RG, pos=None):
    """
    Plot the pharmacophore reduced graph (RG).
    If pos is provided (e.g. ellipse centers), use it.
    Otherwise, fall back to kamada_kawai_layout.
    """
    labels = {n: RG.nodes[n]['group'].rsplit('_', 1)[0] for n in RG.nodes}

    atom_color_map = {
        'Zn': '#B2BEB5', 'Hf':'#B2BEB5', 'Sc':'#B2BEB5',
        'Co':'#99AFD7', 'Ta':'#99AFD7', 'Ti':'#99AFD7',
        'Ni':'#F1BD78', 'W':'#F1BD78', 'V':'#F1BD78',
        'Cu':'#8C819A', 'Re':'#8C819A', 'Cr':'#8C819A',
        'Mo':'#F6CF68', 'Zr':'#F6CF68', 'Fe':'#F6CF68',
        'Nb':'#9CCE8D', 'Y':'#9CCE8D', 'Mn':'#9CCE8D'
    }
    default_color = 'lightgrey'
    node_colors = [atom_color_map.get(labels[n], default_color) for n in RG.nodes]

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- Use given pos, else kamada layout ---
    if pos is None:
        pos = nx.kamada_kawai_layout(RG)
         # Flip horizontally
        for n, (x, y) in pos.items():
            pos[n] = (-x, y)   # flip x-axis


    nx.draw(
        RG,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1,
        font_size=8,
        ax=ax
    )

    all_coords = np.array(list(pos.values()))
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    x_pad = (xmax - xmin) * 0.2
    y_pad = (ymax - ymin) * 0.2
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    ax.set_aspect('equal')

    plt.close(fig)
    return fig
    

def plot_pharma_graph_shapes(G):
    labels = {n:G.nodes[n]['group'].rsplit('_',1)[0] for n in G.nodes}

    atom_color_map = {
    'Zn': '#B2BEB5', 'Hf':'#B2BEB5', 'Sc':'#B2BEB5',
    'Co':'#99AFD7', 'Ta':'#99AFD7', 'Ti':'#99AFD7',
    'Ni':'#F1BD78', 'W':'#F1BD78', 'V':'#F1BD78',
    'Cu':'#8C819A', 'Re':'#8C819A', 'Cr':'#8C819A',
    'Mo':'#F6CF68', 'Zr':'#F6CF68', 'Fe':'#F6CF68',
    'Nb':'#9CCE8D', 'Y':'#9CCE8D', 'Mn':'#9CCE8D'
}
    default_color = 'lightgrey'

    atom_shape_map = {
    'Zn':'d', 'Hf':'s', 'Sc':'o',
    'Co':'d', 'Ta':'s', 'Ti':'o',
    'Ni':'d', 'W':'s', 'V':'o',
    'Cu':'d', 'Re':'s', 'Cr':'o',
    'Mo':'d', 'Zr':'s', 'Fe':'o',
    'Nb':'d', 'Y':'s', 'Mn':'o'
}
    default_shape = 'o'

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()

    # Compute layout
    #pos = nx.spring_layout(G, seed=42)

    pos = nx.kamada_kawai_layout(G)
      # Flip horizontally
    for n, (x, y) in pos.items():
        pos[n] = (-x, y)   # flip x-axis

    # Draw nodes by shape group
    for shape in set(atom_shape_map.values()):
        nodelist = [n for n in G.nodes if atom_shape_map.get(labels[n], default_shape) == shape]
        node_colors = [atom_color_map.get(labels[n], default_color) for n in nodelist]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodelist,
            node_color=node_colors,
            node_shape=shape,
            node_size=200,
            edgecolors='black',
            linewidths=1
        )

    all_coords = np.array(list(pos.values()))
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    x_pad = (xmax - xmin) * 0.2
    y_pad = (ymax - ymin) * 0.2
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    ax.set_aspect('equal')

    # Draw edges + labels
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    #plt.show()
    plt.close(fig) 
    return fig


def mol_to_pharma_graph(mol):
    mol = mol_with_atom_index(mol)
    rg_dict_mol=reduce_graph(mol)
    G = mol_to_graph(mol)
    G = label_nodes_with_pharmacophores(G, rg_dict_mol)
    return G


def plot_mol_with_pharma_circles(mol, circle_padding=0.8, show=True):
    """
    Plot a molecule with pharmacophore circles around reduced graph nodes.
    Stores the ellipse center in each RG node for later use.
    Returns: figure and reduced graph (with ellipse_center in each node), and pos for visualization with plot_phamra_graph 
    """
    # Make a safe copy
    mol = Chem.Mol(mol)
    
    # Atom colors for pharmacophore groups
    default_color = "lightgrey"
    atom_color_map = {
        'Zn': '#B2BEB5', 'Hf':'#B2BEB5', 'Sc':'#B2BEB5',
        'Co':'#99AFD7', 'Ta':'#99AFD7', 'Ti':'#99AFD7',
        'Ni':'#F1BD78', 'W':'#F1BD78', 'V':'#F1BD78',
        'Cu':'#8C819A', 'Re':'#8C819A', 'Cr':'#8C819A',
        'Mo':'#F6CF68', 'Zr':'#F6CF68', 'Fe':'#F6CF68',
        'Nb':'#9CCE8D', 'Y':'#9CCE8D', 'Mn':'#9CCE8D'
    }
    
    # --- Reduced graph ---
    RG = reduce_graph_from_mol_nx(mol)
    labels = {n: RG.nodes[n]['group'].rsplit('_',1)[0] for n in RG.nodes}

    # --- Compute 2D coords ---
    Chem.Kekulize(mol, clearAromaticFlags=True)
    AllChem.Compute2DCoords(mol)
    coords = mol.GetConformer().GetPositions()
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_axis_off()
    
    bond_trim_nonC = 0.4  # shorten near heteroatoms
    double_offset = 0.1   # line spacing for double bonds
    
    # --- Draw bonds ---
    for bond in mol.GetBonds():
        b_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1, _ = coords[b_idx]
        x2, y2, _ = coords[e_idx]
        
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            continue
        ux, uy = dx/length, dy/length
        
        a1, a2 = mol.GetAtomWithIdx(b_idx).GetSymbol(), mol.GetAtomWithIdx(e_idx).GetSymbol()
        trim_start = bond_trim_nonC if a1 != "C" else 0.0
        trim_end   = bond_trim_nonC if a2 != "C" else 0.0
        
        if bond.GetBondType() in (Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE):
            trim_start = max(trim_start, 0.1)
            trim_end   = max(trim_end, 0.1)
        
        x1s, y1s = x1 + trim_start*ux, y1 + trim_start*uy
        x2s, y2s = x2 - trim_end*ux,   y2 - trim_end*uy
        
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.DOUBLE:
            ox, oy = -uy*double_offset, ux*double_offset
            ax.plot([x1s+ox, x2s+ox], [y1s+oy, y2s+oy], '-', color='black', lw=2)
            ax.plot([x1s-ox, x2s-ox], [y1s-oy, y2s-oy], '-', color='black', lw=2)
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            ox, oy = -uy*double_offset*1.5, ux*double_offset*1.5
            ax.plot([x1s, x2s], [y1s, y2s], '-', color='black', lw=2)
            ax.plot([x1s+ox, x2s+ox], [y1s+oy, y2s+oy], '-', color='black', lw=2)
            ax.plot([x1s-ox, x2s-ox], [y1s-oy, y2s-oy], '-', color='black', lw=2)
        else:
            ax.plot([x1s, x2s], [y1s, y2s], '-', color='black', lw=2)
    
    # --- Draw atoms & heteroatom labels ---
    for i, (x, y, z) in enumerate(coords):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()
        if symbol == "C":
            ax.plot(x, y, 'o', color='black', markersize=0, zorder=3)
        else:
            ax.text(x, y, symbol, fontsize=10, ha='center', va='center',
                    color='black', weight='semibold', zorder=4)
    
    # --- Overlay pharmacophore ellipses ---
    for n in RG.nodes:
        color = atom_color_map.get(labels[n], default_color)
        atom_indices = RG.nodes[n].get('atom_indices', [])
        if not atom_indices:
            # Node has no atoms assigned
            continue
        
        atom_indices = np.array(atom_indices, dtype=int)
        group_coords = coords[atom_indices, :2]
        xmin, ymin = group_coords.min(axis=0)
        xmax, ymax = group_coords.max(axis=0)
        
        width = (xmax - xmin) + circle_padding
        height = (ymax - ymin) + circle_padding
        center = ((xmax + xmin)/2, (ymax + ymin)/2)
        
        # Store the ellipse center for later use
        RG.nodes[n]['ellipse_center'] = center
        
        ellipse = Ellipse(center, width, height,
                          edgecolor=color, facecolor=color,
                          alpha=0.85, linewidth=2, zorder=1)
        ax.add_patch(ellipse)
    
    ax.axis("equal")
    if show:
        plt.show()
    plt.close(fig)

    pos = {n: RG.nodes[n]["ellipse_center"] for n in RG.nodes}

    return fig, RG, pos