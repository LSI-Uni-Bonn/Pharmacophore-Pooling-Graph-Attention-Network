#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import get_ipython

get_ipython().run_line_magic('env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8')

import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset
import pandas as pd
from networks import GAT, PPGAT
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.data import DataLoader
import random
import pickle
from scipy.stats import wilcoxon
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from reduceGraph import mol_to_graph, graph_to_pyg, reduce_graph_from_mol, mol_to_pool_idx, graph_to_pyg_oh, reduce_graph_from_mol_oh
import argparse


parser = argparse.ArgumentParser(description="Run GAT/PPGAT experiments with a specified number of trials.")
parser.add_argument("--trials", type=int, default=5, help="Number of runs for each target")
args = parser.parse_args()

trials = args.trials
print(f"Number of trials set to: {trials}")



#set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)


def analogue_series_split(data, seed=42, test_size=0.3, n_splits=10, n_cpds_tolerance=5):
    
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for idx, core in enumerate(data.core):
        scaffolds[core].append(idx)
    
    n_total_test = int(np.floor(test_size * len(data)))
    rng = np.random.RandomState(seed)
    for i in range(n_splits):

        scaffold_list = list(scaffolds.values())
        scaffold_indices = rng.permutation(len(scaffold_list))
        scaffold_sets = [scaffold_list[i] for i in scaffold_indices]

        train_index = []
        test_index = []

        for scaffold_set in scaffold_sets:
            if len(test_index) + len(scaffold_set) <= n_total_test:
                test_index.extend(scaffold_set)
            else:
                train_index.extend(scaffold_set)
        
        # Check for tolerance requirement
        if np.abs(len(test_index) - n_total_test) <= n_cpds_tolerance:
            print(f"Split {i} meets tolerance.")
            print(f"Train proportion: {len(train_index)/len(data):.2f}")
            print(f"Test proportion: {len(test_index)/len(data):.2f}")
            yield train_index, test_index
        else:
            print(f"Warning: Split {i} does not meet tolerance. Test size: {len(test_index)}, Expected: {n_total_test}")
            yield train_index, test_index




def smiles_to_data(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  
    graph = mol_to_graph(mol)   # mol to networkx 
    data = graph_to_pyg_oh(graph)  # networx graph to pytorch geometric graph
    data.y = torch.tensor([label], dtype=torch.float) #add label 
    return data

def dataframe_to_pyg_dataset(df, smiles_col='nonstereo_aromatic_smiles', label_col='label'):
    data_list = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        label = row[label_col]
        data = smiles_to_data(smiles, label)
        if data is not None:
            data.smiles = smiles
            data_list.append(data)

    return data_list

def smiles_to_rgdata(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  
    data = reduce_graph_from_mol_oh(mol)  # pytorch geometric RG graph from mol 
    data.y = torch.tensor([label], dtype=torch.float) #add label 
    return data
def dataframe_to_rg_pyg_dataset(df, smiles_col='nonstereo_aromatic_smiles', label_col='label'):
    data_list = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        label = row[label_col]
        data = smiles_to_rgdata(smiles, label)
        if data is not None:
            #add smiles to dataset 
            data.smiles = smiles            
            data_list.append(data)

    return data_list

def smiles_to_ppgat_data(smiles, label):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None  
    G = mol_to_graph(mol)   # mol to networkx 
    data = graph_to_pyg_oh(G)  # networx graph to pytorch geometric graph
    data.y = torch.tensor([label], dtype=torch.float) #add label 
    pharma_index, new_edge_index, new_edge_attr = mol_to_pool_idx(mol)
    data.pharma_index = pharma_index
    data.new_edge_index = new_edge_index
    data.new_edge_attr = new_edge_attr
    return data

def dataframe_to_ppgat_pyg_dataset(df, smiles_col='nonstereo_aromatic_smiles', label_col='label'):
    data_list = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        label = row[label_col]
        data = smiles_to_ppgat_data(smiles, label)
        if data is not None:
            data.smiles = smiles
            data_list.append(data)

    return data_list

def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            probs = torch.sigmoid(out).view(-1).cpu().numpy()
            labels = data.y.view(-1).cpu().numpy()

            # Ensure probs and labels are arrays, even for batch size 1
            probs = np.atleast_1d(probs)
            labels = np.atleast_1d(labels)

            all_probs.extend(probs)
            all_labels.extend(labels)

    y_pred = (np.array(all_probs) > 0.5).astype(int)
    acc = accuracy_score(all_labels, y_pred)
    auroc = roc_auc_score(all_labels, all_probs)
    return acc, auroc

def subset_by_smiles(dataset, smiles_subset):
    smiles_set = set(smiles_subset)
    return [data for data in dataset if data.smiles in smiles_set]


def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            probs = torch.sigmoid(out).view(-1).cpu().numpy()
            labels = data.y.view(-1).cpu().numpy()

            # Ensure probs and labels are arrays, even for batch size 1
            probs = np.atleast_1d(probs)
            labels = np.atleast_1d(labels)

            all_probs.extend(probs)
            all_labels.extend(labels)

    y_pred = (np.array(all_probs) > 0.5).astype(int)
    acc = accuracy_score(all_labels, y_pred)
    auroc = roc_auc_score(all_labels, all_probs)
    return acc, auroc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with open('datasets/df_random_vs_st_dataset_revised.pkl', 'rb') as f:
    all_data = pickle.load(f)

targets = all_data['accession'].unique()



GAT_acc_runs = {}
GAT_rg_acc_runs = {}
PPGAT_acc_runs = {}

GAT_auroc_runs = {}
GAT_rg_auroc_runs = {}
PPGAT_auroc_runs = {}


for target in targets:
    print("TARGET:", target)

    df_data = pd.read_csv(f"analogue_series_split/df_ccr_results_{target}.csv")

    #
    # CREATE SPLITS 
    df_train_all_splits = pd.DataFrame()
    df_test_all_splits = pd.DataFrame()

    for trial, (train_index, test_index) in enumerate(
        analogue_series_split(df_data, seed=42, test_size=0.2, n_splits=trials)
    ):
        df_train = df_data.iloc[train_index].copy()
        df_test  = df_data.iloc[test_index].copy()

        df_train["trial"] = trial
        df_test["trial"]  = trial

        df_train_all_splits = pd.concat([df_train_all_splits, df_train])
        df_test_all_splits  = pd.concat([df_test_all_splits, df_test])

    # STORE SCORES
    acc_scores = []
    auroc_scores = []

    acc_scores_rg = []
    auroc_scores_rg= []

    acc_scores_ppgat = []
    auroc_scores_ppgat = []

    #inactive data
    filtered_data = all_data[all_data['accession'] == target]


    # MAIN RUN LOOP 
    for run in range(trials):
        print("RUN:", run)

        train_val_set = df_train_all_splits[df_train_all_splits['trial'] == run] 
        test_set = df_test_all_splits[df_test_all_splits['trial'] == run].copy()

        # get val split
        train_set = pd.DataFrame()
        val_set= pd.DataFrame()

        for trial, (train_index, val_index) in enumerate(
            analogue_series_split(train_val_set, seed=42, test_size=0.1, n_splits=1)
        ):
            df_train = train_val_set.iloc[train_index].copy()
            df_val   = train_val_set.iloc[val_index].copy()

            train_set = pd.concat([train_set, df_train])
            val_set   = pd.concat([val_set, df_val])

        # labels
        test_set["label"] = 1
        train_set["label"] = 1
        val_set["label"] = 1

        # inactive data
        
        inactive_data = filtered_data[filtered_data['label'] == 0].copy()

        inactive_data = inactive_data.rename(columns={
            "chembl_cid": "cid",
            "chembl_tid": "tid",
            "nonstereo_aromatic_smiles": "smiles"
        })

        n_train_active = len(train_set)
        n_test_active  = len(test_set)
        n_val_active   = len(val_set)

        total_needed = n_train_active + n_test_active + n_val_active

        inactive_sample = inactive_data.sample(
            n=total_needed,
            replace=False,
            random_state= 42 + run
        )

        inactive_train = inactive_sample.iloc[:n_train_active]
        inactive_test  = inactive_sample.iloc[n_train_active : n_train_active + n_test_active]
        inactive_val   = inactive_sample.iloc[n_train_active + n_test_active:]

        train_balanced = pd.concat([train_set, inactive_train]).reset_index(drop=True)
        test_balanced  = pd.concat([test_set, inactive_test]).reset_index(drop=True)
        val_balanced   = pd.concat([val_set, inactive_val]).reset_index(drop=True)

        # datasets

        dataset = torch.load(f'datasets/Gdatasets/{target}_dataset.pt', weights_only=False)
        rg_dataset = torch.load(f'datasets/RGdatasets/{target}_RG_dataset.pt', weights_only=False)
        ppgat_dataset = torch.load(f'datasets/PPGATdatasets/{target}_PPGAT_dataset.pt', weights_only=False)


        test_set = subset_by_smiles(dataset, test_balanced.smiles)
        train_set = subset_by_smiles(dataset, train_balanced.smiles)
        val_set = subset_by_smiles(dataset, val_balanced.smiles)

        test_rg = subset_by_smiles(rg_dataset, test_balanced.smiles)
        train_rg = subset_by_smiles(rg_dataset, train_balanced.smiles)
        val_rg = subset_by_smiles(rg_dataset, val_balanced.smiles)

        test_ppgat = subset_by_smiles(ppgat_dataset, test_balanced.smiles)
        train_ppgat = subset_by_smiles(ppgat_dataset, train_balanced.smiles)
        val_ppgat = subset_by_smiles(ppgat_dataset, val_balanced.smiles)

  

        # GAT
        print("GAT")


        #get best configurations:

        with open(f"results/models/gat/{target}/best_config.json", "r") as f:
            best_config = json.load(f)

        lr = best_config["lr"]
        batch_size = best_config["batch_size"]

        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader  = DataLoader(test_set, batch_size=batch_size)
        val_loader   = DataLoader(val_set, batch_size=batch_size)

        model = GAT(
            in_channels=train_set[0].x.shape[1],
            edge_attr_dim=train_set[0].edge_attr.shape[1] if train_set[0].edge_attr is not None else 0,
            hidden_channels=64,
            out_channels=1,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data).view(-1), data.y.float())
                loss.backward()
                optimizer.step()

        final_acc, final_auroc = evaluate_model(model, test_loader, device)
        acc_scores.append(final_acc)
        auroc_scores.append(final_auroc)

        # GAT-rg
        print("GAT-rg")

        #get best configurations:

        with open(f"results/models/gat_rg/{target}/best_config.json", "r") as f:
            best_config = json.load(f)

        lr = best_config["lr"]
        batch_size = best_config["batch_size"]

        train_loader = DataLoader(train_rg, batch_size=batch_size)
        test_loader  = DataLoader(test_rg, batch_size=batch_size)

        model = GAT(
            in_channels=train_rg[0].x.shape[1],
            edge_attr_dim=train_rg[0].edge_attr.shape[1] if train_rg[0].edge_attr is not None else 0,
            hidden_channels=64,
            out_channels=1,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data).view(-1), data.y.float())
                loss.backward()
                optimizer.step()

        final_acc, final_auroc = evaluate_model(model, test_loader, device)
        acc_scores_rg.append(final_acc)
        auroc_scores_rg.append(final_auroc)

        # PPGAT
        print("PPGAT")

        #get best configuration 
        with open(f"results/models/ppgat/{target}/best_config.json", "r") as f:
            best_config = json.load(f)

        lr = best_config["lr"]
        batch_size = best_config["batch_size"]

        train_loader = DataLoader(train_ppgat, batch_size=batch_size)
        test_loader  = DataLoader(test_ppgat, batch_size=batch_size)

        model = PPGAT(
            in_channels=train_ppgat[0].x.shape[1],
            edge_attr_dim=train_ppgat[0].edge_attr.shape[1] if train_ppgat[0].edge_attr is not None else 0,
            hidden_channels=64,
            out_channels=1,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data).view(-1), data.y.float())
                loss.backward()
                optimizer.step()

        final_acc, final_auroc = evaluate_model(model, test_loader, device)
        acc_scores_ppgat.append(final_acc)
        auroc_scores_ppgat.append(final_auroc)


    # STORE ALL RUNS PER TARGET (for stat test)
    GAT_acc_runs[target] = acc_scores
    GAT_auroc_runs[target] = auroc_scores

    GAT_rg_acc_runs[target] = acc_scores_rg
    GAT_rg_auroc_runs[target] = auroc_scores_rg

    PPGAT_acc_runs[target] = acc_scores_ppgat
    PPGAT_auroc_runs[target] = auroc_scores_ppgat

#save file
rows = []

for target in targets:
    for run in range(trials):
        rows.append({
            "target": target,
            "run": run,

            "GAT_acc": GAT_acc_runs[target][run],
            "GAT_auroc": GAT_auroc_runs[target][run],

            "GAT_rg_acc": GAT_rg_acc_runs[target][run],
            "GAT_rg_auroc": GAT_rg_auroc_runs[target][run],

            "PPGAT_acc": PPGAT_acc_runs[target][run],
            "PPGAT_auroc": PPGAT_auroc_runs[target][run],
        })

runs_df = pd.DataFrame(rows)

runs_df.to_csv("analogue_series_runs.csv", index=False)


# FINAL TABLE
results = []

for target in targets:
    results.append({
        "target": target,

        "GAT_acc_mean": np.mean(GAT_acc_runs[target]),
        "GAT_acc_std":  np.std(GAT_acc_runs[target]),
        "GAT_auroc_mean": np.mean(GAT_auroc_runs[target]),
        "GAT_auroc_std":  np.std(GAT_auroc_runs[target]),

        "GAT_rg_acc_mean": np.mean(GAT_rg_acc_runs[target]),
        "GAT_rg_acc_std":  np.std(GAT_rg_acc_runs[target]),
        "GAT_rg_auroc_mean": np.mean(GAT_rg_auroc_runs[target]),
        "GAT_rg_auroc_std":  np.std(GAT_rg_auroc_runs[target]),

        "PPGAT_acc_mean": np.mean(PPGAT_acc_runs[target]),
        "PPGAT_acc_std":  np.std(PPGAT_acc_runs[target]),
        "PPGAT_auroc_mean": np.mean(PPGAT_auroc_runs[target]),
        "PPGAT_auroc_std":  np.std(PPGAT_auroc_runs[target]),
    })

results_df = pd.DataFrame(results)


results_df.to_csv("results_analogue_series_split.csv", index=False)



#For random splits 
#for statistical test: STORE ALL RUN RESULTS 
GAT_acc_runs_random = {}
GAT_rg_acc_runs_random = {}
PPGAT_acc_runs_random = {}

GAT_auroc_runs_random = {}
GAT_rg_auroc_runs_random = {}
PPGAT_auroc_runs_random = {}



for target in targets:
    print("TARGET:", target)

    dataset = torch.load(f'datasets/Gdatasets/{target}_dataset.pt', weights_only=False)
    rg_dataset = torch.load(f'datasets/RGdatasets/{target}_RG_dataset.pt', weights_only=False)
    ppgat_dataset = torch.load(f'datasets/PPGATdatasets/{target}_PPGAT_dataset.pt', weights_only=False)


    acc_scores = []
    auroc_scores = []

    acc_scores_rg = []
    auroc_scores_rg= []

    acc_scores_ppgat = []
    auroc_scores_ppgat = []


    for run in range(trials):
        print("RUN:", run)
            
        n = len(dataset)
        indices = np.arange(n)

        #independent run
        np.random.seed(42 + run) 
        np.random.shuffle(indices)

        split_80 = int(0.8 * n)
        train_val_idx = indices[:split_80]
        test_idx = indices[split_80:]
        split_90 = int(0.9 * len(train_val_idx))
        train_idx = train_val_idx[:split_90]
        val_idx = train_val_idx[split_90:]


        # GAT 
        train_set = [dataset[i] for i in train_idx]
        val_set   = [dataset[i] for i in val_idx]
        test_set  = [dataset[i] for i in test_idx]

        # RG 
        train_rg = [rg_dataset[i] for i in train_idx]
        val_rg   = [rg_dataset[i] for i in val_idx]
        test_rg  = [rg_dataset[i] for i in test_idx]

        # PPGAT 
        train_ppgat = [ppgat_dataset[i] for i in train_idx]
        val_ppgat   = [ppgat_dataset[i] for i in val_idx]
        test_ppgat  = [ppgat_dataset[i] for i in test_idx]

        # GAT
        print("GAT")
        
        #get best configuration 
        with open(f"results/models/gat/{target}/best_config.json", "r") as f:
            best_config = json.load(f)

        lr = best_config["lr"]
        batch_size = best_config["batch_size"]

        
        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader  = DataLoader(test_set, batch_size=batch_size)
        val_loader   = DataLoader(val_set, batch_size=batch_size)

        model = GAT(
            in_channels=train_set[0].x.shape[1],
            edge_attr_dim=train_set[0].edge_attr.shape[1] if train_set[0].edge_attr is not None else 0,
            hidden_channels=64,
            out_channels=1,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data).view(-1), data.y.float())
                loss.backward()
                optimizer.step()

        final_acc, final_auroc = evaluate_model(model, test_loader, device)
        acc_scores.append(final_acc)
        auroc_scores.append(final_auroc)

        # GAT-rg
        print("GAT-rg")

        #get best configuration 
        with open(f"results/models/gat_rg/{target}/best_config.json", "r") as f:
            best_config = json.load(f)

        lr = best_config["lr"]
        batch_size = best_config["batch_size"]

        train_loader = DataLoader(train_rg, batch_size=batch_size)
        test_loader  = DataLoader(test_rg, batch_size=batch_size)

        model = GAT(
            in_channels=train_rg[0].x.shape[1],
            edge_attr_dim=train_rg[0].edge_attr.shape[1] if train_rg[0].edge_attr is not None else 0,
            hidden_channels=64,
            out_channels=1,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data).view(-1), data.y.float())
                loss.backward()
                optimizer.step()

        final_acc, final_auroc = evaluate_model(model, test_loader, device)
        acc_scores_rg.append(final_acc)
        auroc_scores_rg.append(final_auroc)

        # PPGAT
        print("PPGAT")

        #get best configuration 
        with open(f"results/models/ppgat/{target}/best_config.json", "r") as f:
            best_config = json.load(f)

        lr = best_config["lr"]
        batch_size = best_config["batch_size"]

        train_loader = DataLoader(train_ppgat, batch_size=batch_size)
        test_loader  = DataLoader(test_ppgat, batch_size=batch_size)

        model = PPGAT(
            in_channels=train_ppgat[0].x.shape[1],
            edge_attr_dim=train_ppgat[0].edge_attr.shape[1] if train_ppgat[0].edge_attr is not None else 0,
            hidden_channels=64,
            out_channels=1,
            heads=4
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 101):
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data).view(-1), data.y.float())
                loss.backward()
                optimizer.step()

        final_acc, final_auroc = evaluate_model(model, test_loader, device)
        acc_scores_ppgat.append(final_acc)
        auroc_scores_ppgat.append(final_auroc)


    # Store runs (for stat test)
    GAT_acc_runs_random[target] = acc_scores
    GAT_auroc_runs_random[target] = auroc_scores

    GAT_rg_acc_runs_random[target] = acc_scores_rg
    GAT_rg_auroc_runs_random[target] = auroc_scores_rg

    PPGAT_acc_runs_random[target] = acc_scores_ppgat
    PPGAT_auroc_runs_random[target] = auroc_scores_ppgat

#SAVE
rows = []

for target in targets:
    for run in range(trials):
        rows.append({
            "target": target,
            "run": run,

            "GAT_acc": GAT_acc_runs_random[target][run],
            "GAT_auroc": GAT_auroc_runs_random[target][run],

            "GAT_rg_acc": GAT_rg_acc_runs_random[target][run],
            "GAT_rg_auroc": GAT_rg_auroc_runs_random[target][run],

            "PPGAT_acc": PPGAT_acc_runs_random[target][run],
            "PPGAT_auroc": PPGAT_auroc_runs_random[target][run],
        })

runs_df = pd.DataFrame(rows)

runs_df.to_csv("random_split_runs.csv", index=False)

# FINAL TABLE
results = []

for target in targets:
    results.append({
        "target": target,

        "GAT_acc_mean": np.mean(GAT_acc_runs_random[target]),
        "GAT_acc_std":  np.std(GAT_acc_runs_random[target]),
        "GAT_auroc_mean": np.mean(GAT_auroc_runs_random[target]),
        "GAT_auroc_std":  np.std(GAT_auroc_runs_random[target]),

        "GAT_rg_acc_mean": np.mean(GAT_rg_acc_runs_random[target]),
        "GAT_rg_acc_std":  np.std(GAT_rg_acc_runs_random[target]),
        "GAT_rg_auroc_mean": np.mean(GAT_rg_auroc_runs_random[target]),
        "GAT_rg_auroc_std":  np.std(GAT_rg_auroc_runs_random[target]),

        "PPGAT_acc_mean": np.mean(PPGAT_acc_runs_random[target]),
        "PPGAT_acc_std":  np.std(PPGAT_acc_runs_random[target]),
        "PPGAT_auroc_mean": np.mean(PPGAT_auroc_runs_random[target]),
        "PPGAT_auroc_std":  np.std(PPGAT_auroc_runs_random[target]),
    })

results_df = pd.DataFrame(results)
print(results_df)


results_df.to_csv("results_random_splits.csv", index=False)


