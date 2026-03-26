import pandas as pd
from rdkit import Chem
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.model_selection import StratifiedKFold
import sys
import os
from reduceGraph import get_rg_edges
from networks import GAT, PPGAT
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients
from PIL import Image
from io import BytesIO 
from IPython.display import display
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.cm as cm
from rdkit.Chem import Draw
import numpy as np
from matplotlib import cm
import io
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.explain.config import ExplanationType
from edgeshaper import edgeshaper
import edgeshaper
import json
import random
import networkx as nx
from reduceGraph import reduce_graph_from_mol_nx, plot_pharma_graph, plot_mol_with_pharma_circles
from scipy.ndimage import gaussian_filter
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO
from rdkit.Chem import AllChem
from matplotlib.patches import Ellipse
from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
from io import BytesIO
import tempfile



def visualize_graph_with_edge_importance(
    G, edge_importance=None, edge_index=None, pos=None, figsize=(8, 8)
):
   

    # --- Node labels ---
    labels = {n: G.nodes[n]['group'].rsplit('_', 1)[0] for n in G.nodes}

    # --- Node colors ---
    atom_color_map = {
        'Zn': '#B2BEB5', 'Hf': '#B2BEB5', 'Sc': '#B2BEB5',
        'Co': '#99AFD7', 'Ta': '#99AFD7', 'Ti': '#99AFD7',
        'Ni': '#F1BD78', 'W': '#F6CF68', 'V': '#F1BD78',
        'Cu': '#8C819A', 'Re': '#8C819A', 'Cr': '#8C819A',
        'Mo': '#F6CF68', 'Zr': '#F6CF68', 'Fe': '#F6CF68',
        'Nb': '#9CCE8D', 'Y': '#9CCE8D', 'Mn': '#9CCE8D'
    }
    default_color = 'lightgrey'
    node_colors = [atom_color_map.get(labels[n], default_color) for n in G.nodes]

    # --- Layout ---
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    # sanitize NaN/inf positions
    for n, (x, y) in pos.items():
        if not np.isfinite(x) or not np.isfinite(y):
            pos[n] = (0.0, 0.0)

    # --- Draw ---
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           edgecolors='black', linewidths=1,
                           node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # --- Edge importance overlay ---
    if edge_importance is not None and edge_index is not None:
        edge_dict = {}
        for (u, v), w in zip(edge_index.T, edge_importance):
            key = tuple(sorted((int(u), int(v))))
            edge_dict.setdefault(key, []).append(float(w))

        edges, importances = [], []
        for (u, v), ws in edge_dict.items():
            edges.append((u, v))
            importances.append(np.mean(ws))
        importances = np.array(importances, dtype=float)

        if importances.size > 0:
            max_abs = np.max(np.abs(importances)) + 1e-8
            importances = importances / max_abs

            # map importance to color
            edge_colors = []
            for w in importances:
                if w >= 0:
                    color = (1.0, 1.0 - abs(w), 1.0 - abs(w))  # red
                else:
                    color = (1.0 - abs(w), 1.0 - abs(w), 1.0)  # blue
                edge_colors.append(color)

            nx.draw_networkx_edges(G, pos, edgelist=edges,
                                   edge_color=edge_colors, width=4, ax=ax)

    # --- Base black edges underlay ---
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1, ax=ax)

    # --- Adjust view ---
    all_coords = np.array(list(pos.values()))
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)

    # prevent zero-span or collapsed figures
    if xmax - xmin < 1e-6:
        xmax, xmin = xmin + 1, xmin - 1
    if ymax - ymin < 1e-6:
        ymax, ymin = ymin + 1, ymin - 1

    x_pad, y_pad = (xmax - xmin) * 0.2, (ymax - ymin) * 0.2
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Safe figure-to-image conversion ---
    buf = BytesIO()
    try:
        fig.savefig(buf, format='png', dpi=150)
    except Exception as e:
        print(f"[Warning] savefig failed: {e}. Retrying with tempfile...")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, format='png', dpi=150)
            tmp.seek(0)
            buf.write(tmp.read())
    finally:
        plt.close(fig)

    buf.seek(0)
    img = Image.open(buf)
    img.load()  # ensure full image load into memory

    # strip problematic metadata
    img.info = {k: str(v) for k, v in img.info.items() if isinstance(v, (str, int, float))}

    return img

def visualize_molecule_with_edge_importance(mol, edge_importance, edge_index, figsize=(800, 800)):
    """
    Visualize a molecule with edge importance (signed) using RDKit.
    Positive -> red shade, Negative -> blue shade.
    Edges are treated as undirected, summed if both directions exist.
    Normalization is global, like visualize_molecule_with_gaussian_blurr.
    """

    # Prepare molecule for drawing
    mol = Draw.PrepareMolForDrawing(mol)

    num_bonds = len(mol.GetBonds())

    # Map atom pairs to RDKit bond indices
    rdkit_bonds = {}
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        rdkit_bonds[(u, v)] = i

    # Aggregate edge importance into bond indices (undirected)
    rdkit_bonds_phi = [0.0] * num_bonds
    for i in range(len(edge_importance)):
        u = int(edge_index[0][i])
        v = int(edge_index[1][i])
        phi = float(edge_importance[i])
        if (u, v) in rdkit_bonds:
            rdkit_bonds_phi[rdkit_bonds[(u, v)]] += phi
        if (v, u) in rdkit_bonds:
            rdkit_bonds_phi[rdkit_bonds[(v, u)]] += phi

    # Global normalization (like Gaussian blur version)
    max_abs = max(abs(x) for x in rdkit_bonds_phi) + 1e-8
    rdkit_bonds_phi = [x / max_abs for x in rdkit_bonds_phi]

    # Map normalized values to colors
    bond_colors = {}
    for i, w in enumerate(rdkit_bonds_phi):
        if w >= 0:
            # Positive -> red
            bond_colors[i] = (1.0, 1.0 - abs(w), 1.0 - abs(w))
        else:
            # Negative -> blue
            bond_colors[i] = (1.0 - abs(w), 1.0 - abs(w), 1.0)

    # RDKit drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(*figsize)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()  # force atoms to black

    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    return Image.open(BytesIO(png))

def visualize_graph_with_gaussian_blurr(
    G, edge_importance, edge_index, pos=None,
    figsize=(6,6), sigma=12, img_size=400, padding=0.2,
    boost=1.0, n_samples=30
):
    """
    Visualize a NetworkX graph with Gaussian-blurred edge importance.
    Positive contributions -> red glow
    Negative contributions -> blue glow
    Blur is distributed along the edges (tube-like).

    Normalization is global: all edges are scaled relative to the
    maximum absolute edge importance in the graph.
    """
    # Node colors
    atom_color_map = {
        'Zn': '#B2BEB5', 'Hf':'#B2BEB5', 'Sc':'#B2BEB5',
        'Co':'#99AFD7', 'Ta':'#99AFD7', 'Ti':'#99AFD7',
        'Ni':'#F1BD78', 'W':'#F6CF68', 'V':'#F1BD78',
        'Cu':'#8C819A', 'Re':'#8C819A', 'Cr':'#8C819A',
        'Mo':'#F6CF68', 'Zr':'#F6CF68', 'Fe':'#F6CF68',
        'Nb':'#9CCE8D', 'Y':'#9CCE8D', 'Mn':'#9CCE8D'
    }
    default_color = "lightgrey"
    labels = {n: G.nodes[n]['group'].rsplit('_', 1)[0] for n in G.nodes}
    node_colors = [atom_color_map.get(labels[n], default_color) for n in G.nodes]

    # Aggregate edges as undirected
    edge_dict = {}
    for (u, v), w in zip(edge_index.T, edge_importance):
        key = tuple(sorted((int(u), int(v))))
        edge_dict.setdefault(key, []).append(float(w))

    edges, importances = [], []
    for (u, v), ws in edge_dict.items():
        edges.append((u, v))
        importances.append(np.mean(ws))
    importances = np.array(importances, dtype=float)

    # Global normalization across all edges
    max_abs = np.max(np.abs(importances)) + 1e-8
    importances = importances / max_abs  # values in [-1,1]

    # Layout
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           edgecolors="black", linewidths=1,
                           node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="black", width=1.0, ax=ax)
    ax.axis("off")

    # Expand limits
    all_coords = np.array(list(pos.values()))
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    x_pad, y_pad = (xmax - xmin) * padding, (ymax - ymin) * padding
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)

    # Heatmap grid
    width, height = img_size, img_size
    heat_r = np.zeros((height, width))
    heat_b = np.zeros((height, width))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def to_px(x, y):
        px = int((x - xlim[0]) / (xlim[1]-xlim[0]) * (width-1))
        py = int((y - ylim[0]) / (ylim[1]-ylim[0]) * (height-1))
        return px, py

    # Precompute circular Gaussian kernel
    kernel_size = int(6*sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    xx, yy = np.meshgrid(np.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                         np.linspace(-kernel_size//2, kernel_size//2, kernel_size))
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel /= kernel.max()

    # Add Gaussian contributions along edges
    for (u, v), w in zip(edges, importances):
        if abs(w) < 1e-8:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xs = np.linspace(x0, x1, n_samples)
        ys = np.linspace(y0, y1, n_samples)
        for x, y in zip(xs, ys):
            px, py = to_px(x, y)
            x1_idx = max(0, px - kernel_size//2)
            x2_idx = min(width, px + kernel_size//2 + 1)
            y1_idx = max(0, py - kernel_size//2)
            y2_idx = min(height, py + kernel_size//2 + 1)

            kx1 = max(0, kernel_size//2 - px)
            kx2 = kx1 + (x2_idx - x1_idx)
            ky1 = max(0, kernel_size//2 - py)
            ky2 = ky1 + (y2_idx - y1_idx)

            if w > 0:
                heat_r[y1_idx:y2_idx, x1_idx:x2_idx] += w * kernel[ky1:ky2, kx1:kx2]
            else:
                heat_b[y1_idx:y2_idx, x1_idx:x2_idx] += (-w) * kernel[ky1:ky2, kx1:kx2]

    # Global normalization of both red & blue
    global_max = max(heat_r.max(), heat_b.max(), 1e-8)
    heat_r = np.clip(heat_r / global_max * boost, 0, 1)
    heat_b = np.clip(heat_b / global_max * boost, 0, 1)

    # Blend into RGBA
    heat_rgb = np.zeros((height, width, 4))
    heat_rgb[...,0] = heat_r
    heat_rgb[...,2] = heat_b
    heat_rgb[...,3] = np.clip(heat_r + heat_b, 0, 1)  # alpha from combined intensity

    ax.imshow(heat_rgb, origin="lower", extent=[*xlim, *ylim], interpolation="bilinear")
    #plt.close(fig)
        # --- convert figure to PIL image ---
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    img.info.clear()  # Remove problematic metadata
    plt.close(fig)
    return img


def visualize_molecule_with_gaussian_blurr(mol, edge_importance, edge_index,
                                           atom_width=0.2, bond_length=0.5, bond_width=0.5):
    """
    Visualize molecule explanations (edge_importance) as Gaussian-blurred heatmap like EdgeSHAPer.

    mol: RDKit Mol object
    edge_importance: array-like, importance per edge (signed values allowed)
    edge_index: 2 x num_edges tensor or array mapping edges to atom indices
    atom_width, bond_length, bond_width: visualization parameters (same as EdgeSHAPer)
    """

    # Prepare molecule for drawing
    mol = Draw.PrepareMolForDrawing(mol)

    num_bonds = len(mol.GetBonds())

    # Map atom pairs to RDKit bond indices
    rdkit_bonds = {}
    for i in range(num_bonds):
        init_atom = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        end_atom = mol.GetBondWithIdx(i).GetEndAtomIdx()
        rdkit_bonds[(init_atom, end_atom)] = i

    # Aggregate edge importance into RDKit bond order
    rdkit_bonds_phi = [0.0] * num_bonds
    for i in range(len(edge_importance)):
        init_atom = int(edge_index[0][i])
        end_atom = int(edge_index[1][i])
        phi_value = float(edge_importance[i])

        if (init_atom, end_atom) in rdkit_bonds:
            rdkit_bonds_phi[rdkit_bonds[(init_atom, end_atom)]] += phi_value
        if (end_atom, init_atom) in rdkit_bonds:
            rdkit_bonds_phi[rdkit_bonds[(end_atom, init_atom)]] += phi_value

    # Call EdgeSHAPer’s Gaussian blur drawer
    plt.clf()
    canvas = mapvalues2mol(
        mol,
        None,
        rdkit_bonds_phi,
        atom_width=atom_width,
        bond_length=bond_length,
        bond_width=bond_width
    )
    img = transform2png(canvas.GetDrawingText())
    plt.clf()

    return img


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

       # Convert to PIL Image safely
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # Fix for RDKit / IPythonConsole metadata bug
    img.info = {k: str(v) for k, v in img.info.items() if isinstance(v, (str, int, float))}


    return img, RG, pos

