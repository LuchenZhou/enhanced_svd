#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =================================================
CELL_IN   = 1.15  
PAD_IN    = 1.6   
MAX_TICKS = 6     
GRID_COLOR = "#B0B0B0"
GRID_LW    = 0.8
OUT_PDF    = "head_score_heatmaps_morandi_BIG.pdf"


SUP_FS   = 110   
TITLE_FS = 98   
LABEL_FS = 98   
TICK_FS  = 98   
CB_FS    = 98   
# ====================================================================


plt.rcParams.update({
    "font.size": 24,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def load_head_scores(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = [tuple(map(int, k.split('-'))) for k in data.keys()]
    layers = sorted({i for i, _ in pairs})
    heads  = sorted({j for _, j in pairs})
    mat = np.zeros((len(heads), len(layers)), dtype=float)
    for k, scores in data.items():
        i, j = map(int, k.split('-'))
        mat[heads.index(j), layers.index(i)] = np.mean(scores)
    return mat, layers, heads

def make_morandi_cmap():
    colors = [
        "#A8E0F5",  
        "#2F7945",  
        "#F0AA8F",  
        "#E7ACCC",  
        "#C43D23",  # high
    ]
    return LinearSegmentedColormap.from_list("morandi", colors, N=256)

def main():

    files_and_titles = [
        ("head_score/Llama-2-7B-32K-Instruct.json",            "Llama-2-7B-32K-Instruct"),
        ("head_score/Llama-3-8B-Instruct-Gradient-1048k.json", "Llama-3-8B-1048K"),
        ("head_score/Llama-3-8B-Instruct-Gradient-4194k.json", "Llama-3-8B-4194K"),
        ("head_score/Mistral-7B-Instruct-v0.3.json",           "Mistral-7B-v0.3"),
    ]

    loaded = []
    W_max, H_max = 0, 0
    for fpath, title in files_and_titles:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Cannot find: {fpath}")
        mat, layers, heads = load_head_scores(fpath)
        loaded.append((mat, layers, heads, title))
        H, W = mat.shape
        H_max = max(H_max, H)
        W_max = max(W_max, W)

    # 2×2 
    ncols, nrows = 2, 2
    fig_w = ncols * (W_max * CELL_IN) + PAD_IN * 2
    fig_h = nrows * (H_max * CELL_IN) + PAD_IN * 2

    cmap = make_morandi_cmap()
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    fig.suptitle("Retrieval Head Score Heatmaps", fontsize=SUP_FS, fontweight="bold")

    xstep_default = max(1, W_max // MAX_TICKS)
    ystep_default = max(1, H_max // MAX_TICKS)

    im = None
    for idx, (mat, layers, heads, title) in enumerate(loaded):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        H, W = mat.shape
        xstep = max(1, W // MAX_TICKS) if W != W_max else xstep_default
        ystep = max(1, H // MAX_TICKS) if H != H_max else ystep_default

        im = ax.imshow(
            mat,
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            interpolation="nearest",  
            aspect="equal"            
        )

        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=18)

        ax.set_xticks(np.arange(0, W, xstep))
        ax.set_xticklabels([layers[i] for i in range(0, W, xstep)],
                           rotation=90, fontsize=TICK_FS, fontweight="bold")
        ax.set_yticks(np.arange(0, H, ystep))
        ax.set_yticklabels([heads[i] for i in range(0, H, ystep)],
                           fontsize=TICK_FS, fontweight="bold")

        ax.tick_params(axis="both", length=0, width=0)

        ax.set_xticks(np.arange(-.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-.5, H, 1), minor=True)
        ax.grid(which="minor", color=GRID_COLOR, linewidth=GRID_LW)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
            spine.set_color("#7A7A7A")

        if row == nrows - 1:
            ax.set_xlabel("Layer", fontsize=LABEL_FS, fontweight="bold", labelpad=14)
        if col == 0:
            ax.set_ylabel("Head Id", fontsize=LABEL_FS, fontweight="bold", labelpad=14)

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean Head Score", fontsize=CB_FS, fontweight="bold", labelpad=14)
    cb.ax.tick_params(labelsize=CB_FS, width=0)

    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"✅ saved {OUT_PDF}")

if __name__ == "__main__":
    main()
