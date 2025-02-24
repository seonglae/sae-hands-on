#!/usr/bin/env python
import os
import re
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from os.path import join
from feat_match import load_sae
from convert import convert

sns.set(style="whitegrid")
LAYER_PALETTE = sns.color_palette("viridis", 12)  # 12-color palette

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


# Load weight from SAE model by loading checkpoint with load_sae and returning W_dec.
def load_weight(sae_path, local=True, site=None, device="cpu"):
    sae = load_sae(sae_path, local, site, device)
    weight = sae.W_dec.detach().cpu().numpy()
    return weight

# Compute UMAP embedding.
def compute_umap(data):
    reducer = umap.UMAP()
    return reducer.fit_transform(data)

# Helper: convert figure canvas to RGB image array.
def fig_to_img(fig):
    fig.canvas.draw()
    s = fig.canvas.tostring_argb()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(s, dtype="uint8").reshape((h, w, 4))
    # Convert ARGB to RGB by dropping the alpha channel.
    return img[:, :, 1:]

# Plot UMAP embeddings in subplots for each layer.
def plot_umap_subplots(group, out_path, title_prefix="UMAP Projection", local=True, site=None, device="cpu"):
    group = sorted(group, key=lambda x: x[0])
    weights = [load_weight(path, local, site, device) for _, path in group]
    embeddings = [compute_umap(weights) for weights in weights]

    n = len(group); n_cols = 4; n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = np.array(axs).flatten()
    for i, (layer, path) in enumerate(group):
        weight = weights[i]
        embedding = embeddings[i]
        ax = axs[i]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, color=LAYER_PALETTE[i % 12], alpha=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(False)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.suptitle(f"{title_prefix} (Sparsity {group[0][1] if group else ''})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close()
    return embeddings, weights

def create_transition_gif(group, weights, embeddings, out_path, total_frames=60, local=True, site=None, device="cpu"):
    # Compute global x and y limits across all embeddings.
    all_points = np.concatenate(embeddings, axis=0)
    global_min_x, global_max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    global_min_y, global_max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    frames = []
    n_sections = len(embeddings) - 1
    frames_per_section = total_frames // n_sections if n_sections > 0 else total_frames
    for i in range(n_sections):
        emb_start = embeddings[i]
        emb_end = embeddings[i+1]
        n_points = min(emb_start.shape[0], emb_end.shape[0])
        emb_start, emb_end = emb_start[:n_points], emb_end[:n_points]
        # Define start and end colors from the palette.
        color_start = np.array(LAYER_PALETTE[i % len(LAYER_PALETTE)])
        color_end = np.array(LAYER_PALETTE[(i+1) % len(LAYER_PALETTE)])
        for t in np.linspace(0, 1, frames_per_section, endpoint=False):
            interp = (1-t) * emb_start + t * emb_end
            interp_color = (1-t) * color_start + t * color_end  # interpolate color between layers
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(interp[:, 0], interp[:, 1], s=5, color=interp_color, alpha=0.7)
            ax.set_title(f"Transition: Layer {group[i][0]} → {group[i+1][0]}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)
            ax.grid(False)
            plt.tight_layout()
            img = fig_to_img(fig)
            frames.append(img)
            plt.close(fig)
    # Append a fixed frame for the last layer using its palette color.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embeddings[-1][:, 0], embeddings[-1][:, 1], s=5, color=LAYER_PALETTE[len(group)-1], alpha=0.7)
    ax.set_title(f"Layer {group[-1][0]}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xlim(global_min_x, global_max_x)
    ax.set_ylim(global_min_y, global_max_y)
    ax.grid(False)
    plt.tight_layout()
    img = fig_to_img(fig)
    frames.append(img)
    plt.close(fig)
    imageio.mimsave(out_path, frames, fps=10)

def plot_combined_umap(group, weights, embeddings, out_path):
    sorted_group = sorted(group, key=lambda x: x[0])
    labels = []
    for i, (layer, path) in enumerate(sorted_group):
        labels.extend([i] * weights[i].shape[0])
    combined_weights = np.concatenate(weights, axis=0)
    embedding = compute_umap(combined_weights)
    # Map each point's label to its color from the palette.
    colors = [LAYER_PALETTE[i % len(LAYER_PALETTE)] for i in labels]
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=colors, alpha=0.7)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Combined UMAP for all layers")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    model_indices = range(12)
    base = f"./checkpoints/gpt2-small_blocks.{{}}.hook_resid_pre_12288_topk_{{}}_0.0003_49_TinyStories_24413"
    sae1_list = [base.format(i, 16) for i in model_indices]
    sae2_list = [base.format(i, 32) for i in model_indices]
    checkpoints = sae1_list + sae2_list

    group_16, group_32 = [], []
    for i, checkpoint in enumerate(sae1_list):
        group_16.append((i + 1, checkpoint))
    for i, checkpoint in enumerate(sae2_list):
        group_32.append((i + 1, checkpoint))
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if group_16:
        embeddings, weights = plot_umap_subplots(group_16, join(output_dir, "umap_16.png"), local=True, site=None, device=device)
        plot_combined_umap(group_16, weights, embeddings, join(output_dir, "umap_combined_16.png"))
        create_transition_gif(group_16, weights, embeddings, join(output_dir, "umap_transition_16.gif"), local=True, site=None, device=device)
    if group_32:
        embeddings, weights = plot_umap_subplots(group_32, join(output_dir, "umap_32.png"), local=True, site=None, device=device)
        plot_combined_umap(group_32, weights, embeddings, join(output_dir, "umap_combined_32.png"))
        create_transition_gif(group_32, weights, embeddings, join(output_dir, "umap_transition_32.gif"), local=True, site=None, device=device)

if __name__ == "__main__":
    main()
