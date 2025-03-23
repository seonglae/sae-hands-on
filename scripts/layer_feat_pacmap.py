#!/usr/bin/env python
import os
import re
import torch
import pacmap
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
device = "mps" if torch.backends.mps.is_avimage.pngailable() else device


# Load weight from SAE model by loading checkpoint with load_sae and returning W_dec.
def load_weight(sae_path, local=True, site=None, device="cpu"):
    sae = load_sae(sae_path, local, site, device)
    weight = sae.W_dec.detach().cpu().numpy()
    return weight

# Compute PaCMAP embedding.
def compute_pacmap(data):
    reducer = pacmap.PaCMAP()
    return reducer.fit_transform(data)

# Compute unified PaCMAP embedding for all layers at once to share the same space
def compute_unified_pacmap(group):
    all_weights = []
    layer_indices = []
    layer_boundaries = [0]
    
    # Collect weights from all layers
    for layer_idx, (_, path) in enumerate(group):
        weight = load_weight(path)
        all_weights.append(weight)
        layer_indices.extend([layer_idx] * weight.shape[0])
        layer_boundaries.append(layer_boundaries[-1] + weight.shape[0])
    
    # Combine all weights into one array
    combined_weights = np.concatenate(all_weights, axis=0)
    
    # Apply PaCMAP to get unified embedding
    unified_embedding = compute_pacmap(combined_weights)
    
    # Split embedding by layer
    layer_embeddings = []
    for i in range(len(layer_boundaries) - 1):
        start, end = layer_boundaries[i], layer_boundaries[i+1]
        layer_embeddings.append(unified_embedding[start:end])
    
    return unified_embedding, layer_embeddings, layer_indices, all_weights

# Helper: convert figure canvas to RGB image array.
def fig_to_img(fig):
    fig.canvas.draw()
    s = fig.canvas.tostring_argb()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(s, dtype="uint8").reshape((h, w, 4))
    # Convert ARGB to RGB by dropping the alpha channel.
    return img[:, :, 1:]

# Plot PaCMAP embeddings in subplots for each layer using the unified embedding space
def plot_unified_pacmap_subplots(group, layer_embeddings, out_path, title_prefix="PaCMAP Projection"):
    group = sorted(group, key=lambda x: x[0])
    n = len(group); n_cols = 4; n_rows = (n + n_cols - 1) // n_cols
    
    # Calculate global limits across all embeddings
    all_points = np.concatenate(layer_embeddings, axis=0)
    global_min_x, global_max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    global_min_y, global_max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = np.array(axs).flatten()
    
    for i, ((layer, _), embedding) in enumerate(zip(group, layer_embeddings)):
        ax = axs[i]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, color=LAYER_PALETTE[i % 12], alpha=0.7)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("PaCMAP-1")
        ax.set_ylabel("PaCMAP-2")
        # Set all plots to use the same limits
        ax.set_xlim(global_min_x, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)
        ax.grid(False)
    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.suptitle(f"{title_prefix} (Sparsity {group[0][1] if group else ''})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close()

def create_unified_transition_gif(group, layer_embeddings, out_path, total_frames=60):
    # Compute global x and y limits across all embeddings
    all_points = np.concatenate(layer_embeddings, axis=0)
    global_min_x, global_max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    global_min_y, global_max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    frames = []
    n_sections = len(layer_embeddings) - 1
    frames_per_section = total_frames // n_sections if n_sections > 0 else total_frames
    
    for i in range(n_sections):
        emb_start = layer_embeddings[i]
        emb_end = layer_embeddings[i+1]
        n_points = min(emb_start.shape[0], emb_end.shape[0])
        emb_start, emb_end = emb_start[:n_points], emb_end[:n_points]
        
        # Define start and end colors from the palette
        color_start = np.array(LAYER_PALETTE[i % len(LAYER_PALETTE)])
        color_end = np.array(LAYER_PALETTE[(i+1) % len(LAYER_PALETTE)])
        
        for t in np.linspace(0, 1, frames_per_section, endpoint=False):
            interp = (1-t) * emb_start + t * emb_end
            interp_color = (1-t) * color_start + t * color_end  # interpolate color between layers
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(interp[:, 0], interp[:, 1], s=5, color=interp_color, alpha=0.7)
            ax.set_title(f"Transition: Layer {group[i][0]} → {group[i+1][0]}")
            ax.set_xlabel("PaCMAP-1")
            ax.set_ylabel("PaCMAP-2")
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)
            ax.grid(False)
            plt.tight_layout()
            
            img = fig_to_img(fig)
            frames.append(img)
            plt.close(fig)
    
    # Append a fixed frame for the last layer using its palette color
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(layer_embeddings[-1][:, 0], layer_embeddings[-1][:, 1], s=5, color=LAYER_PALETTE[len(group)-1], alpha=0.7)
    ax.set_title(f"Layer {group[-1][0]}")
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.set_xlim(global_min_x, global_max_x)
    ax.set_ylim(global_min_y, global_max_y)
    ax.grid(False)
    plt.tight_layout()
    
    img = fig_to_img(fig)
    frames.append(img)
    plt.close(fig)
    
    imageio.mimsave(out_path, frames, fps=10)

def plot_combined_pacmap(unified_embedding, layer_indices, out_path):
    # Map each point's label to its color from the palette
    colors = [LAYER_PALETTE[i % len(LAYER_PALETTE)] for i in layer_indices]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(unified_embedding[:, 0], unified_embedding[:, 1], s=5, c=colors, alpha=0.7)
    plt.xlabel("PaCMAP-1")
    plt.ylabel("PaCMAP-2")
    plt.title("Combined PaCMAP for all layers")
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
        # Compute unified PaCMAP embeddings for all layers
        unified_embedding, layer_embeddings, layer_indices, all_weights = compute_unified_pacmap(group_16)
        
        # Plot each layer in the unified embedding space
        plot_unified_pacmap_subplots(group_16, layer_embeddings, join(output_dir, "unified_pacmap_16.png"))
        
        # Plot the combined view showing all layers at once
        plot_combined_pacmap(unified_embedding, layer_indices, join(output_dir, "unified_pacmap_combined_16.png"))
        
        # Create the transition animation between layers
        create_unified_transition_gif(group_16, layer_embeddings, join(output_dir, "unified_pacmap_transition_16.gif"))
    
    if group_32:
        # Compute unified PaCMAP embeddings for all layers
        unified_embedding, layer_embeddings, layer_indices, all_weights = compute_unified_pacmap(group_32)
        
        # Plot each layer in the unified embedding space
        plot_unified_pacmap_subplots(group_32, layer_embeddings, join(output_dir, "unified_pacmap_32.png"))
        
        # Plot the combined view showing all layers at once
        plot_combined_pacmap(unified_embedding, layer_indices, join(output_dir, "unified_pacmap_combined_32.png"))
        
        # Create the transition animation between layers
        create_unified_transition_gif(group_32, layer_embeddings, join(output_dir, "unified_pacmap_transition_32.gif"))

if __name__ == "__main__":
    main()
