"""
Example usage:
    python scripts/layer_wise.py --num_samples=100
"""

import os
from os.path import join
import fire
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate, get_act_name
from convert import convert

# Palettes
LAYER_PALETTE = sns.color_palette("viridis", 12)
POSITION_PALETTE = "flare"
QUANTILE_PALETTE = "mako"

# Global arrays (12 layers for GPT-2 small)
token_means = [None] * 12
token_stds = [None] * 12
token_densities = [None] * 12
token_counts = [None] * 12
feat_means = [None] * 12
feat_stds = [None] * 12
feat_densities = [None] * 12
feat_counts = [None] * 12
quantiles_wi_first = [None] * 12
quantiles_wo_first = [None] * 12


def compute_token_stat(acts: torch.Tensor, exclude_zeros=True, chunk_size=25000):
    # Returns count, mean, std, density per token
    device = acts.device
    acts = acts.transpose(0, 1).flatten(start_dim=1)
    num_tokens, total_length = acts.shape
    token_sum = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    token_sq_sum = torch.zeros(num_tokens, dtype=torch.float32, device=device)
    token_count = torch.zeros(num_tokens, dtype=torch.int64, device=device)

    for i in tqdm(range(0, total_length, chunk_size), desc="Token Stats"):
        end = min(i + chunk_size, total_length)
        chunk = acts[:, i:end]
        if exclude_zeros:
            mask = (chunk != 0)
            maskf = mask.float()
            token_sum += (chunk * maskf).sum(dim=1)
            token_sq_sum += (chunk**2 * maskf).sum(dim=1)
            token_count += mask.sum(dim=1)
        else:
            token_sum += chunk.sum(dim=1)
            token_sq_sum += (chunk**2).sum(dim=1)
            token_count += (end - i)

    count_clamped = torch.clamp(token_count, min=1)
    mean = token_sum / count_clamped
    var = (token_sq_sum / count_clamped) - mean**2
    std = torch.sqrt(var)
    density = token_count.float() / total_length
    return token_count, mean, std, density


def compute_feat_stat(acts: torch.Tensor, exclude_zeros=True, chunk_size=1000, ignore_first=True):
    # Returns count, mean, std, density per feature
    device = acts.device
    if ignore_first:
        acts = acts[:, 1:, :]
    acts = acts.transpose(0, 2).flatten(start_dim=1)
    num_features, total_length = acts.shape
    feat_sum = torch.zeros(num_features, dtype=torch.float32, device=device)
    feat_sq_sum = torch.zeros(num_features, dtype=torch.float32, device=device)
    feat_count = torch.zeros(num_features, dtype=torch.int64, device=device)

    for i in tqdm(range(0, total_length, chunk_size), desc="Feature Stats"):
        end = min(i + chunk_size, total_length)
        chunk = acts[:, i:end]
        if exclude_zeros:
            mask = (chunk != 0)
            maskf = mask.float()
            feat_sum += (chunk * maskf).sum(dim=1)
            feat_sq_sum += (chunk**2 * maskf).sum(dim=1)
            feat_count += mask.sum(dim=1)
        else:
            feat_sum += chunk.sum(dim=1)
            feat_sq_sum += (chunk**2).sum(dim=1)
            feat_count += (end - i)

    count_clamped = torch.clamp(feat_count, min=1)
    mean = feat_sum / count_clamped
    var = (feat_sq_sum / count_clamped) - mean**2
    std = torch.sqrt(var)
    density = feat_count.float() / total_length
    return feat_count, mean, std, density


def compute_feat_quantiles(acts: torch.Tensor, level=10, ignore_first=True):
    # Returns per-feature quantiles shape (num_features, level)
    acts = acts[:, 1 if ignore_first else 0:, :]
    device = acts.device
    feat_count = acts.shape[2]
    feat_quantiles = []
    for i in tqdm(range(feat_count), desc="Computing quantiles"):
        feat_acts = acts[:, :, i].flatten()
        feat_acts = feat_acts[feat_acts != 0]
        if feat_acts.size(0) == 0:
            feat_quantiles.append(torch.zeros(level))
            continue
        feat_quantile = torch.quantile(feat_acts.float(), torch.linspace(0, 1, level + 1)[1:]).to(torch.float16)
        feat_quantiles.append(feat_quantile)
    return torch.stack(feat_quantiles)


def plot_quantiles_on_ax(ax, quantiles, log_scale=False, palette="mako"):
    num_features, num_quantiles = quantiles.shape
    q_levels = np.linspace(1, 10, num_quantiles - 1)
    data = []
    for i in reversed(range(1, num_quantiles)):
        decile_label = f"Decile {int(q_levels[i-1])}"
        for j in range(num_features):
            data.append({
                "Feature": j,
                "Activation": float(quantiles[j, i]),
                "Decile": decile_label
            })
    df = pd.DataFrame(data)
    sns.lineplot(
        data=df, x="Feature", y="Activation", hue="Decile", marker="o", markeredgewidth=0,
        palette=palette, linewidth=0, alpha=0.7, legend=False, ax=ax
    )
    if log_scale:
        ax.set_yscale('log')

def zero_hook(module, inputs, outputs):
    """ 
    Simple version of a steering hook. Adds a weighted vector
    to the residual. Customize if needed.
    """
    embedding = outputs[0]
    zero = torch.zeros_like(embedding).unsqueeze(0)
    return zero


def shuffle_hook(module, inputs, outputs):
    embedding = outputs[0]
    shuffled = embedding[torch.randperm(embedding.size(0)), :]
    return (shuffled)


def main(num_samples=1000, sae_id="jbloom/GPT2-Small-SAEs-Reformatted", llm_id="gpt2", site="resid_pre", layers=12, images_folder="images", zero=False, local=False, shuffle=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(llm_id).to(device)
    llm.eval()
    if zero:
        llm.transformer.wpe.register_forward_hook(zero_hook)
    if shuffle:
        llm.transformer.wpe.register_forward_hook(shuffle_hook)

    # Load dataset
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        streaming=True,
        max_length=1024,
        add_bos_token=True,
    )
    sample_subset = token_dataset.select(range(num_samples))["tokens"]

    # Process layer by layer
    hooks = [get_act_name(site, i) for i in range(layers)]
    for layer_i, hook in enumerate(hooks):
        print(f"\nLayer {layer_i} - Loading SAE")
        if local:
            sae, original = convert(f'./checkpoints/gpt2-small_blocks.{layer_i}.hook_resid_pre_12288_topk_16_0.0003_49_TinyStories_24413')
        else:
            sae, _, _ = SAE.from_pretrained(sae_id, hook, device=device)
        all_acts = []

        print(f"Layer {layer_i} - Gathering activations for {num_samples} sequences")
        for encoding in tqdm(sample_subset, desc=f"Layer {layer_i}"):
            with torch.no_grad():
                outputs = llm(encoding.to(device), output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_i]
                act = sae.encode(hidden_state).to(torch.float16)
                all_acts.append(act.cpu())

        print(f"Layer {layer_i} - Aggregating Tensors")
        stacked_acts = torch.cat(all_acts, dim=0).to(torch.float16) # (num_samples, num_tokens, num_features)
        del all_acts

        print(f"Layer {layer_i} - Computing token-level stats")
        t_count, t_mean, t_std, t_density = compute_token_stat(stacked_acts, exclude_zeros=True)

        print(f"Layer {layer_i} - Computing feature-level stats")
        f_count, f_mean, f_std, f_density = compute_feat_stat(stacked_acts, exclude_zeros=True)

        print(f"Layer {layer_i} - Computing feature quantiles")
        q_wi = compute_feat_quantiles(stacked_acts, level=10, ignore_first=False)
        q_wo = compute_feat_quantiles(stacked_acts, level=10, ignore_first=True)

        print(f"Layer {layer_i} - Storing condensed metrics")
        token_means[layer_i] = t_mean.cpu().numpy()
        token_stds[layer_i] = t_std.cpu().numpy()
        token_counts[layer_i] = t_count.cpu().numpy()
        token_densities[layer_i] = t_density.cpu().numpy()
        feat_means[layer_i] = f_mean.cpu().numpy()
        feat_stds[layer_i] = f_std.cpu().numpy()
        feat_counts[layer_i] = f_count.cpu().numpy()
        feat_densities[layer_i] = f_density.cpu().numpy()
        quantiles_wi_first[layer_i] = q_wi.cpu().numpy()
        quantiles_wo_first[layer_i] = q_wo.cpu().numpy()

        del stacked_acts, sae, q_wi, q_wo
        torch.cuda.empty_cache()
        print(f"Layer {layer_i} done.\n")

    # --------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------
    os.makedirs(images_folder, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 18})

    # 1) quantiles with bos
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Quantiles (With BOS)")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        q = quantiles_wi_first[layer_i]  # shape: (num_features, 10)
        plot_quantiles_on_ax(ax, q, log_scale=False, palette=QUANTILE_PALETTE)
        ax.set_title(f"Layer {layer_i}")
        ax.set_yscale('log')

    plt.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    q_levels = np.linspace(1, 10, len(handles))
    new_labels = [f"Decile {q:.0f}" for q in q_levels]
    fig.legend(handles, new_labels, title="Decile", loc="upper right")
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}quantiles_with_bos.png"))
    plt.close()

    # 2) quantiles without bos
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Quantiles (Without BOS)")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        q = quantiles_wo_first[layer_i]  # shape: (num_features, 10)
        plot_quantiles_on_ax(ax, q, log_scale=False, palette=QUANTILE_PALETTE)
        ax.set_title(f"Layer {layer_i}")
        ax.set_yscale('log')

    plt.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    q_levels = np.linspace(1, 10, len(handles))
    new_labels = [f"Decile {q:.0f}" for q in q_levels]
    fig.legend(handles, new_labels, title="Decile", loc="upper right")
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}quantiles_without_bos.png"))
    plt.close()

    # 3) token_meanvs_std
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Token Mean vs. Standard Deviation")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        tm = token_means[layer_i][1:]
        ts = token_stds[layer_i][1:]
        color_vals = np.arange(len(tm))
        sc = ax.scatter(tm, ts, c=color_vals, cmap=POSITION_PALETTE, s=8)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Mean")
        ax.set_ylabel("Standard Deviation")
        # Annotate outliers
        for idx in range(len(tm)):
            if tm[idx] > 100:
                ax.annotate(str(idx), (tm[idx], ts[idx]), xytext=(5, 5),
                            textcoords="offset points", fontsize=6, color='black')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.11, 0.01, 0.77])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Token Index')
    tick_locs = np.linspace(0, len(tm) - 1, min(len(tm), 10))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(int(x)) for x in tick_locs])
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}token_meanvs_std.png"))
    plt.close()

    # 4) avg_activation_with_bos
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Avg Activation per Token (With BOS)")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        tm = token_means[layer_i]
        x_idx = np.arange(len(tm))
        ax.plot(x_idx, tm, color="grey", alpha=0.5)
        sc = ax.scatter(x_idx, tm, c=x_idx, cmap=POSITION_PALETTE, s=8)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Activation")
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.11, 0.01, 0.77])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Token Index')
    tick_locs = np.linspace(0, len(tm) - 1, min(len(tm), 10))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(int(x)) for x in tick_locs])
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}avg_activation_with_bos.png"))
    plt.close()

    # 5) avg_activation_without_bos
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Avg Activation per Token (Without BOS)")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        tm = token_means[layer_i]
        if len(tm) > 1:
            x_idx = np.arange(len(tm) - 1)
            val_ = tm[1:]
            ax.plot(x_idx, val_, color="grey", alpha=0.5)
            sc = ax.scatter(x_idx, val_, c=x_idx, cmap=POSITION_PALETTE, s=8)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Token Index (Skipping BOS)")
        ax.set_ylabel("Activation")
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.11, 0.01, 0.77])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Token Index')
    tick_locs = np.linspace(0, len(tm) - 1, min(len(tm), 10))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(int(x)) for x in tick_locs])
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}avg_activation_without_bos.png"))
    plt.close()

    # 6) avg_activation_smoothing
    plt.figure(figsize=(12, 8))
    plt.title("Smoothed Avg Activation per Token (All Layers)")
    for layer_i in range(12):
        tm = token_means[layer_i]
        if len(tm) > 1:
            x_idx = np.arange(len(tm) - 1)
            val_ = tm[1:]
            smoothed_val = np.convolve(val_, np.ones(10)/10, mode='valid')
            plt.plot(x_idx[:len(smoothed_val)], smoothed_val, color=LAYER_PALETTE[layer_i], label=f"Layer {layer_i}", alpha=0.7)
    plt.xlabel("Token Index (Skipping BOS)")
    plt.ylabel("Smoothed Activation")
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}avg_activation_smoothing.png"))
    plt.close()

    # 7) Token count vs. token density
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Token Count vs. Token Density")
    for layer_i in range(12):
        color_vals = np.arange(len(token_means[layer_i]) - 1)
        ax = axs[layer_i // 4, layer_i % 4]
        tm = token_means[layer_i][1:]
        td = token_densities[layer_i][1:]
        if tm is not None and len(tm) == len(td):
            sc = ax.scatter(tm, td, s=12, cmap=POSITION_PALETTE, c=color_vals)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Activation Average")
        ax.set_ylabel("Token Density")
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.11, 0.01, 0.77])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Token Index')
    tick_locs = np.linspace(0, len(tm) - 1, min(len(tm), 10))
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(int(x)) for x in tick_locs])
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}token_count_vs_density.png"))
    plt.close()

    # 8) Feature count vs. feature density
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Feature Count vs. Feature Density")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        fm = feat_means[layer_i]
        fd = feat_densities[layer_i]
        ax.scatter(fm, fd, s=5, color=LAYER_PALETTE[layer_i])
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Activation Average")
        ax.set_ylabel("Feature Density")
    plt.tight_layout()
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}feat_count_vs_density.png"))
    plt.close()

    # --------------------------------------------------------------------
    # Combined across-layers: feature means + feature mean vs. std
    # --------------------------------------------------------------------
    all_feat_means = np.stack(feat_means, axis=0)  # (12, num_features)
    all_feat_stds = np.stack(feat_stds, axis=0)    # (12, num_features)
    num_features = all_feat_means.shape[1]

    # 1) Combined_feat_mean.png
    plt.figure(figsize=(12, 8))
    plt.title("Feature Mean Distribution (All Layers)")
    for layer_i in range(12):
        sns.kdeplot(
            all_feat_means[layer_i],
            color=LAYER_PALETTE[layer_i],
            label=f"Layer {layer_i}"
        )
    plt.xlabel("Feature Mean")
    plt.ylabel("Density")
    plt.xscale('log')
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}feat_mean_dist.png"))
    plt.close()

    # 2) Combined_feat_meanvs_std.png
    plt.figure(figsize=(12, 8))
    plt.title("Feature Mean vs. Standard Deviation (All Layers)")
    for layer_i in range(12):
        fm = all_feat_means[layer_i]
        fs = all_feat_stds[layer_i]
        plt.scatter(fm, fs, color=LAYER_PALETTE[layer_i], s=5, label=f"Layer {layer_i}", alpha=0.7)
    plt.xlabel("Mean")
    plt.ylabel("Standard Deviation")
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}combined_feat_meanvs_std.png"))
    plt.close()

    # 4) Activation Mean vs. Count of Features with Activation > Median (All Layers)
    plt.figure(figsize=(12, 8))
    plt.title("Count of Features with Activation > Median (All Layers)")
    for layer_i in range(12):
      q_wi = quantiles_wo_first[layer_i]
      fm = feat_means[layer_i]
      median = q_wi[:, 4]
      x = np.linspace(0, np.max(fm), 100)
      counts = []
      for threshold in x:
        count = np.sum(median < threshold)
        counts.append(count)
      plt.plot(x, counts, color=LAYER_PALETTE[layer_i], label=f"Layer {layer_i}")
    plt.xlabel("Activation Mean")
    plt.ylabel("Count > Median")
    plt.tight_layout()
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}activation_mean_vs_median.png"))
    plt.close()

    # 4) Activation Mean vs. Count of Features with Activation > Mean (All Layers)
    plt.figure(figsize=(12, 8))
    plt.title("Count of Features with Activation > Mean (All Layers)")
    for layer_i in range(12):
      fm = feat_means[layer_i]
      x = np.linspace(0, np.max(fm), 100)
      counts = []
      for threshold in x:
        count = np.sum(fm < threshold)
        counts.append(count)
      plt.plot(x, counts, color=LAYER_PALETTE[layer_i], label=f"Layer {layer_i}")
    plt.xlabel("Activation Mean")
    plt.ylabel("Count > Mean")
    plt.tight_layout()
    plt.savefig(join(images_folder, f"{'zero_' if zero else ''}{'shuffle_' if shuffle else ''}activation_mean_vs_mean.png"))
    plt.close()

    print("Done. All metrics computed and visualized.")


if __name__ == "__main__":
    fire.Fire(main)
