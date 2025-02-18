"""
Example usage:
    python scripts/layer-wise.py --num_samples=100
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

# Palettes
LAYER_PALETTE = sns.color_palette("viridis", 12)
POSITION_PALETTE = "flare"
QUANTILE_PALETTE = "mako"

# Global arrays (12 layers for GPT-2 small)
token_means = [None] * 12
token_stds = [None] * 12
token_densities = [None] * 12
token_counts = [None] * 12
sae_token_losses = [None] * 12
sae_token_l1 = [None] * 12
sae_token_l2 = [None] * 12
feat_means = [None] * 12
feat_stds = [None] * 12
feat_densities = [None] * 12
feat_counts = [None] * 12
quantiles_wi_first = [None] * 12
quantiles_wo_first = [None] * 12
llm_token_losses = None

# ------------------------------------------------------------------------
# Minimal comments, one-liners only
# ------------------------------------------------------------------------

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
        feat_quantile = torch.quantile(feat_acts, torch.linspace(0, 1, level + 1)[1:])
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
        data=df, x="Feature", y="Activation", hue="Decile",
        palette=palette, linewidth=1, alpha=0.7, legend=False, ax=ax
    )
    if log_scale:
        ax.set_yscale('log')


def main(num_samples=200, sae_id="jbloom/GPT2-Small-SAEs-Reformatted", llm_id="gpt2", site="resid_pre", layers=12, images_folder="images"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    llm = AutoModelForCausalLM.from_pretrained(llm_id).to(device)
    llm.eval()

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
        sae, _, _ = SAE.from_pretrained(sae_id, hook, device=device)
        all_acts, all_l1, all_l2, ces = [], [], [], []

        print(f"Layer {layer_i} - Gathering activations for {num_samples} sequences")
        for encoding in tqdm(sample_subset, desc=f"Layer {layer_i}"):
            with torch.no_grad():
                outputs = llm(encoding.to(device), output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_i]
                logits = outputs.logits
                shift_logits = logits[:-1, :]
                labels = encoding[1:].to(device)
                ce = F.cross_entropy(shift_logits, labels, reduction='none')
                ces.append(ce.unsqueeze(0).to(torch.float16).cpu())
                act = sae.encode(hidden_state).float()
                recon = sae.decode(act)
                act = act.to(torch.float16)
                l1 = torch.mean(torch.abs(act), dim=2)
                l2 = F.mse_loss(hidden_state, recon, reduction='none').mean(dim=2)
                all_acts.append(act.cpu())
                all_l1.append(l1.unsqueeze(0).to(torch.float16).cpu())
                all_l2.append(l2.unsqueeze(0).to(torch.float16).cpu())

        print(f"Layer {layer_i} - Aggregating Tensors")
        stacked_acts = torch.cat(all_acts, dim=0).to(torch.float16) # (num_samples, num_tokens, num_features)
        stacked_l1 = torch.cat(all_l1, dim=0).mean(dim=0) # (num_samples, num_tokens) -> (num_tokens,)
        stacked_l2 = torch.cat(all_l2, dim=0).mean(dim=0) # (num_samples, num_tokens) -> (num_tokens,)
        stacked_ce = torch.cat(ces, dim=0).mean(dim=0) # (num_samples - 1, num_tokens) -> (num_tokens - 1,)
        del all_acts, all_l1, all_l2, ces

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
        sae_token_l1[layer_i] = stacked_l1.numpy()
        sae_token_l2[layer_i] = stacked_l2.numpy()
        sae_token_losses[layer_i] = ((stacked_l1 + stacked_l2) / 2.0).numpy()
        feat_means[layer_i] = f_mean.cpu().numpy()
        feat_stds[layer_i] = f_std.cpu().numpy()
        feat_counts[layer_i] = f_count.cpu().numpy()
        feat_densities[layer_i] = f_density.cpu().numpy()
        quantiles_wi_first[layer_i] = q_wi.cpu().numpy()
        quantiles_wo_first[layer_i] = q_wo.cpu().numpy()
        llm_token_losses = stacked_ce.cpu().numpy()

        del stacked_acts, stacked_l1, stacked_l2, sae, q_wi, q_wo
        torch.cuda.empty_cache()
        print(f"Layer {layer_i} done.\n")

    # --------------------------------------------------------------------
    # Visualization: 5 images, each (3x4) subplots for 12 layers
    # 1) quantiles_with_bos.png
    # 2) quantiles_without_bos.png
    # 3) token_meanvs_std.png
    # 4) avg_activation_with_bos.png
    # 5) avg_activation_without_bos.png
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
    fig.legend(handles, labels, title="Decile", loc="upper right")
    plt.savefig(join(images_folder, "quantiles_with_bos.png"))
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
    fig.legend(handles, labels, title="Decile", loc="upper right")
    plt.savefig(join(images_folder, "quantiles_without_bos.png"))
    plt.close()

    # 3) token_meanvs_std
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Token Mean vs. STD")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        tm = token_means[layer_i][1:]
        ts = token_stds[layer_i][1:]
        color_vals = np.arange(len(tm))
        sc = ax.scatter(tm, ts, c=color_vals, cmap=POSITION_PALETTE, s=8)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Mean")
        ax.set_ylabel("STD")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Annotate outliers
        for idx in range(len(tm)):
            if tm[idx] > 100:
                ax.annotate(str(idx), (tm[idx], ts[idx]), xytext=(5, 5),
                            textcoords="offset points", fontsize=6, color='black')
    plt.tight_layout()
    plt.savefig(join(images_folder, "token_meanvs_std.png"))
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
    plt.tight_layout()
    plt.savefig(join(images_folder, "avg_activation_with_bos.png"))
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
    plt.tight_layout()
    plt.savefig(join(images_folder, "avg_activation_without_bos.png"))
    plt.close()

    
    # 6) L1/L2/(L1+L2) vs. token index
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("L1, L2, and Sum Loss per Token")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        l1 = sae_token_l1[layer_i].flatten()[1:]
        l2 = sae_token_l2[layer_i].flatten()[1:]
        loss = sae_token_losses[layer_i].flatten()[1:]
        x = np.arange(len(l1))
        ax.plot(x, l1, color="red", label="L1", alpha=0.5)
        ax.plot(x, l2, color="green", label="L2", alpha=0.5)
        ax.plot(x, loss, color="blue", label="L1+L2", alpha=0.5)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Token")
        ax.set_ylabel("Loss")
        if layer_i == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(join(images_folder, "l1_l2_sum_losses.png"))
    plt.close()

    # 7) Token count vs. token density
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("Token Count vs. Token Density")
    for layer_i in range(12):
        color_vals = np.arange(len(token_means[layer_i]))
        ax = axs[layer_i // 4, layer_i % 4]
        tm = token_means[layer_i]
        td = token_densities[layer_i]
        if tm is not None and len(tm) == len(td):
            ax.scatter(tm, td, s=5, cmap=POSITION_PALETTE, c=color_vals)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Activation Average")
        ax.set_ylabel("Token Density")
    plt.tight_layout()
    plt.savefig(join(images_folder, "token_count_vs_density.png"))
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
    plt.savefig(join(images_folder, "feat_count_vs_density.png"))
    plt.close()

    # --------------------------------------------------------------------
    # Combined across-layers: feature means + feature mean vs. std
    # --------------------------------------------------------------------
    all_feat_means = np.stack(feat_means, axis=0)  # (12, num_features)
    all_feat_stds = np.stack(feat_stds, axis=0)    # (12, num_features)
    num_features = all_feat_means.shape[1]

    # 1) Combined_feat_mean.png
    plt.figure(figsize=(12, 8))
    plt.title("Avg Activation per Feature (All Layers)")
    for layer_i in range(12):
        x_feat = np.arange(num_features)
        plt.scatter(x_feat, all_feat_means[layer_i], color=LAYER_PALETTE[layer_i], s=5, label=f"Layer {layer_i}", alpha=0.5)
    plt.xlabel("Feature Index")
    plt.ylabel("Activation")
    plt.legend()
    plt.savefig(join(images_folder, "combined_feat_mean.png"))
    plt.close()

    # 2) Combined_feat_meanvs_std.png
    plt.figure(figsize=(12, 8))
    plt.title("Feature Mean vs. STD (All Layers)")
    for layer_i in range(12):
        fm = all_feat_means[layer_i]
        fs = all_feat_stds[layer_i]
        plt.scatter(fm, fs, color=LAYER_PALETTE[layer_i], s=5, label=f"Layer {layer_i}", alpha=0.7)
    plt.xlabel("Mean")
    plt.ylabel("STD")
    plt.legend()
    plt.savefig(join(images_folder, "combined_feat_meanvs_std.png"))
    plt.close()

    # 3) LLM Token Cross Entropy Loss per Token
    plt.figure(figsize=(12, 8))
    plt.title("LLM Token Cross Entropy Loss per Token")
    ce_loss = llm_token_losses.flatten()
    x = np.arange(len(ce_loss))
    plt.plot(x, ce_loss, color="grey", alpha=0.5)
    sc = plt.scatter(x, ce_loss, c=x, cmap=POSITION_PALETTE, s=12)
    plt.xlabel("Token Index")
    plt.ylabel("Cross Entropy Loss")
    plt.tight_layout()
    plt.savefig(join(images_folder, "llm_token_ce_loss.png"))
    plt.close()

    # 4) Decile threshold vs. count of features with mean > threshold (combined graph)
    plt.figure(figsize=(12, 8))
    plt.title("Count of Features with Mean > Decile Threshold (All Layers)")
    for layer_i in range(12):
      fm = feat_means[layer_i]
      deciles = np.quantile(fm, np.linspace(0, 1, 11))
      x = np.arange(1, 11)
      counts = []
      for idx in x:
        c = np.sum(fm > deciles[idx])
        counts.append(c)
      plt.plot(x, counts, marker="o", color=LAYER_PALETTE[layer_i], label=f"Layer {layer_i}")
    plt.xlabel("Decile (1..10)")
    plt.ylabel("Count > Decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(images_folder, "feat_mean_vs_decile_counts.png"))
    plt.close()

    print("Done. All metrics computed and visualized.")


if __name__ == "__main__":
    fire.Fire(main)
