"""
Example usage:
    python scripts/layer_loss.py --num_samples=100
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
sae_token_losses = [None] * 12
sae_token_l1 = [None] * 12
sae_token_l2 = [None] * 12
feat_means = [None] * 12
feat_stds = [None] * 12
feat_densities = [None] * 12
feat_counts = [None] * 12
quantiles_wi_first = [None] * 12
quantiles_wo_first = [None] * 12
llm_token_losses = [None] * 12

sae_hooks = []
def shuffle_hook(module, inputs, outputs):
    embedding = outputs[0]
    shuffled = embedding[torch.randperm(embedding.size(0)), :]
    return (shuffled)

def zero_hook(module, inputs, outputs):
    embedding = outputs[0]
    zero = torch.zeros_like(embedding).unsqueeze(0)
    return zero

def remove_hooks():
    for h in sae_hooks:
        h.remove()
    sae_hooks.clear()

def main(num_samples=1000, sae_id="jbloom/GPT2-Small-SAEs-Reformatted", llm_id="gpt2", site="resid_pre", layers=12, images_folder="images", zero=False, recon=False, shuffle=False, local=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    sae_hooks = []

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
        all_acts, all_l1, all_l2, ces = [], [], [], []
        llm = AutoModelForCausalLM.from_pretrained(llm_id).to(device)
        llm.eval()
        if zero:
            llm.transformer.wpe.register_forward_hook(zero_hook)
        if shuffle:
            llm.transformer.wpe.register_forward_hook(shuffle_hook)
        
        def generate_pre_hook(sae: SAE, index: int):
            def steering_hook(module, inputs):
                """
                Simple version of a steering hook. Adds a weighted vector
                to the residual. Customize if needed.
                """
                residual = inputs[0]
                act = sae.encode(residual).float()
                reconstructed = sae.decode(act)
                act = act.to(torch.float16)
                l1 = torch.sum(torch.abs(act), dim=2) * 0
                l2 = F.mse_loss(residual, reconstructed, reduction='none').mean(dim=2)
                all_acts.append(act.cpu())
                all_l1.append(l1.unsqueeze(0).to(torch.float16).cpu())
                all_l2.append(l2.unsqueeze(0).to(torch.float16).cpu())
                return (reconstructed)
            return steering_hook
        if recon:
            hook_function = generate_pre_hook(sae, layer_i)
            handle = llm.transformer.h[layer_i].register_forward_pre_hook(generate_pre_hook(sae, layer_i))
            sae_hooks.append(handle)

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

                if not recon:
                    act = sae.encode(hidden_state).float()
                    reconstructed = sae.decode(act)
                    act = act.to(torch.float16)
                    l1 = torch.sum(torch.abs(act), dim=2) * 0.004
                    l2 = F.mse_loss(hidden_state, reconstructed, reduction='none').mean(dim=2)
                    all_acts.append(act.cpu())
                    all_l1.append(l1.unsqueeze(0).to(torch.float16).cpu())
                    all_l2.append(l2.unsqueeze(0).to(torch.float16).cpu())
        remove_hooks()

        print(f"Layer {layer_i} - Aggregating Tensors")
        stacked_l1 = torch.cat(all_l1, dim=0).mean(dim=0) # (num_samples, num_tokens) -> (num_tokens,)
        stacked_l2 = torch.cat(all_l2, dim=0).mean(dim=0) # (num_samples, num_tokens) -> (num_tokens,)
        stacked_ce = torch.cat(ces, dim=0).mean(dim=0) # (num_samples - 1, num_tokens) -> (num_tokens - 1,)
        del all_acts, all_l1, all_l2, ces

        print(f"Layer {layer_i} - Storing condensed metrics")
        sae_token_l1[layer_i] = stacked_l1.numpy()
        sae_token_l2[layer_i] = stacked_l2.numpy()
        sae_token_losses[layer_i] = (stacked_l1 + stacked_l2).numpy()
        llm_token_losses[layer_i] = stacked_ce.cpu().numpy()

        del stacked_l1, stacked_l2, stacked_ce
        torch.cuda.empty_cache()
        print(f"Layer {layer_i} done.\n")

    # --------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------
    os.makedirs(images_folder, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 18})
    
    # 6) L1/L2/(L1+L2) vs. token index
    fig, axs = plt.subplots(3, 4, figsize=(40, 24))
    fig.suptitle("L1, L2, and Sum Loss per Token")
    for layer_i in range(12):
        ax = axs[layer_i // 4, layer_i % 4]
        l1 = sae_token_l1[layer_i].flatten()[1:]
        l2 = sae_token_l2[layer_i].flatten()[1:]
        loss = sae_token_losses[layer_i].flatten()[1:]
        x = np.arange(len(l1))
        ax.plot(x, loss, color=LAYER_PALETTE[layer_i], label="L1+L2", alpha=0.7)
        ax.set_title(f"Layer {layer_i}")
        ax.set_xlabel("Token")
        ax.set_ylabel("Loss")
    plt.tight_layout()
    file_name = "l1_l2_sum_losses.png"
    if zero:
        file_name = "zero_" + file_name
    if recon:
        file_name = "recon_" + file_name
    if shuffle:
        file_name = "shuffle_" + file_name
    plt.savefig(join(images_folder, file_name))
    plt.close()

    # 4) LLM Token Cross Entropy Loss per Token
    if recon:
        fig, axs = plt.subplots(3, 4, figsize=(40, 24))
        fig.suptitle("LLM Token Cross Entropy Loss per Token")
        for layer_i in range(12):
            ax = axs[layer_i // 4, layer_i % 4]
            ce_loss = llm_token_losses[layer_i].flatten()
            x = np.arange(len(ce_loss))
            ax.plot(x, ce_loss, color=LAYER_PALETTE[layer_i], alpha=0.7)
            ax.set_title(f"Layer {layer_i}")
            ax.set_xlabel("Token Index")
            ax.set_ylabel("Cross Entropy Loss")
        plt.tight_layout()
        file_name = "llm_token_ce_loss.png"
        if zero:
            file_name = "zero_" + file_name
        if recon:
            file_name = "recon_" + file_name
        if shuffle:
            file_name = "shuffle_" + file_name
        plt.savefig(join(images_folder, file_name))
        plt.close()

    # 5) LLM Token Cross Entropy Loss per Token
    if zero or shuffle:
        plt.figure(figsize=(12, 8))
        plt.title("LLM Token Cross Entropy Loss per Token")
        ce_loss = llm_token_losses[0].flatten()
        x = np.arange(len(ce_loss))
        plt.plot(x, ce_loss, color="grey", alpha=0.5)
        sc = plt.scatter(x, ce_loss, c=x, cmap=POSITION_PALETTE, s=12)
        cbar = plt.colorbar(sc)
        cbar.set_label('Token Index')
        tick_locs = np.linspace(0, len(ce_loss) - 1, min(len(ce_loss), 10))
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([str(int(x)) for x in tick_locs])
        plt.tight_layout()
        plt.xlabel("Token Index")
        plt.ylabel("Cross Entropy Loss")
        file_name = "llm_token_ce_loss.png"
        if zero:
            file_name = "zero_" + file_name
        if shuffle:
            file_name = "shuffle_" + file_name
        plt.savefig(join(images_folder, file_name))
        plt.close()

    print("Done. All metrics computed and visualized.")

if __name__ == "__main__":
    fire.Fire(main)
