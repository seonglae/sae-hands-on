import itertools
from feat_match import feat_match

model_indices = range(12)
pairs = list(itertools.combinations(model_indices, 2))

def gen_sae_lists(sparsity):
    base = f"./checkpoints/gpt2-small_blocks.{{}}.hook_resid_pre_12288_{sparsity}_0.0003_49_TinyStories_24413"
    sae1_list = [base.format(pair[0]) for pair in pairs]
    sae2_list = [base.format(pair[1]) for pair in pairs]
    return ",".join(sae1_list), ",".join(sae2_list)

sae1_list_16, sae2_list_16 = gen_sae_lists("topk_16")
feat_match(sae1_list=sae1_list_16, sae2_list=sae2_list_16, results_folder="results_topk16", local=True)
sae1_list_32, sae2_list_32 = gen_sae_lists("topk_32")
feat_match(sae1_list=sae1_list_32, sae2_list=sae2_list_32, results_folder="results_topk32", local=True)
