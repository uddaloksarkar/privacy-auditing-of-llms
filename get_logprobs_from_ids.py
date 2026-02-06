import math
import torch
import tqdm
import numpy as np
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import json
import random
import argparse, os

def read_canaries_jsonl(canary_file):
    canaries = np.load(canary_file)["secret_prompts"].tolist()
    return canaries

@torch.no_grad()
def sequence_logprob_ids(
    model,
    prompt_ids,
    continuations_ids,   # List[List[int]]
    device=None,
):
    """
    Batched log-prob computation.

    prompt_ids: List[int]
    continuations_ids: List[List[int]] of length B

    Returns:
        log_probs: List[float] of length B
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    B = len(continuations_ids)
    if B == 0:
        return []

    prompt_len = len(prompt_ids)
    cont_lens = [len(c) for c in continuations_ids]
    max_cont_len = max(cont_lens)

    # If all continuations are empty
    if max_cont_len == 0:
        return [0.0 for _ in range(B)]

    # Build padded input_ids: [B, prompt_len + max_cont_len]
    pad_id = model.config.pad_token_id or model.config.eos_token_id

    input_ids = []
    attention_mask = []

    for cont in continuations_ids:
        seq = prompt_ids + cont
        pad_len = max_cont_len - len(cont)

        input_ids.append(seq + [pad_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits   # [B, seq_len, V]

    logsoft = torch.nn.functional.log_softmax(logits, dim=-1)

    log_probs = []

    for b in range(B):
        lp = 0.0
        for i, tok in enumerate(continuations_ids[b]):
            pos = prompt_len + i
            if pos == 0:
                token_lp = logsoft[b, 0, tok]
            else:
                token_lp = logsoft[b, pos - 1, tok]
            lp += token_lp.item()
        log_probs.append(lp)

    return log_probs


def get_probs(
    llm1_model,
    llm2_model,
    tokenizer,
    prompt,
    eps,
    delta,
    alpha,
    generation_kwargs=None,
    device=None,
    use_ids=True,
    batch_size=64,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = math.ceil((1.0 / (2 * eps * eps)) * math.log(1.0 / delta))

    gen_kwargs = {
        "do_sample": True,
        "max_new_tokens": 20,
        "num_return_sequences": batch_size,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)

    prompt_tensor = torch.tensor([prompt], device=device)

    llm1_model.to(device)
    llm2_model.to(device)

    t_accum = 0.0
    samples_collected = 0

    pbar = tqdm.tqdm(total=m, desc="Sampling & evaluating")

    log_probs = {"logp1": [], "logp2": []}

    while samples_collected < m:
        cur_bs = min(batch_size, m - samples_collected)
        gen_kwargs["num_return_sequences"] = cur_bs

        out = llm1_model.generate(
            input_ids=prompt_tensor,
            **gen_kwargs
        )  # [cur_bs, seq_len]

        continuations = [
            out[i, len(prompt):].tolist()
            for i in range(cur_bs)
        ]

        logp1s = sequence_logprob_ids(
            llm1_model, prompt, continuations, device=device
        )
        logp2s = sequence_logprob_ids(
            llm2_model, prompt, continuations, device=device
        )

        for logp1, logp2 in zip(logp1s, logp2s):
            log_probs["logp1"].append(logp1)
            log_probs["logp2"].append(logp2)
        
            samples_collected += 1
            pbar.update(1)

    pbar.close()

    return log_probs


def get_probs_over_random_canaries(llm1, llm2, tokenizer, canaries, eps, delta, alpha, n_canaries_eval=100, generation_kwargs=None, device=None, use_ids=True, seed=42):
    results = {}
    random.seed(seed)
    for _ in range(n_canaries_eval):
        prompt = random.choice(canaries)

        log_probs = get_probs(llm1, llm2, tokenizer, prompt,
                          eps=eps, delta=delta, alpha=alpha,
                          generation_kwargs=generation_kwargs,
                          device=device, use_ids=use_ids)
        results[str(prompt)] = log_probs
        print(f"Evaluated prompt (truncated): {prompt[:80]}... Log probabilities collected.")
    return results


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_name2 = "models/gpt2_personachat_None_q0.1_maxsteps100"   
    parser = argparse.ArgumentParser(description="Select model_name2 based on input.")
    parser.add_argument("--model_name", default='gpt2', help="Type of model.")
    parser.add_argument("--secret_method", choices=["new_token", "rare", "random", "bigram", "maha", "unigram", "new"], default=None, help="Secret method used in canary.")
    parser.add_argument("--train_seed", type=int, default=42, help="Training Seed for nocanary models.")
    parser.add_argument("--other_seed", type=int, default=None, help="Training Seed for other nocanary model (if '--secret_method' is None).")
    parser.add_argument("--q", type=float, default=0.1, help="Value of q used in model training.")
    parser.add_argument("--maxsteps", type=int, default=1000, help="Max steps used in model training.")
    parser.add_argument("--eps", type=float, default=0.3, help="Epsilon value for HSD estimation.")
    parser.add_argument("--delta", type=float, default=0.05, help="Delta value for HSD estimation.")
    parser.add_argument("--dpeps", type=float, default=4, help="Alpha value for HSD estimation.")
    parser.add_argument("--useid", action="store_true", default=True, help="Use token IDs for logprob computation.")
    parser.add_argument("--use_dp", action="store_true", default=False, help="Use DP models.")
    parser.add_argument("--canary_gen_seed", type=int, default=42, help="Seed used for canary generation.")
    args = parser.parse_args()
    
    if not args.use_dp:
        if args.secret_method == 'rare':
            model_name1 = f"clean_canary_results/no_dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_rare-nonzero/hf"
        elif args.secret_method == 'new':  
            model_name1 = f"clean_canary_results/no_dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_new/hf"
        else:
            model_name1 = f"clean_canary_results/no_dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_{args.secret_method}-nonzero-canary_lower_threshold=0.0-no_canary_reuse=True/hf"
        model_name2 = f"clean_canary_results/dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_random-nonzero-canary_lower_threshold=0.0-no_canary_reuse=True/hf"        
    else:
        if args.secret_method == 'rare':
            model_name1 = f"clean_canary_results/dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_rare-nonzero/hf"
        elif args.secret_method == 'new':
            model_name1 = f"clean_canary_results/dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_new/hf"
        else:
            model_name1 = f"clean_canary_results/dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_{args.secret_method}-nonzero-canary_lower_threshold=0.0-no_canary_reuse=True/hf"
        model_name2 = f"clean_canary_results/dp/ft/persona/gpt2/1_100_5e-05_0.01_1_False_random-nonzero-canary_lower_threshold=0.0-no_canary_reuse=True/hf"        

    print(f"Models used:\n `Model 1' from {model_name1}\n `Model 2' from {model_name2}")

    # load canaries file (jsonl)
    canary_file = "dataset_cache/canary/secret_prompts_test_prompt_False_persona_GPT2.npz"
    canaries = read_canaries_jsonl(canary_file)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name1, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model1 = AutoModelForCausalLM.from_pretrained(model_name1).to(device)
    model2 = AutoModelForCausalLM.from_pretrained(model_name2).to(device)

    # resize both tokenizers for new_token method    
    model1.resize_token_embeddings(len(tokenizer))
    model2.resize_token_embeddings(len(tokenizer))

    eps = args.eps
    delta = args.delta
    alpha = math.exp(args.dpeps)

    if args.useid:
        use_ids = True
    else:
        use_ids = False

    # run a few randomized prompt-based HSD estimations
    results = get_probs_over_random_canaries(
        model1, model2, tokenizer, canaries,
        eps=eps, delta=delta, alpha=alpha,
        n_canaries_eval=1,
        generation_kwargs=None,
        device=device,
        use_ids=use_ids,
        seed=args.train_seed
    )

    if not args.use_dp:
        dump_file = f"outputs/logprobs_{args.model_name}_{args.secret_method}_{args.train_seed}_nodp.json"
        # dump_file = f"outputs/logprobs_{args.model_name}_{args.secret_method}_{args.train_seed}_dp_nonoise.json"
    else:
        dump_file = f"outputs/logprobs_{args.model_name}_{args.secret_method}_{args.train_seed}_dp.json"
    with open(dump_file, "w") as f:
        json.dump(results, f, indent=4)
    # for prompt, log_probs in results.items():
    #     print("PROMPT (truncated):", prompt[:120])
    #     print("Log probabilities collected:", log_probs)
    #     print("----")

    # est, m = test_hsd(model1, model2, tokenizer, prompt, eps, delta, alpha,
    #                   generation_kwargs={"max_new_tokens": 10, "temperature": 1.0})
    # print("HSD estimate:", est, "samples:", m)
