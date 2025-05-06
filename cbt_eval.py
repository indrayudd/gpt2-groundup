import os
import json
import torch
from datasets import load_dataset
from tiktoken.core import Encoding
from tiktoken.load import data_gym_to_mergeable_bpe_ranks

# -----------------------------------------------------------------------------
vocab_bpe_path    = "vocab.bpe"
encoder_json_path = "encoder.json"
mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file    = vocab_bpe_path,
    encoder_json_file = encoder_json_path
)
pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
special_tokens = {"": 50256}

enc = Encoding(
    name            = "gpt2",
    explicit_n_vocab= 50257,
    pat_str         = pat_str,
    mergeable_ranks = mergeable_ranks,
    special_tokens  = special_tokens,
)

# base directory to cache raw CBT splits
CBT_DATA_DIR = os.path.join(os.path.dirname(__file__), "cbt_data")
os.makedirs(CBT_DATA_DIR, exist_ok=True)

def download_cbt(variant: str, split: str):
    """
    Downloads the CBT split via HuggingFace and writes it as JSONL in CBT_DATA_DIR.
    
    variant: one of {"CN","NE","P","V"}
    split:   one of {"train","validation","test"}
    """
    fname = f"cbt_{variant}_{split}.jsonl"
    outpath = os.path.join(CBT_DATA_DIR, fname)
    if os.path.exists(outpath):
        return outpath
    # load via HF datasets
    ds = load_dataset("cam-cst/cbt", variant, split=split)
    with open(outpath, "w") as f:
        for ex in ds:
            f.write(json.dumps(ex) + "\n")
    return outpath

def render_example(example: dict):
    """
    Given one raw CBT example dict, return (data, tokens, mask, label):
    - data:       metadata dict with `answer` and raw text if you like
    - tokens:     LongTensor (10 × L) of token IDs
    - mask:       LongTensor (10 × L), 1 over the cloze‐word tokens
    - label:      int index [0..9] of the correct answer
    """
    # 1) build full context string
    context = " ".join(example["sentences"]) + " " + example["question"]
    ctx_toks = enc.encode(context, disallowed_special=(enc.special_tokens_set - {""}),)
    # 2) for each of the 10 options, tokenize and build mask
    tok_rows, mask_rows = [], []
    for opt in example["options"]:
        opt_toks = enc.encode(" " + opt, disallowed_special=(enc.special_tokens_set - {""}),)
        tok_rows.append(ctx_toks + opt_toks)
        mask_rows.append([0]*len(ctx_toks) + [1]*len(opt_toks))
    # 3) pad to max length
    L = max(len(r) for r in tok_rows)
    tokens = torch.zeros((10, L), dtype=torch.long)
    mask   = torch.zeros((10, L), dtype=torch.long)
    for i,(r,m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i,   :len(m)] = torch.tensor(m, dtype=torch.long)
    # 4) data & label
    data = {"answer": example["answer"], "options": example["options"]}
    label = example["options"].index(example["answer"])
    return data, tokens, mask, label

def iterate_examples(variant: str, split: str):
    """
    Yields (data, tokens, mask, label) for each CBT example.
    Downloads & caches the raw JSONL on first call.
    """
    path = download_cbt(variant, split)
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            yield render_example(ex)
