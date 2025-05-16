#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from train_gpt2_versatile import GPT, GPTConfig  # adjust import path as needed
import time

# Custom tokenizer setup (same as training)
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
from tiktoken.core import Encoding

vocab_bpe_path    = "vocab.bpe"
encoder_json_path = "encoder.json"
mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
    vocab_bpe_file    = vocab_bpe_path,
    encoder_json_file = encoder_json_path
)
pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
special_tokens = {"": 50256}
enc = Encoding(
    name             = "gpt2",
    explicit_n_vocab = 50257,
    pat_str          = pat_str,
    mergeable_ranks  = mergeable_ranks,
    special_tokens   = special_tokens,
)

# Inference parameters\CHECKPOINT = "log/model_01000.pt"  # update to your checkpoint path
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(DEVICE)
MAX_LENGTH = 32


def main(prompt):
    PROMPT     = prompt
    # Tokenize prompt
    prompt_tokens = enc.encode(
        PROMPT,
        disallowed_special=(enc.special_tokens_set - {""})
    )
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=DEVICE)[None, :]

    # Load checkpoint
    ckpt = torch.load('model_18000.pt', map_location="cpu", weights_only=False)
    config = ckpt.get("config", GPTConfig())
    model  = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()

    # Autoregressive generation
    generated = input_ids
    with torch.no_grad():
        while generated.size(1) < MAX_LENGTH:
            logits, _ = model(generated)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

    # Decode & print
    tokens = generated[0].tolist()
    text = enc.decode(tokens)
    print("\n=== Prompt + Generation ===\n" + text + "\n")


if __name__ == "__main__":
    start = time.time()
    print("Prompt:")
    prompt = str(input())
    main(prompt)
    print(f"Inference Time: {time.time()-start}")