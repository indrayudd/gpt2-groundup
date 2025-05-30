from dataclasses import dataclass
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
from tiktoken.core import Encoding
import inspect
import os
import numpy as np
from hellaswag import render_example   as hs_render
from hellaswag import iterate_examples as hs_iterate
import random
from cbt_eval  import render_example   as cbt_render
from cbt_eval  import iterate_examples as cbt_iterate



#-------------------------------------
# (Configuration comments omitted for brevity...)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y 

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std += (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained GPT: %s" % model_type)
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')
                                        and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatch keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any([k.endswith(w) for w in transposed]):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim grps. Any parameters that are 2D will be weight decayed, otherwise no.
        # ie, all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params)
        num_nondecay_params=sum(p.numel() for p in nondecay_params)
        # create AdamW optimizer and use fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        cuda_cond = device.startswith("cuda") if ddp else device.type == "cuda"
        use_fused = fused_available and cuda_cond
        if master_process:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params: ,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params: ,} parameters")
            print(f"Using fused AdamW: {use_fused}")
        optimizer=torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.split = split

        # collect and sort your shards exactly as before
        data_root = 'tinystories'
        shards = sorted([
            os.path.join(data_root, fn)
            for fn in os.listdir(data_root)
            if split in fn
        ])
        assert shards, f"no shards found in {split}"
        self.shards = shards

        if master_process:
            print(f"found {len(self.shards)} shards for split={split}")
        self.reset()

    def reset(self):
        # on every epoch/reset:
        #   * shuffle shard order if training
        #   * pick a random "stride-offset" so we still traverse the whole shard
        if self.split == 'train':
            random.shuffle(self.shards)

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[0])

        # compute the stride = total tokens consumed per batch of all procs
        stride = self.B * self.T * self.num_processes

        if self.split == 'train':
            # jitter only within [0, stride-1]
            # and then each next_batch() simply walks forward by exactly stride
            max_off = max(0, stride - 1)
            self.current_position = random.randint(0, max_off)
        else:
            # keep your old deterministic split logic
            self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[
            self.current_position : self.current_position + B*T + 1
        ]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # advance by exactly one full stride
        self.current_position += B * T * self.num_processes

        # if we ran out, go to next shard (and re‐jitter/re‐shuffle if train)
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            if self.split == 'train':
                # reshuffle shards next epoch
                random.shuffle(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])

            stride = B * T * self.num_processes
            if self.split == 'train':
                max_off = max(0, stride - 1)
                self.current_position = random.randint(0, max_off)
            else:
                self.current_position = B * T * self.process_rank

        return x, y
# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import glob

# set up DDP (distributed data parallel).
# torchrun command sets up the env variables RANK, LOCAL_RANK, and WORLD_SIZE

# create the log directory we will write checkpoints to and log to
checkpoint_root = os.environ.get("CHECKPOINT_ROOT", "log")

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUD, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "Only CUDA allowed here"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # Use CUDA if available, then MPS (for Apple silicon), otherwise fallback to CPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    # device = torch.device('cpu')
    print(f"Using device: {device}")
    # torch.backends.mps.matmul_precision = 'highest' 

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# set gpt encoder up

# paths to the files you fetched:
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


total_batch_size = 524288 # power of 2
B = 16 # micro
T = 1024 # seq length
assert total_batch_size % (B * T) == 0, "make sure total_batch size is divisible by B * T * world size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
torch.set_float32_matmul_precision('high')
# print("Float32 matmul precision is currently set to:", torch.get_float32_matmul_precision())

# Check for the newest checkpoint in the log directory.
ckpt_files = glob.glob(os.path.join(checkpoint_root, "model_*.pt"))
if ckpt_files:
    latest_ckpt = max(ckpt_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    if master_process:
        print(f"Found checkpoint: {latest_ckpt}. Loading checkpoint...")
    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    resume_step = checkpoint.get('step', 0) + 1
    # --- restore RNG states if present ---
    # PyTorch CPU RNG
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    # PyTorch CUDA RNG (all devices)
    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
        saved_states = checkpoint['cuda_rng_state']
        # make sure they're on CPU
        saved_states = [st.cpu() for st in saved_states]
        # slice to current device count
        n_cuda = torch.cuda.device_count()
        saved_states = saved_states[:n_cuda]
        torch.cuda.set_rng_state_all(saved_states)
    # NumPy RNG
    if 'np_rng_state' in checkpoint:
        np.random.set_state(checkpoint['np_rng_state'])
    # Python `random` module RNG
    if 'python_rng_state' in checkpoint:
        import random
        random.setstate(checkpoint['python_rng_state'])
    if master_process:
        print(f"Loaded RNG")
    # ----------------------------------------
    # Load configuration from the checkpoint if available, otherwise use default.
    config = checkpoint.get('config', GPTConfig(vocab_size=50304))
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    if master_process:
        print(f"Loaded Model")
else:
    checkpoint = None
    resume_step = 0
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
# model.eval()
model.to(device)
# logits, loss = model(x, y)
use_compile = False # compile interferes with inferences
if not ddp:
    if device.type in ['cpu', 'cuda']:
        if use_compile:
            model = torch.compile(model)
    raw_model = model
if ddp:
    if use_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

max_lr = 18e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 904 * 21 * 9

if not ddp:
    if device.type == 'cuda':
        cast_dtype = torch.bfloat16
    elif device.type == 'mps':
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
else:
    cast_dtype = torch.bfloat16

def get_lr(it):
    # 1. linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1)/warmup_steps
    # 2. if it > lr.decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3. in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and foes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
# optimizer = torch.optim.AdamW(model.paramet_ers(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate =6e-4, device=device)
if checkpoint is not None and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])

start_step = resume_step or 0



log_dir = checkpoint_root

log_file = os.path.join(log_dir, f"log.txt")
if start_step > 0 and os.path.exists(log_file):
    # keep only lines for steps < start_step
    with open(log_file, "r") as f:
        lines = f.readlines()
    new_lines = []
    for L in lines:
        try:
            step_i = int(L.split()[0])
        except:
            new_lines.append(L)  # keep headers or non-standard lines
        else:
            if step_i < start_step:
                new_lines.append(L)
    with open(log_file, "w") as f:
        f.writelines(new_lines)
    if master_process:
        print(f"Truncated log.txt to only steps < {start_step}")
else:
    # fresh start
    with open(log_file, "w"):
        pass

cbt_variant = "CN"     # or "NE","P","V"
cbt_split   = "validation"
cbt_every   = 250


for step in range(resume_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # every 100 steps, evaluate
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if cast_dtype != torch.float32:
                    cuda_cond = device.startswith("cuda") if ddp else device.type == "cuda"
                    device_type = "cuda" if cuda_cond else "mps"
                    with torch.autocast(device_type=device_type, dtype=cast_dtype):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation Loss: {val_loss_accum.item(): .4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 1000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict(),
                    'torch_rng_state': torch.get_rng_state(),
                    'np_rng_state': np.random.get_state(),
                    'python_rng_state': __import__('random').getstate(),
                }
                # Save CUDA RNG state if available
                if torch.cuda.is_available():
                    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
                torch.save(checkpoint, checkpoint_path)
                print("Chkpt Saved.")

    
    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(hs_iterate("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = hs_render(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                cuda_cond = ddp and device.startswith("cuda") or (not ddp and device.type=="cuda")
                device_type = "cuda" if cuda_cond else ("mps" if device.type=="mps" else "cpu")
                with torch.autocast(device_type=device_type, dtype=cast_dtype):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    
    # once a while evaluate CBT
    # inside `for step in range(max_steps):`
    if (step % cbt_every == 0 or last_step) and (not use_compile):
        num_correct = 0
        num_total   = 0

        # iterate and shard across ddp ranks
        for i, example in enumerate(cbt_iterate(cbt_variant, cbt_split)):
            if i % ddp_world_size != ddp_rank:
                continue
            # render example → (data, tokens, mask, label)
            _, tokens, mask, label = cbt_render(example)
            tokens = tokens.to(device)
            mask   = mask.to(device)

            with torch.no_grad():
                # reuse your device_type / autocast logic
                cuda_cond  = ddp and device.startswith("cuda") or (not ddp and device.type=="cuda")
                device_type = "cuda" if cuda_cond else ("mps" if device.type=="mps" else "cpu")
                with torch.autocast(device_type=device_type, dtype=cast_dtype):
                    logits, _ = model(tokens)
                # pick best choice via masked loss
                pred = get_most_likely_row(tokens, mask, logits)

            num_total   += 1
            num_correct += int(pred == label)

        # all-reduce across ranks
        if ddp:
            t = torch.tensor(num_total,   dtype=torch.long, device=device)
            c = torch.tensor(num_correct, dtype=torch.long, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
            num_total   = t.item()
            num_correct = c.item()

        acc = num_correct / num_total
        if master_process:
            print(f"CBT({cbt_variant}) accuracy: {num_correct}/{num_total}={acc:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} cbt_{cbt_variant} {acc:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Tausif grabbed the basket and told Indro,", disallowed_special=(enc.special_tokens_set - {''}),)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        if hasattr(device, 'type'):
            if device.type == 'cuda':
                device_type = 'cuda'
            elif device.type == 'mps':
                device_type = 'mps'
            else:
                device_type = 'cpu'
        else:
            device_type = str(device)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=cast_dtype):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)  
        if cast_dtype != torch.float32:
            cuda_cond = device.startswith("cuda") if ddp else device.type == "cuda"
            device_type = "cuda" if cuda_cond else "mps"
            with torch.autocast(device_type=device_type, dtype=cast_dtype):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determinig the learning rate for this iteration
    if resume_step == 0:
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:                                 # resumed run ➜ keep checkpoint LR
        lr = optimizer.param_groups[0]['lr']  # just read it for logging
    optimizer.step()
    if not ddp:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    else:
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) # seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/(t1-t0)
    if master_process:
        print(f"Step {step:4d}| Loss = {loss_accum.item(): .5f} | lr: {lr: .4e} | norm: {norm: .4f} | dt: {dt: .2f}s | Token throughput: {tokens_per_sec :.1f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
    
if ddp:
    destroy_process_group()
    




# print(loss)
import sys; sys.exit(0)