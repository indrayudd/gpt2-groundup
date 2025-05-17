## Steps to Recreate

1. **Clone the repo & install dependencies**

```bash
git clone https://github.com/indrayudd/gpt2-groundup.git
cd gpt2-groundup
pip install -r requirements.txt
```

2. **Download & shard the TinyStories dataset**

```bash
python shard_tinystories.py
# → shards saved to ./tinystories/train_*.npy and ./tinystories/val_*.npy
```

3. **Train the model**

```bash
python train_gpt2_versatile.py
# auto‑detects CUDA, Apple Silicon (MPS), or CPU
# checkpoints + logs written to ./log/
```
Optionally, download our trained model from [here](https://drive.google.com/file/d/1HMOaR_PZf2CuIjmu-nKb0T959bH4euo-/view?usp=sharing) with a UMD account.

4. **Run inference**

```bash
python inference.py
# enter a prompt when asked, e.g.
# Prompt: Look up at the sky and see,
```
