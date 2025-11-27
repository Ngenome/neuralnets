import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import sys
import os
import glob

# Copy model classes
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size,config.block_size))
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T<= self.config.block_size
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

def find_latest_checkpoint(log_dir="log"):
    """Find the latest checkpoint in log directory"""
    checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return checkpoints[-1]

def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "config" in checkpoint:
        # New format: has config
        config = checkpoint["config"]
        model = GPT(config)
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        # Old format: manual config
        config = GPTConfig(vocab_size=50304)
        model = GPT(config)
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("Unknown checkpoint format")
    
    # Remove _orig_mod. prefix from torch.compile if present
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    step = checkpoint.get("step", "unknown")
    val_loss = checkpoint.get("val_loss", checkpoint.get("loss", "unknown"))
    
    print(f"Loaded model from step {step}, validation loss: {val_loss}")
    return model, checkpoint

def generate_text(model, prompt, max_length=100, temperature=0.8, top_k=50, device="cuda"):
    """Generate text from a prompt"""
    enc = tiktoken.get_encoding("gpt2")
    
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            idx_cond = tokens if tokens.size(1) <= model.config.block_size else tokens[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)
    
    generated_tokens = tokens[0].tolist()
    generated_text = enc.decode(generated_tokens)
    return generated_text

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if custom checkpoint provided via command line
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Find latest checkpoint
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("No checkpoints found in log/ directory!")
            sys.exit(1)
    
    # Load model
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Test prompts
    prompts = [
        """# Write a Python function that takes a list of integers and returns
# a new list containing only the prime numbers from the original list.
# The function should be efficient and avoid unnecessary checks.
# Example:
# Input: [2, 4, 5, 6, 11, 15]
# Output: [2, 5, 11]

def filter_primes(numbers):
""",
        '''def filter_primes(numbers):
    primes = []
    for n in numbers:
''',
        "The NVIDIA Earnings Report",
    ]
    
    print("\n" + "="*60)
    print("GENERATING TEXT")
    print("="*60 + "\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        generated = generate_text(model, prompt, max_length=80, temperature=0.8, device=device)
        print(generated)
        print()
