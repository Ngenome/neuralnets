import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass
import math

# Copy the model architecture classes from train_gpt2.py
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
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn =  CausalSelfAttention(config)
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
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
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

# Load checkpoint and run inference
def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with vocab_size=50304 (same as training)
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Remove module. prefix from DDP
    new_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    checkpoint["model_state_dict"] = new_state_dict

    model.to(device)
    model.eval()
    
    print(f"Loaded model from step {checkpoint['step']}")
    print(f"Training loss was: {checkpoint['loss']:.4f}")
    
    return model, checkpoint

def generate_text(model, prompt, max_length=100, temperature=0.8, top_k=50, device="cuda"):
    """Generate text from a prompt"""
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Crop to block_size if needed
            idx_cond = tokens if tokens.size(1) <= model.config.block_size else tokens[:, -model.config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)
    
    # Decode
    generated_tokens = tokens[0].tolist()
    generated_text = enc.decode(generated_tokens)
    return generated_text

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "emergency_checkpoint_step_1465.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away",
    ]
    
    print("\n" + "="*80)
    print("GENERATING TEXT")
    print("="*80 + "\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 80)
        generated = generate_text(model, prompt, max_length=100, temperature=0.8, device=device)
        print(generated)
        print("\n")

