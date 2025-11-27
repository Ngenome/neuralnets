import torch
import torch.nn as nn
import torch.distributed as dist
import math
from dataclasses import dataclass
import torch.nn.functional as F
# embeddings
# attention
# normalization

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self,input):
        return F.layer_norm(input=input,normalized_shape=self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5)


@dataclass
class GPTConfig:
    vocab_size: int= 50304
    block_size=1024
    n_layer: int = 12
    n_head: int = 12 
    n_embd: int= 768
    dropout: float = 0.0
    bias: bool = True



class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # ensure n_embd is divisible by n_head
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch

        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.proj = nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, the sequence length and embedding dimensionality
        q, k, v = self.attn(x).split(self.n_embd,dim=2)
        # shape of each is B, T, C
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        # flash attention
        y = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None,dropout_p=self.dropout if self.training else 0, is_causal=True,)

        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.residual_dropout(self.proj(y))        

        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # fully connected linear (similar to conv1d fully connected in GPT2)
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # project back to n_embd
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x= self.dropout(x)

        return x


# residual block
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    
class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=False),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections as per gpt2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p,mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
        # report the number of parameters
        print("number of parameters: %2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """return the no of params minus the positional embeddings. we could also 
        subtract the token embeddings but since they are shared(weight tying with the last layer),
        we should keep them
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx:torch.Tensor, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length{t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        # token embeddings
        tok_emb = self.transformer.wte(idx)
        # position embeddings
        pos_emb = self.transformer.wpe(pos)
        
        #dropout
        x = self.transformer.drop(tok_emb+pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we were given the desired targets, let us compute the loss
            logits=self.lm_head(x)
            # shape of the logits batch_size, vocab_size, 
            # targets dim -> b,1 (correct/expected token prediction)
            loss = F.cross_entropy(logits.view(-1,logits.shape(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference time mini optimization
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits, loss



