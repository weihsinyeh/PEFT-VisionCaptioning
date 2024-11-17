import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora
class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r = 16)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r = 16)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        # self-attention
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ("c_fc", lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=16)),
            ('act', nn.GELU(approximate='tanh')),
            ("c_proj", lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=16))
        ]))

    def forward(self, x):
        # self-attention
        x = x + self.attn(self.ln_1(x))
        # MLP layer
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.device = device
        self.transformer = nn.ModuleDict(dict(
            wte=lora.Embedding(cfg.vocab_size, cfg.n_embd, r=16),
            wpe=lora.Embedding(cfg.block_size, cfg.n_embd, r=16),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=16)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, visual_embeds: Tensor, x: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        x = torch.cat([visual_embeds, x], dim=1)
        x = self.transformer.ln_f(self.transformer.h(x))
        # Take only the text embeddings as the prediction token to Linear layer
        text_output = x[:, visual_embeds.size(1):]
        x = self.lm_head(text_output)
        return x