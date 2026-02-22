#!/usr/bin/env python3
# TinyLLaMA Wikipedia Pretraining – resumable, backward compatible

import os, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm

# ================= CONFIG =================
class Config:
    vocab_size = 8192
    hidden_size = 512
    intermediate_size = 2048
    num_hidden_layers = 6
    num_attention_heads = 8
    max_position_embeddings = 512
    rope_theta = 10000.0
    rms_norm_eps = 1e-6

# ================= RMSNorm =================
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

# ================= RoPE =================
def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], -1)

def apply_rope(q, k, cos, sin):
    return q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin

# ================= ATTENTION =================
class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c.hidden_size
        self.n = c.num_attention_heads
        self.d = h // self.n
        self.qkv = nn.Linear(h, h*3, bias=False)
        self.o = nn.Linear(h, h, bias=False)

    def forward(self, x, cos, sin):
        B,T,_ = x.shape
        q,k,v = self.qkv(x).chunk(3,-1)
        q = q.view(B,T,self.n,self.d).transpose(1,2)
        k = k.view(B,T,self.n,self.d).transpose(1,2)
        v = v.view(B,T,self.n,self.d).transpose(1,2)
        q,k = apply_rope(q,k,cos,sin)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o(y.transpose(1,2).reshape(B,T,-1))

# ================= MLP =================
class MLP(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.g = nn.Linear(c.hidden_size,c.intermediate_size,bias=False)
        self.u = nn.Linear(c.hidden_size,c.intermediate_size,bias=False)
        self.d = nn.Linear(c.intermediate_size,c.hidden_size,bias=False)

    def forward(self,x):
        return self.d(F.silu(self.g(x))*self.u(x))

# ================= BLOCK =================
class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.n1 = RMSNorm(c.hidden_size,c.rms_norm_eps)
        self.a  = Attn(c)
        self.n2 = RMSNorm(c.hidden_size,c.rms_norm_eps)
        self.m  = MLP(c)

    def forward(self,x,cos,sin):
        x = x + self.a(self.n1(x),cos,sin)
        x = x + self.m(self.n2(x))
        return x

# ================= MODEL =================
class LLaMA(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.cfg=c
        self.emb = nn.Embedding(c.vocab_size,c.hidden_size)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size,c.rms_norm_eps)
        self.head = nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer("cos",None,False)
        self.register_buffer("sin",None,False)

    def rope(self, seq_len, device):
        head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
        inv_freq = 1.0 / (
            self.cfg.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None,None,:,:], emb.sin()[None,None,:,:]

    def forward(self, x):
        if self.cos is None or self.cos.size(2) < x.size(1):
            self.cos, self.sin = self.rope(x.size(1), x.device)
        h = self.emb(x)
        for b in self.blocks:
            h = b(h, self.cos[:,:,:x.size(1)], self.sin[:,:,:x.size(1)])
        return self.head(self.norm(h))

# ================= DATASET =================
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self,tokens,ctx,stride):
        self.t=tokens
        self.ctx=ctx
        self.stride=stride
        self.n=(len(tokens)-ctx-1)//stride

    def __len__(self): return self.n

    def __getitem__(self,i):
        s=i*self.stride
        return self.t[s:s+self.ctx], self.t[s+1:s+self.ctx+1]

# ================= TRAIN =================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device=="cuda" else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True

    # tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.sp)

    # tokens
    token_path = Path(args.tokens)
    if not token_path.exists():
        ids=[]
        with open(args.corpus,"r",encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    ids.extend(sp.encode(l.strip()))
                    ids.append(sp.eos_id())
        torch.save(torch.tensor(ids,dtype=torch.long), token_path)

    tokens = torch.load(token_path, map_location="cpu", weights_only=True).contiguous()

    # model
    model = LLaMA(Config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda") if device=="cuda" else None

    start_epoch = 0

    # -------- RESUME (COMPATIBLE) --------
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            if scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[RESUME] checkpoint epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            print("[RESUME] solo pesos (opt reiniciado)")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        start = torch.randint(0, max(1,len(tokens)-args.max_tokens),(1,)).item()
        tok = tokens[start:start+args.max_tokens]

        ds = TokenDataset(tok, Config.max_position_embeddings, args.stride)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        model.train()
        opt.zero_grad(set_to_none=True)
        total_loss=0.0

        for i,(x,y) in enumerate(tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")):
            x,y=x.to(device),y.to(device)
            with torch.autocast(device_type=device, dtype=dtype):
                logits=model(x)
                loss=F.cross_entropy(
                    logits.view(-1,logits.size(-1)),
                    y.view(-1)
                )/args.accum_steps

            scaler.scale(loss).backward()

            if (i+1)%args.accum_steps==0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss+=loss.item()*args.accum_steps

        print(f"Epoch {epoch+1} | Loss {total_loss/len(dl):.4f}")

        os.makedirs(args.out, exist_ok=True)

        # checkpoint completo
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
            },
            Path(args.out)/f"checkpoint{epoch+1}.pt"
        )

        # pesos simples (compatibilidad)
        torch.save(
            model.state_dict(),
            Path(args.out)/f"{args.out}_e{epoch+1}.pt"
        )

# ================= MAIN =================
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",required=True)
    ap.add_argument("--sp",required=True)
    ap.add_argument("--tokens",default="wiki_tokens.pt")
    ap.add_argument("--out",default="tinyllama_wiki")
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--resume")
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--accum_steps",type=int,default=8)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--stride",type=int,default=256)
    ap.add_argument("--max_tokens",type=int,default=12_000_000)
    train(ap.parse_args())
