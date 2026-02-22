#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm

from train import LLaMA, Config   # importa tu modelo EXACTO

@torch.no_grad()
def generate(
    model,
    sp,
    prompt,
    max_new_tokens=128,
    temperature=0.9,
    top_k=40,
    device="cuda"
):
    model.eval()
    ids = [sp.bos_id()] + sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if x.size(1) > Config.max_position_embeddings:
            x = x[:, -Config.max_position_embeddings:]

        logits = model(x)[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

        x = torch.cat([x, next_id], dim=1)

        if next_id.item() == sp.eos_id():
            break

    return sp.decode(x[0].tolist())


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sp = spm.SentencePieceProcessor(model_file=args.sp)

    model = LLaMA(Config)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.to(device)

    out = generate(
        model,
        sp,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        device=device
    )

    print("\n=== OUTPUT ===\n")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sp", required=True)
    ap.add_argument("--prompt", default="Érase una vez")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    main(ap.parse_args())
