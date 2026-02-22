#!/usr/bin/env python3
import argparse, torch, sentencepiece as spm
from pathlib import Path
from tqdm import tqdm

def main(args):
    sp = spm.SentencePieceProcessor(model_file=args.sp)
    out = Path(args.out)

    tokens = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pretokenizando"):
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line)
            tokens.extend(ids)
            tokens.append(sp.eos_id())

    tokens = torch.tensor(tokens, dtype=torch.long)
    torch.save(tokens, out)

    print(f"[OK] tokens.pt generado: {out}")
    print(f"Total tokens: {len(tokens):,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--sp", required=True)
    ap.add_argument("--out", default="tokens.pt")
    main(ap.parse_args())
