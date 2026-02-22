#!/usr/bin/env python3
import argparse, sentencepiece as spm
from pathlib import Path

def main(args):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.Train(
        input=args.corpus,
        model_prefix=args.out.replace(".model",""),
        vocab_size=args.vocab,
        model_type="bpe",
        character_coverage=0.9995,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_id=0,
        byte_fallback=True,
        normalization_rule_name="nfkc",

        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True
    )



    print(f"[OK] Tokenizer generado: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="tokenizer.model")
    ap.add_argument("--vocab", type=int, default=8192)
    main(ap.parse_args())
