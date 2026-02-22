import torch
from transformers import LlamaConfig, LlamaForCausalLM
import argparse
from pathlib import Path

def convert(ckpt_path, out_dir):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True
    )

    model = LlamaForCausalLM(config)
    hf_sd = model.state_dict()

    # --- embeddings ---
    hf_sd["model.embed_tokens.weight"] = sd["emb.weight"]
    hf_sd["lm_head.weight"] = sd["head.weight"]

    # --- blocks ---
    for i in range(6):
        prefix = f"blocks.{i}"
        hf_prefix = f"model.layers.{i}"

        # RMSNorms
        hf_sd[f"{hf_prefix}.input_layernorm.weight"] = sd[f"{prefix}.n1.w"]
        hf_sd[f"{hf_prefix}.post_attention_layernorm.weight"] = sd[f"{prefix}.n2.w"]

        # QKV split
        qkv = sd[f"{prefix}.a.qkv.weight"]
        q, k, v = qkv.chunk(3, dim=0)

        hf_sd[f"{hf_prefix}.self_attn.q_proj.weight"] = q
        hf_sd[f"{hf_prefix}.self_attn.k_proj.weight"] = k
        hf_sd[f"{hf_prefix}.self_attn.v_proj.weight"] = v
        hf_sd[f"{hf_prefix}.self_attn.o_proj.weight"] = sd[f"{prefix}.a.o.weight"]

        # MLP (SwiGLU)
        hf_sd[f"{hf_prefix}.mlp.gate_proj.weight"] = sd[f"{prefix}.m.g.weight"]
        hf_sd[f"{hf_prefix}.mlp.up_proj.weight"] = sd[f"{prefix}.m.u.weight"]
        hf_sd[f"{hf_prefix}.mlp.down_proj.weight"] = sd[f"{prefix}.m.d.weight"]

    # Final norm
    hf_sd["model.norm.weight"] = sd["norm.w"]

    model.load_state_dict(hf_sd)
    model.save_pretrained(out_dir)

    print("HF export complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    convert(args.ckpt, args.out)
