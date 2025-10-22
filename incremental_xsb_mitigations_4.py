#!/usr/bin/env python3
"""
Minimal XSB pipeline (no attention steering, no custom classes).
Does exactly this:
1) load base model (for generation) and big model (for rephrasing)
2) load dataset CSV with columns [prompt,label,focus]
3) iterate rows
4) for each row: run 3 identifiers (SHAP, Feature Ablation, Integrated Gradients*best-effort),
   then run 3 generations: control, ignore-word, rephrased (rephrase done by big model, answered by base model)
5) save per-sample JSONL + identification metrics JSON + simple CSV of generations

Designed to be robust and dependency-minimal. No Fire, no attention hooks, no logits processors.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import (
    FeatureAblation,
    ShapleyValueSampling,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
)
from sklearn.metrics import f1_score, precision_score, recall_score

# -----------------------------
# Utils
# -----------------------------

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_jsonl(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Loading
# -----------------------------

def load_model(model_name: str, device: Optional[str] = None, dtype: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {model_name} on device {device} ...")
    if dtype is None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": 0} if device.startswith("cuda:") else ("auto" if device == "cuda" else None),
        pad_token_id=tok.eos_token_id,
        low_cpu_mem_usage=True,
    )
    print("model loaded.")
    if device == "cpu":
        print("moving model to CPU...")
        model = model.to(device)
    print("set model to eval mode.")
    model.eval()
    return tok, model


@torch.inference_mode()
def greedy_generate(tok, model, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


# -----------------------------
# Identification
# -----------------------------

def init_identifiers(model, tok):
    sv = ShapleyValueSampling(model)
    fa = FeatureAblation(model)
    lig = None
    try:
        lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    except Exception:
        lig = None

    skip_ids = []
    for tid in [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id]:
        if tid is not None:
            skip_ids.append(tid)

    sv_llm = LLMAttribution(sv, tok)
    fa_llm = LLMAttribution(fa, tok)
    lig_llm = LLMGradientAttribution(lig, tok) if lig is not None else None
    return sv_llm, fa_llm, lig_llm, skip_ids


def _top_token_by_attr(inp: TextTokenInput, scores: np.ndarray, tok) -> str:
    best_idx, best_val = None, None
    for i, t in enumerate(inp.values):
        tid = tok.convert_tokens_to_ids(t)
        if tid in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id, tok.unk_token_id):
            continue
        if t.startswith('<|') or t.startswith('['):
            continue
        v = abs(float(scores[i]))
        if best_val is None or v > best_val:
            best_val = v
            best_idx = i
    if best_idx is None:
        return ""
    token = inp.values[best_idx]
    return token.replace('Ġ', ' ').strip()


@torch.inference_mode()
def identify_all(tok, model, prompt: str, sv_llm, fa_llm, lig_llm, skip_ids):
    inp = TextTokenInput(prompt, tok, skip_tokens=skip_ids)
    if len(inp.values) == 0:
        return {"shap": "", "ablation": "", "ig": ""}

    # get a tiny target continuation to condition attribution
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    target = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True) or "."

    preds = {"shap": "", "ablation": "", "ig": ""}
    try:
        sv_res = sv_llm.attribute(inp, target=target, skip_tokens=skip_ids, n_samples=20)
        sv_scores = sv_res.seq_attr.detach().cpu().numpy()
        preds["shap"] = _top_token_by_attr(inp, sv_scores, tok)
    except Exception:
        pass
    try:
        fa_res = fa_llm.attribute(inp, target=target, skip_tokens=skip_ids)
        fa_scores = fa_res.seq_attr.detach().cpu().numpy()
        preds["ablation"] = _top_token_by_attr(inp, fa_scores, tok)
    except Exception:
        pass
    try:
        if lig_llm is not None:
            lig_res = lig_llm.attribute(inp, target=target, skip_tokens=skip_ids)
            lig_scores = lig_res.seq_attr.detach().cpu().numpy()
            preds["ig"] = _top_token_by_attr(inp, lig_scores, tok)
    except Exception:
        pass
    return preds


# -----------------------------
# Mitigations (no attention dependencies)
# -----------------------------

def ignore_word_prompt(prompt: str, focus: str) -> str:
    return f"{prompt}\n\nIgnore token: {focus}" if focus.strip() else prompt


def rephrase_prompt(big_tok, big_model, prompt: str, focus: str) -> str:
    instr = (
        "You are a helpful prompt engineer. Rephrase the user's prompt to avoid using the specific focus word. "
        "The new prompt must preserve the original task intent while steering away from the focus term.\n\n"
        f"Focus word to avoid: {focus}\n"
        f"Original prompt: {prompt}\n\n"
        "Return only the rephrased prompt without commentary."
    )
    return greedy_generate(big_tok, big_model, instr, max_new_tokens=196)


# -----------------------------
# Metrics for identification
# -----------------------------

def _contains(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    return pred.lower() in gold.lower()


def summarize_id(correct_flags: List[bool]) -> Dict[str, float]:
    if not correct_flags:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    acc = float(np.mean(correct_flags)) * 100
    y_true = [1] * len(correct_flags)
    y_pred = [1 if x else 0 for x in correct_flags]
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser("Minimal XSB pipeline")
    p.add_argument("--data", type=str, default="data/XSB.csv")
    p.add_argument("--out", type=str, default="runs/xsb_minimal")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--big_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    seed_everything(42)
    ensure_dir(args.out)

    print(f"Loading dataset from {args.data} ...")
    df = pd.read_csv(args.data, usecols=["prompt", "label", "focus"]).dropna(subset=["prompt"]).reset_index(drop=True)
    df["focus"] = df["focus"].fillna("")

    if args.limit is not None:
        df = df.iloc[args.start: args.start + args.limit].reset_index(drop=True)
    elif args.start > 0:
        df = df.iloc[args.start:].reset_index(drop=True)

    print(f"Dataset size: {len(df)} rows")

    print(f"Loading base model:) {args.base_model}")
    base_tok, base_model = load_model(args.base_model, device=args.device)

    #print(f"Loading big model (for rephrasing): {args.big_model}")
    #big_tok, big_model = load_model(args.big_model, device=args.device)

    print("Initializing identifiers (Captum)...")
    sv_llm, fa_llm, lig_llm, skip_ids = init_identifiers(base_model, base_tok)

    per_sample = []
    shap_ok, abl_ok, ig_ok = [], [], []

    for i, row in df.iterrows():
        prompt = str(row["prompt"]) or ""
        gold_label = str(row["label"]) if not pd.isna(row["label"]) else ""
        gold_focus = str(row["focus"]) if not pd.isna(row["focus"]) else ""
        print(f"\n[{i+1}/{len(df)}] {prompt[:60]!r}")

        # Identification
        preds = identify_all(base_tok, base_model, prompt, sv_llm, fa_llm, lig_llm, skip_ids)
        s_ok = _contains(preds.get("shap", ""), gold_focus)
        a_ok = _contains(preds.get("ablation", ""), gold_focus)
        i_ok = _contains(preds.get("ig", ""), gold_focus)
        shap_ok.append(s_ok); abl_ok.append(a_ok)
        if preds.get("ig", ""):
            ig_ok.append(i_ok)

        # Generations
        control_out = greedy_generate(base_tok, base_model, prompt, max_new_tokens=args.max_new_tokens)
        ig_prompt = ignore_word_prompt(prompt, gold_focus or preds.get("shap", ""))
        ignore_out = greedy_generate(base_tok, base_model, ig_prompt, max_new_tokens=args.max_new_tokens)
        #re_prompt = rephrase_prompt(big_tok, big_model, prompt, gold_focus or preds.get("shap", ""))
        #re_out = greedy_generate(base_tok, base_model, re_prompt, max_new_tokens=args.max_new_tokens)

        per_sample.append({
            "prompt": prompt,
            "label": gold_label,
            "focus": gold_focus,
            "shap_focus": preds.get("shap", ""),
            "ablation_focus": preds.get("ablation", ""),
            "ig_focus": preds.get("ig", ""),
            "ignore_word_prompt": ig_prompt,
            #"rephrased_prompt": re_prompt,
            "control_output": control_out,
            "ignore_word_output": ignore_out,
            #"rephrased_output": re_out,
        })

    # Save outputs
    save_jsonl(per_sample, os.path.join(args.out, "per_sample.jsonl"))

    id_metrics = {
        "shap": summarize_id(shap_ok),
        "feature_ablation": summarize_id(abl_ok),
        "integrated_gradients": summarize_id(ig_ok),
    }
    with open(os.path.join(args.out, "identification_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(id_metrics, f, indent=2)

    # Quick CSV for the three generations
    pd.DataFrame([
        {
            "prompt": r["prompt"],
            "focus": r["focus"],
            "control": r["control_output"],
            "ignore_word": r["ignore_word_output"],
            "rephrased": r["rephrased_output"],
        }
        for r in per_sample
    ]).to_csv(os.path.join(args.out, "generations.csv"), index=False)

    print("\nDone. Wrote:")
    print(" -", os.path.join(args.out, "per_sample.jsonl"))
    print(" -", os.path.join(args.out, "identification_metrics.json"))
    print(" -", os.path.join(args.out, "generations.csv"))


if __name__ == "__main__":
    main()
