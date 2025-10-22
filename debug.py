#!/usr/bin/env python3
import csv, json
from pathlib import Path
from typing import List, Union, Optional, Tuple
import re

import fire
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM

from captum.attr import (
    LayerIntegratedGradients,
    FeatureAblation,
    ShapleyValueSampling,
)

DEFAULT_CHAT_TEMPLATE = (
    "{% if messages and messages[0]['role'] == 'system' %}"
    "{% set system_prompt = messages[0]['content'] %}{% set messages = messages[1:] %}"
    "{% else %}{% set system_prompt = '' %}{% endif %}"
    "<|begin_of_text|>"
    "{% if system_prompt %}<|start_header_id|>system<|end_header_id|>\n{{ system_prompt }}<|eot_id|>{% endif %}"
    "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n{{ m['content'] }}<|eot_id|>{% endfor %}"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

from typing import Optional

def _valid_prompt_mask(prompt_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    special_ids = set(tokenizer.all_special_ids or [])
    ids = prompt_ids[0].tolist()
    return torch.tensor([tid not in special_ids for tid in ids], dtype=torch.bool, device=prompt_ids.device)

_MARKERS = ("Ġ", "▁")
def _clean_token(tok: str) -> str:
    for m in _MARKERS:
        tok = tok.replace(m, "")
    tok = re.sub(r"^##", "", tok)
    if tok.startswith("<|") and tok.endswith("|>"):
        tok = ""
    return tok

def _normalize_ws(s: str) -> str:
    # make substring search more robust to template newlines/spaces
    return re.sub(r"\s+", " ", s).strip()

def _span_mask_from_offsets(
    rendered_text: str,
    user_text: str,
    tokenizer,
    prompt_ids: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Prefer mapping by character spans using offset_mapping (fast & accurate).
    Returns a boolean mask (T_prompt,) that is True only for tokens whose offsets
    fall fully inside the user_text span. Returns None if it can't find a span.
    """
    # 1) locate user_text inside rendered_text (robust to whitespace)
    r_norm = _normalize_ws(rendered_text)
    u_norm = _normalize_ws(user_text)
    start = r_norm.find(u_norm)
    if start == -1:
        # try exact (non-normalized) as a secondary attempt
        start = rendered_text.find(user_text)
        if start == -1:
            return None
        end = start + len(user_text)
        ref = rendered_text
    else:
        end = start + len(u_norm)
        ref = r_norm

    # 2) tokenize rendered with offsets (fast tokenizers required; LLaMA fast tokenizer supports this)
    enc = tokenizer(ref, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc.get("offset_mapping", None)
    ids  = enc["input_ids"]
    if offs is None:
        return None

    # 3) map offsets to mask
    mask_list = []
    for (s, e) in offs:
        # token is inside span if it lies fully within [start, end)
        inside = (s >= start) and (e <= end) and (e > s)
        mask_list.append(inside)

    # Now we must align this “no special tokens” encoding with the actual prompt_ids
    # We re-tokenize rendered_text WITHOUT add_special_tokens just like above so ids match.
    # prompt_ids currently came from tokenizer(rendered_text) (likely with add_special_tokens=True).
    # To avoid mismatch, we recompute prompt ids the same way as for offsets and return that mask,
    # and we’ll use that encoding for attribution masking indices.
    # => We return a mask aligned to the no-special tokenization; the caller will also use
    #    a matching tokenization for scoring or will remap. Simpler approach:
    #    Re-tokenize rendered_text for attribution as well with add_special_tokens=False.
    return torch.tensor(mask_list, dtype=torch.bool, device=prompt_ids.device)

def _span_mask_by_subsequence_ids(
    tokenizer, rendered_text: str, user_text: str, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Fallback: find the token-id subsequence of user_text within rendered_text
    (both encoded with add_special_tokens=False). Returns a boolean mask aligned
    to the rendered_text's ids (no specials).
    """
    enc_all = tokenizer(rendered_text, add_special_tokens=False)
    enc_usr = tokenizer(user_text,    add_special_tokens=False)
    hay = enc_all["input_ids"]
    needle = enc_usr["input_ids"]
    if not hay or not needle or len(needle) > len(hay):
        return None

    # naive subsequence search (works fine for typical lengths)
    start = -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            start = i
            break
    if start == -1:
        # try with a leading space (common for BPE/SentencePiece)
        enc_usr2 = tokenizer(" " + user_text, add_special_tokens=False)
        needle2 = enc_usr2["input_ids"]
        for i in range(len(hay) - len(needle2) + 1):
            if hay[i:i+len(needle2)] == needle2:
                start = i
                needle = needle2
                break
        if start == -1:
            return None

    mask = [False] * len(hay)
    for i in range(start, start + len(needle)):
        mask[i] = True
    return torch.tensor(mask, dtype=torch.bool, device=device)

def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _truncate_on_stops(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    i = min((text.find(s) for s in stops if s in text), default=len(text))
    return text[:i].rstrip()

def _render_prompt(tokenizer, user_text: str, system: Optional[str], chat_template: Optional[str]) -> str:
    # Prefer chat template; fall back to plain text
    if hasattr(tokenizer, "apply_chat_template"):
        has_tpl = getattr(tokenizer, "chat_template", None) is not None
        tpl = None if has_tpl else (chat_template or DEFAULT_CHAT_TEMPLATE)
        try:
            return tokenizer.apply_chat_template(
                ([{"role": "system", "content": system}] if system else []) +
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True,
                chat_template=tpl,
                tokenize=False,
            )
        except Exception:
            pass
    return (f"{system}\n\n{user_text}" if system else user_text).strip()

@torch.no_grad()
def _gen(
    model, tokenizer, rendered_text: str, max_new_tokens: int, temperature: float, top_p: float,
    do_sample: bool, repetition_penalty: float
) -> Tuple[Tensor, str]:
    inputs = tokenizer(rendered_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return gen_tokens, text

def _scores_from_attr(attr: Tensor) -> Tensor:
    """
    attr: (1, T, D) or (T, D) → returns (T,) mean absolute attribution over D
    """
    if attr.dim() == 3:
        attr = attr[0]
    return attr.abs().mean(dim=-1)

def _captum_layer_integrated_gradients(
    model, tokenizer, prompt_ids: torch.Tensor, gen_ids: torch.Tensor, pad_id: int, n_steps: int = 16
) -> torch.Tensor:
    emb_layer = model.get_input_embeddings()

    def forward_for_lig(input_ids: torch.Tensor, gen_ids_: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T_prompt)
        B = input_ids.size(0)
        # View the same gen sequence for each batch item (no copy):
        gen_ids_rep = gen_ids_.expand(B, -1)   # (B, L_gen)

        full_ids = torch.cat([input_ids, gen_ids_rep], dim=1)  # (B, T_prompt + L_gen)
        attn = torch.ones_like(full_ids, dtype=torch.long, device=full_ids.device)
        out = model(input_ids=full_ids, attention_mask=attn)
        logits = out.logits  # (B, T, V)

        T_prompt = input_ids.shape[1]
        gen_logits = logits[:, T_prompt-1:-1, :]            # align next-token predictions for gen part
        logps = torch.log_softmax(gen_logits, dim=-1)
        gathered = logps.gather(2, gen_ids_rep.unsqueeze(-1)).squeeze(-1)  # (B, L_gen)
        return gathered.sum(dim=1)  # (B,)

    lig = LayerIntegratedGradients(forward_for_lig, emb_layer)
    baseline_ids = torch.full_like(prompt_ids, fill_value=pad_id)
    attributions, _ = lig.attribute(
        inputs=prompt_ids,
        baselines=baseline_ids,
        additional_forward_args=(gen_ids,),   # (1, L_gen) —> expanded inside forward
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return _scores_from_attr(attributions)

def _captum_feature_ablation(
    model, tokenizer, prompt_ids: Tensor, gen_ids: Tensor, device: torch.device
) -> Tensor:
    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        prompt_embeds = emb_layer(prompt_ids)  # (1, T, D)
        gen_embeds = emb_layer(gen_ids)        # (1, L, D)
    prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)

    def forward_on_prompt_embeds(p_embeds: torch.Tensor) -> torch.Tensor:
        # p_embeds: (B, T_prompt, D)
        B, T_prompt, _ = p_embeds.shape
        gen_embeds_rep = gen_embeds.detach().expand(B, -1, -1)  # (B, L_gen, D)

        full_embeds = torch.cat([p_embeds, gen_embeds_rep], dim=1)  # (B, T_prompt + L_gen, D)
        attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=full_embeds.device)
        out = model(inputs_embeds=full_embeds, attention_mask=attn)
        logits = out.logits
        gen_logits = logits[:, T_prompt-1:-1, :]
        logps = torch.log_softmax(gen_logits, dim=-1)

        # labels view for gather must match batch
        gen_ids_rep = gen_ids.to(p_embeds.device).expand(B, -1)     # (B, L_gen)
        gathered = logps.gather(2, gen_ids_rep.unsqueeze(-1)).squeeze(-1)  # (B, L_gen)
        return gathered.sum(dim=1)  # (B,)

    ablator = FeatureAblation(forward_on_prompt_embeds)
    T_prompt = prompt_embeds.shape[1]
    feature_mask = torch.arange(T_prompt, device=device).unsqueeze(0).unsqueeze(-1).expand_as(prompt_embeds)

    attributions = ablator.attribute(
        prompt_embeds,
        feature_mask=feature_mask,
        perturbations_per_eval=1,
        baselines=torch.zeros_like(prompt_embeds),
    )
    return _scores_from_attr(attributions)

def _captum_shapley(
    model, tokenizer, prompt_ids: Tensor, gen_ids: Tensor, device: torch.device, nsamples: int = 64
) -> Tensor:
    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        prompt_embeds = emb_layer(prompt_ids)  # (1, T, D)
        gen_embeds = emb_layer(gen_ids)        # (1, L, D)
    prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)

    def forward_on_prompt_embeds(p_embeds: Tensor) -> Tensor:
        T_prompt = p_embeds.shape[1]
        full_embeds = torch.cat([p_embeds, gen_embeds.detach()], dim=1)
        attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=full_embeds.device)
        out = model(inputs_embeds=full_embeds, attention_mask=attn)
        logits = out.logits
        gen_logits = logits[:, T_prompt-1:-1, :]
        logps = torch.log_softmax(gen_logits, dim=-1)
        gathered = logps.gather(2, gen_ids.to(device).unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=1)

    shap = ShapleyValueSampling(forward_on_prompt_embeds)
    T_prompt = prompt_embeds.shape[1]
    feature_mask = torch.arange(T_prompt, device=device).unsqueeze(0).unsqueeze(-1).expand_as(prompt_embeds)

    attributions = shap.attribute(
        prompt_embeds,
        feature_mask=feature_mask,
        n_samples=nsamples,
        baselines=torch.zeros_like(prompt_embeds),
    )
    return _scores_from_attr(attributions)

def run(
    model: str,
    input_csv: str,
    output_csv: str,
    limit: int = 2,
    # CSV columns: {id,prompt,type,type_id,label,class,focus,note}
    prompt_col: str = "prompt",
    system: Optional[str] = None,
    chat_template: Optional[str] = None,
    force_plain: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    stop: Union[str, List[str], None] = None,
    trust_remote_code: bool = False,
    device_map: str = "auto",
    # Captum params
    lig_steps: int = 16,
    shap_samples: int = 64,
):
    """
    For each row:
      - Use `prompt` column as input.
      - Generate continuation.
      - Attribute sum log-probs of generated tokens to input prompt tokens.
      - Record the most influential token for LIG, Ablation, and Shapley.
    Adds columns:
      model_output,
      attr_token_lig, attr_idx_lig, attr_score_lig,
      attr_token_ablate, attr_idx_ablate, attr_score_ablate,
      attr_token_shap, attr_idx_shap, attr_score_shap
    """
    # trim display only
    stops: List[str] = [] if stop is None else ([stop] if isinstance(stop, str) else list(stop))

    # Load
    dtype = _select_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, use_fast=True)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code
    ).eval()  # eval mode for attribution
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model_obj.parameters()).device

    # IO
    in_path, out_path = Path(input_csv), Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with in_path.open(newline="", encoding="utf-8") as f_in, out_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header. Expected columns including 'prompt'.")
        fieldnames = list(reader.fieldnames)
        for newcol in [
            "model_output",
            "attr_token_lig","attr_idx_lig","attr_score_lig",
            "attr_token_ablate","attr_idx_ablate","attr_score_ablate",
            "attr_token_shap","attr_idx_shap","attr_score_shap",
        ]:
            if newcol not in fieldnames:
                fieldnames.append(newcol)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if limit and processed >= limit:
                break

            user_text = row.get(prompt_col, "")
            if not user_text:
                row["model_output"] = ""
                for k in ["lig","ablate","shap"]:
                    row[f"attr_token_{k}"] = ""
                    row[f"attr_idx_{k}"] = ""
                    row[f"attr_score_{k}"] = ""
                writer.writerow(row)
                processed += 1
                continue

            # Render + generate
            rendered = _render_prompt(tokenizer, user_text, system, chat_template)
            gen_ids, gen_text = _gen(
                model_obj, tokenizer, rendered,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty
            )
            gen_text = _truncate_on_stops(gen_text, stops)
            row["model_output"] = gen_text

            # Build IDs for attribution on NO-SPECIALS to align with span masks
            enc_prompt_nospec = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            prompt_ids = enc_prompt_nospec["input_ids"].to(device)  # (1, T_prompt_ns)

            # gen_ids we already have from generation; keep as-is
            gen_ids = gen_ids.unsqueeze(0).to(device)  # (1, L_gen)

            # Build mask for the USER span only (aligned to no-special tokenization)
            mask_span = _span_mask_from_offsets(rendered, user_text, tokenizer, prompt_ids)
            if mask_span is None:
                mask_span = _span_mask_by_subsequence_ids(tokenizer, rendered, user_text, device=device)
            # Also exclude specials (though add_special_tokens=False largely removes them)
            mask_valid = _valid_prompt_mask(prompt_ids, tokenizer)
            mask_user = mask_valid & mask_span if mask_span is not None else mask_valid

            if not mask_user.any():
                # Nothing to attribute in user span; write empty attribution fields and continue
                for k in ["lig","ablate","shap"]:
                    row[f"attr_token_{k}"] = ""
                    row[f"attr_idx_{k}"] = ""
                    row[f"attr_score_{k}"] = ""
                writer.writerow(row); processed += 1; continue

            # === 1) LayerIntegratedGradients
            try:
                scores_lig = _captum_layer_integrated_gradients(
                    model_obj, tokenizer, prompt_ids, gen_ids, pad_id, n_steps=lig_steps
                )
                scores_masked = scores_lig.masked_fill(~mask_user, float("-inf"))
                idx_lig = int(torch.argmax(scores_masked).item())
                tok_lig = tokenizer.convert_ids_to_tokens([int(prompt_ids[0, idx_lig].item())])[0]
                row["attr_token_lig"] = _clean_token(tok_lig)
                row["attr_idx_lig"]   = idx_lig
                row["attr_score_lig"] = float(scores_lig[idx_lig].item())
            except Exception as e:
                row["attr_token_lig"] = f"ERROR:{e}"
                row["attr_idx_lig"] = ""
                row["attr_score_lig"] = ""

            # === 2) FeatureAblation
            try:
                scores_ab = _captum_feature_ablation(model_obj, tokenizer, prompt_ids, gen_ids, device=device)
                scores_masked = scores_ab.masked_fill(~mask_user, float("-inf"))
                idx_ab = int(torch.argmax(scores_masked).item())
                tok_ab = tokenizer.convert_ids_to_tokens([int(prompt_ids[0, idx_ab].item())])[0]
                row["attr_token_ablate"] = _clean_token(tok_ab)
                row["attr_idx_ablate"]   = idx_ab
                row["attr_score_ablate"] = float(scores_ab[idx_ab].item())
            except Exception as e:
                row["attr_token_ablate"] = f"ERROR:{e}"
                row["attr_idx_ablate"] = ""
                row["attr_score_ablate"] = ""

            # === 3) ShapleyValueSampling
            try:
                scores_sh = _captum_shapley(model_obj, tokenizer, prompt_ids, gen_ids, device=device, nsamples=shap_samples)
                scores_masked = scores_sh.masked_fill(~mask_user, float("-inf"))
                idx_sh = int(torch.argmax(scores_masked).item())
                tok_sh = tokenizer.convert_ids_to_tokens([int(prompt_ids[0, idx_sh].item())])[0]
                row["attr_token_shap"] = _clean_token(tok_sh)
                row["attr_idx_shap"]   = idx_sh
                row["attr_score_shap"] = float(scores_sh[idx_sh].item())
            except Exception as e:
                row["attr_token_shap"] = f"ERROR:{e}"
                row["attr_idx_shap"] = ""
                row["attr_score_shap"] = ""

            writer.writerow(row)
            processed += 1

    print(f"Done. Processed {processed} row(s). Output saved to: {out_path}")

if __name__ == "__main__":
    fire.Fire(run)
