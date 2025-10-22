#!/usr/bin/env python3
import csv, json
from pathlib import Path
from typing import List, Union, Optional, Tuple

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
    model, tokenizer, prompt_ids: Tensor, gen_ids: Tensor, pad_id: int, n_steps: int = 16
) -> Tensor:
    emb_layer = model.get_input_embeddings()

    def forward_for_lig(input_ids: Tensor, gen_ids: Tensor) -> Tensor:
        full_ids = torch.cat([input_ids, gen_ids], dim=1)
        attn = torch.ones_like(full_ids).to(full_ids.device)
        out = model(input_ids=full_ids, attention_mask=attn)
        logits = out.logits  # (1, T, V)
        T_prompt = input_ids.shape[1]
        gen_logits = logits[:, T_prompt-1:-1, :]
        logps = torch.log_softmax(gen_logits, dim=-1)
        gathered = logps.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)  # (1, L_gen)
        return gathered.sum(dim=1)

    lig = LayerIntegratedGradients(forward_for_lig, emb_layer)
    baseline_ids = torch.full_like(prompt_ids, fill_value=pad_id)
    attributions, _ = lig.attribute(
        inputs=prompt_ids,
        baselines=baseline_ids,
        additional_forward_args=(gen_ids,),
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
    limit: int = 1,
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

            # Build IDs for attribution
            prompt_ids = tokenizer(rendered, return_tensors="pt")["input_ids"].to(device)  # (1, T_prompt)
            gen_ids = gen_ids.unsqueeze(0).to(device)  # (1, L_gen)

            # Skip if nothing generated
            if gen_ids.numel() == 0:
                for k in ["lig","ablate","shap"]:
                    row[f"attr_token_{k}"] = ""
                    row[f"attr_idx_{k}"] = ""
                    row[f"attr_score_{k}"] = ""
                writer.writerow(row)
                processed += 1
                continue

            # === 1) LayerIntegratedGradients
            try:
                scores_lig = _captum_layer_integrated_gradients(
                    model_obj, tokenizer, prompt_ids, gen_ids, pad_id, n_steps=lig_steps
                )
                idx_lig = int(torch.argmax(scores_lig).item())
                tok_lig = tokenizer.convert_ids_to_tokens([int(prompt_ids[0, idx_lig].item())])[0]
                row["attr_token_lig"] = tok_lig
                row["attr_idx_lig"] = idx_lig
                row["attr_score_lig"] = float(scores_lig[idx_lig].item())
            except Exception as e:
                row["attr_token_lig"] = f"ERROR:{e}"
                row["attr_idx_lig"] = ""
                row["attr_score_lig"] = ""

            # === 2) FeatureAblation
            try:
                scores_ab = _captum_feature_ablation(
                    model_obj, tokenizer, prompt_ids, gen_ids, device=device
                )
                idx_ab = int(torch.argmax(scores_ab).item())
                tok_ab = tokenizer.convert_ids_to_tokens([int(prompt_ids[0, idx_ab].item())])[0]
                row["attr_token_ablate"] = tok_ab
                row["attr_idx_ablate"] = idx_ab
                row["attr_score_ablate"] = float(scores_ab[idx_ab].item())
            except Exception as e:
                row["attr_token_ablate"] = f"ERROR:{e}"
                row["attr_idx_ablate"] = ""
                row["attr_score_ablate"] = ""

            # === 3) ShapleyValueSampling
            try:
                scores_sh = _captum_shapley(
                    model_obj, tokenizer, prompt_ids, gen_ids, device=device, nsamples=shap_samples
                )
                idx_sh = int(torch.argmax(scores_sh).item())
                tok_sh = tokenizer.convert_ids_to_tokens([int(prompt_ids[0, idx_sh].item())])[0]
                row["attr_token_shap"] = tok_sh
                row["attr_idx_shap"] = idx_sh
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
