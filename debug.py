#!/usr/bin/env python3
import csv, json
from pathlib import Path
from typing import List, Union, Optional

import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# A safe, minimal Llama-3-style template (works broadly as a generic ChatML-ish template).
# If your model uses a different format, replace this string accordingly.
DEFAULT_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{% set system_prompt = '' %}"
    "{% else %}"
    "{% set system_prompt = messages[0]['content'] %}"
    "{% set messages = messages[1:] %}"
    "{% endif %}"
    "<|begin_of_text|>"
    "{% if system_prompt %}<|start_header_id|>system<|end_header_id|>\n{{ system_prompt }}<|eot_id|>{% endif %}"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _truncate_on_stops(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    cut = min((text.find(s) for s in stops if s in text), default=len(text))
    return text[:cut].rstrip()

def run(
    model: str,
    input_csv: str,
    output_csv: str,
    limit: int = 1,
    prompt: str = "You are a helpful assistant. Given this CSV row as JSON:\n{row_json}\n\nReturn a concise answer.",
    system: Optional[str] = None,
    chat_template: Optional[str] = None,  # NEW: supply your own template if tokenizer lacks one
    force_plain: bool = False,            # NEW: skip chat templating altogether
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    stop: Union[str, List[str], None] = None,
    trust_remote_code: bool = False,
    device_map: str = "auto",
):
    # normalize stops
    stops: List[str] = [] if stop is None else ([stop] if isinstance(stop, str) else list(stop))

    # load
    dtype = _select_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, use_fast=True)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # io
    in_path, out_path = Path(input_csv), Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # helper to build inputs robustly
    def build_inputs(user_text: str):
        if force_plain:
            text = (f"{system}\n\n{user_text}" if system else user_text).strip()
            return tokenizer(text, return_tensors="pt")

        has_apply = hasattr(tokenizer, "apply_chat_template")
        has_template = getattr(tokenizer, "chat_template", None) is not None

        # choose a template: tokenizer's if present, else provided, else default
        templ = None
        if has_template:
            templ = None  # let tokenizer use its own internal template
        elif chat_template:
            templ = chat_template
        else:
            templ = DEFAULT_CHAT_TEMPLATE

        if has_apply:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user_text})

            try:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    chat_template=templ if not has_template else None,
                    return_tensors="pt",
                )
            except Exception:
                # Fall back to plain prompt if anything goes wrong
                pass

        # ultimate fallback: plain prompt
        text = (f"{system}\n\n{user_text}" if system else user_text).strip()
        return tokenizer(text, return_tensors="pt")

    processed = 0
    with in_path.open(newline="", encoding="utf-8") as f_in, out_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header. Please provide a header row.")
        fieldnames = list(reader.fieldnames)
        if "model_output" not in fieldnames:
            fieldnames.append("model_output")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if limit and processed >= limit:
                break

            row_json = json.dumps(row, ensure_ascii=False)
            try:
                user_text = prompt.format(row_json=row_json, **row)
            except KeyError:
                user_text = prompt.format(row_json=row_json)

            model_inputs = build_inputs(user_text)
            for k, v in model_inputs.items():
                model_inputs[k] = v.to(model_obj.device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            with torch.no_grad():
                outputs = model_obj.generate(**model_inputs, **gen_kwargs)

            if "input_ids" in model_inputs:
                gen_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]
            else:
                gen_tokens = outputs[0]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            text = _truncate_on_stops(text, stops)

            row["model_output"] = text
            writer.writerow(row)
            processed += 1

    print(f"Done. Processed {processed} row(s). Output saved to: {out_path}")

if __name__ == "__main__":
    fire.Fire(run)
