#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import List, Union, Optional

import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bf16 on modern GPUs, else fp16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # CPU: keep fp32 for stability
    return torch.float32


def _truncate_on_stops(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].rstrip()


def run(
    model: str,
    input_csv: str,
    output_csv: str,
    limit: int = 1,
    prompt: str = "You are a helpful assistant. Given this CSV row as JSON:\n{row_json}\n\nReturn a concise answer.",
    system: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    stop: Union[str, List[str], None] = None,
    trust_remote_code: bool = False,
    device_map: str = "auto",
):
    """
    Minimum-viable CSV -> HF Transformers pipeline.

    Args:
      model: HF repo ID or local path (folder with tokenizer + model files).
      input_csv: Path to CSV with a header row.
      output_csv: Where to save results (adds a `model_output` column).
      limit: How many rows to process (default: 1).
      prompt: Prompt template; can use {row_json} or {column_name}.
      system: Optional system message (used if the tokenizer supports chat templates).
      max_new_tokens, temperature, top_p, do_sample, repetition_penalty: Generation params.
      stop: A stop string or list of stop strings (post-processed client-side).
      trust_remote_code: Set True if the model repo requires custom code.
      device_map: "auto" (default) or explicit mapping for Accelerate.

    Notes:
      - Works for chat-tuned Llama models via `apply_chat_template` if available;
        otherwise falls back to a plain-text prompt.
      - Uses half precision on GPU automatically, full precision on CPU.
    """
    # Normalize stops
    if stop is None:
        stops: List[str] = []
    elif isinstance(stop, str):
        stops = [stop]
    else:
        stops = list(stop)

    # --- Load tokenizer & model
    dtype = _select_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, use_fast=True)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    # Fallback if no pad token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Prepare IO
    in_path = Path(input_csv)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

            # Format prompt (supports {row_json} and {column_name})
            row_json = json.dumps(row, ensure_ascii=False)
            try:
                user_text = prompt.format(row_json=row_json, **row)
            except KeyError:
                user_text = prompt.format(row_json=row_json)

            # Build inputs
            # If the tokenizer exposes a chat template, use it; else plain text
            if hasattr(tokenizer, "apply_chat_template"):
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user_text})
                model_inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                # Simple fallback for plain CAUSAL LM
                prompt_text = (f"{system}\n\n{user_text}" if system else user_text).strip()
                model_inputs = tokenizer(prompt_text, return_tensors="pt")

            # Move to model device (when device_map != "auto", this is still safe)
            if isinstance(model_inputs, dict):
                for k, v in model_inputs.items():
                    model_inputs[k] = v.to(model_obj.device)

            # Generate
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

            # Decode only the newly generated portion if possible
            if "input_ids" in model_inputs:
                gen_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]
            else:
                # Fallback: decode the whole thing and trust stop trimming
                gen_tokens = outputs[0]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            text = _truncate_on_stops(text, stops)

            row["model_output"] = text
            writer.writerow(row)
            processed += 1

    print(f"Done. Processed {processed} row(s). Output saved to: {out_path}")


if __name__ == "__main__":
    fire.Fire(run)
