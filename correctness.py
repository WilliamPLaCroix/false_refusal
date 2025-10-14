import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import shap
import torch

def main():
    print("Loading XSB dataset...")
    xsb_dataset = pd.read_csv('data/XSB.csv', header='infer', usecols=['prompt', 'label', 'focus'])  # limit to 10 rows for testing
    
    # Fill NaN values in focus column with empty strings
    xsb_dataset['focus'] = xsb_dataset['focus'].fillna('')
    
    # # sample 10 rows, ensure at least one safe and one unsafe
    # xsb_dataset = pd.concat([xsb_dataset[xsb_dataset['label'] == 'safe'].sample(n=5, random_state=42),
    #                          xsb_dataset[xsb_dataset['label'] == 'unsafe'].sample(n=5, random_state=42)], ignore_index=True)
    print(f"XSB dataset shape: {xsb_dataset.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # set model decoder to true
    model.config.is_decoder = True
    # set text-generation params under task_specific_params
    model.config.task_specific_params = model.config.task_specific_params or {}
    model.config.task_specific_params["text-generation"] = {
        "do_sample": True,
        "max_length": 50,
        "temperature": 0,
        "top_k": 5,
        "no_repeat_ngram_size": 2,
    }
    explainer = shap.Explainer(model, tokenizer)

    guesses = []
    correctness = []
    shap_values = explainer(xsb_dataset["prompt"])

    for row in xsb_dataset.itertuples():
        index = row.Index
        prompt = row.prompt
        label = row.label
        focus = row.focus
        #print(f"Prompt: {prompt}\nLabel: {label}\nFocus: {focus}")

        #print(shap_values)
        trigger_token_index = np.mean(shap_values.values[index], axis=1).argmax()
        trigger_token = shap_values[index].data[trigger_token_index]
        #print(f"Trigger token: {trigger_token}\n")
        guesses.append(trigger_token.strip())
        correctness.append(bool(trigger_token in focus))

    print(f"Guesses: {guesses}\nCorrectness: {correctness}")

    xsb_dataset['guesses'] = guesses
    xsb_dataset['correctness'] = correctness

    # find percent correct
    percent_correct = sum(correctness) / len(correctness) * 100
    print(f"Percent correct: {percent_correct:.2f}%")

    # find percent corrent when label is safe
    safe_rows = xsb_dataset[xsb_dataset['label'] == 'safe']
    percent_correct_safe = sum(safe_rows['correctness']) / len(safe_rows) * 100
    print(f"Percent correct when label is safe: {percent_correct_safe:.2f}%")
    # find percent corrent when label is unsafe
    unsafe_rows = xsb_dataset[xsb_dataset['label'] == 'unsafe']
    percent_correct_unsafe = sum(unsafe_rows['correctness']) / len(unsafe_rows) * 100
    print(f"Percent correct when label is unsafe: {percent_correct_unsafe:.2f}%")

    xsb_dataset.to_csv('data/xsb_with_guesses.csv', index=False)

if __name__ == "__main__":
    main()
