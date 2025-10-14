import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import warnings
from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput,
)

# Ignore warnings
warnings.filterwarnings("ignore")


def main():
    print("Loading XSB dataset...")
    xsb_dataset = pd.read_csv('data/XSB.csv', header='infer', usecols=['prompt', 'label', 'focus'])
    
    # Fill NaN values in focus column with empty strings
    xsb_dataset['focus'] = xsb_dataset['focus'].fillna('')
    
    # Filter out empty prompts that might cause issues
    xsb_dataset = xsb_dataset[xsb_dataset['prompt'].str.len() > 0].reset_index(drop=True)
    
    # Sample for testing - uncomment if needed for faster testing
    xsb_dataset = pd.concat([
        xsb_dataset[xsb_dataset['label'] == 'safe'].sample(n=min(5, len(xsb_dataset[xsb_dataset['label'] == 'safe'])), random_state=42),
        xsb_dataset[xsb_dataset['label'] == 'unsafe'].sample(n=min(5, len(xsb_dataset[xsb_dataset['label'] == 'unsafe'])), random_state=42)
    ], ignore_index=True)
    
    print(f"XSB dataset shape: {xsb_dataset.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model with proper padding setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        pad_token_id=tokenizer.eos_token_id
    )
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Initialize attribution methods
    print("Initializing attribution methods...")
    
    # 1. Use ShapleyValueSampling instead of ShapleyValues for computational efficiency
    from captum.attr import ShapleyValueSampling
    sv = ShapleyValueSampling(model)
    sv_llm_attr = LLMAttribution(sv, tokenizer)
    
    # 2. Feature Ablation
    fa = FeatureAblation(model)
    fa_llm_attr = LLMAttribution(fa, tokenizer)
    
    # 3. Layer Integrated Gradients
    try:
        lig = LayerIntegratedGradients(model, model.model.embed_tokens)
        lig_llm_attr = LLMGradientAttribution(lig, tokenizer)
        use_gradients = True
    except AttributeError:
        print("Warning: Could not initialize LayerIntegratedGradients, skipping gradient method")
        use_gradients = False
    
    skip_tokens = [1] if tokenizer.bos_token_id else []  # skip BOS token if present
    
    # Initialize result lists
    shapley_guesses = []
    shapley_correctness = []
    ablation_guesses = []
    ablation_correctness = []
    if use_gradients:
        gradient_guesses = []
        gradient_correctness = []
    
    print("Processing attributions...")
    
    for idx, row in xsb_dataset.iterrows():
        prompt = row['prompt']
        label = row['label']
        focus = row['focus']
        
        print(f"Processing row {idx + 1}/{len(xsb_dataset)}: {prompt[:50]}...")
        
        # Skip empty or very short prompts
        if len(prompt.strip()) < 3:
            print(f"Skipping empty/short prompt at row {idx}")
            shapley_guesses.append("")
            shapley_correctness.append(False)
            ablation_guesses.append("")
            ablation_correctness.append(False)
            if use_gradients:
                gradient_guesses.append("")
                gradient_correctness.append(False)
            continue
        
        try:
            # Create input for attribution
            inp = TextTokenInput(
                prompt, 
                tokenizer,
                skip_tokens=skip_tokens,
            )
            
            # Ensure we have tokens to work with
            if len(inp.values) == 0:
                print(f"No tokens found for prompt at row {idx}")
                shapley_guesses.append("")
                shapley_correctness.append(False)
                ablation_guesses.append("")
                ablation_correctness.append(False)
                if use_gradients:
                    gradient_guesses.append("")
                    gradient_correctness.append(False)
                continue
            
            # Generate a small target for attribution with proper attention mask
            with torch.no_grad():
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True,
                    attention_mask=True
                ).to(device)
                
                outputs = model.generate(
                    inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=5,  # Reduced for efficiency
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )
                target = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            if not target.strip():
                target = "."  # fallback target
            
            # 1. Captum Shapley Values Attribution (using sampling for efficiency)
            try:
                sv_attr_res = sv_llm_attr.attribute(
                    inp, 
                    target=target, 
                    skip_tokens=skip_tokens,
                    n_samples=20  # Limit samples for efficiency
                )
                sv_attr_scores = sv_attr_res.seq_attr.cpu().numpy() if hasattr(sv_attr_res.seq_attr, 'cpu') else sv_attr_res.seq_attr
                sv_trigger_idx = np.argmax(np.abs(sv_attr_scores))
                sv_trigger_token = inp.values[sv_trigger_idx].strip()
                shapley_guesses.append(sv_trigger_token)
                shapley_correctness.append(bool(sv_trigger_token in focus))
            except Exception as e:
                print(f"Error in Shapley Values for row {idx}: {e}")
                shapley_guesses.append("")
                shapley_correctness.append(False)
            
            # 2. Feature Ablation Attribution
            try:
                fa_attr_res = fa_llm_attr.attribute(inp, target=target, skip_tokens=skip_tokens)
                fa_attr_scores = fa_attr_res.seq_attr.cpu().numpy() if hasattr(fa_attr_res.seq_attr, 'cpu') else fa_attr_res.seq_attr
                fa_trigger_idx = np.argmax(np.abs(fa_attr_scores))
                fa_trigger_token = inp.values[fa_trigger_idx].strip()
                ablation_guesses.append(fa_trigger_token)
                ablation_correctness.append(bool(fa_trigger_token in focus))
            except Exception as e:
                print(f"Error in Feature Ablation for row {idx}: {e}")
                ablation_guesses.append("")
                ablation_correctness.append(False)
            
            # 3. Layer Integrated Gradients Attribution
            if use_gradients:
                try:
                    lig_attr_res = lig_llm_attr.attribute(inp, target=target, skip_tokens=skip_tokens)
                    lig_attr_scores = lig_attr_res.seq_attr.cpu().numpy() if hasattr(lig_attr_res.seq_attr, 'cpu') else lig_attr_res.seq_attr
                    lig_trigger_idx = np.argmax(np.abs(lig_attr_scores))
                    lig_trigger_token = inp.values[lig_trigger_idx].strip()
                    gradient_guesses.append(lig_trigger_token)
                    gradient_correctness.append(bool(lig_trigger_token in focus))
                except Exception as e:
                    print(f"Error in Integrated Gradients for row {idx}: {e}")
                    gradient_guesses.append("")
                    gradient_correctness.append(False)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Add empty results for failed cases
            shapley_guesses.append("")
            shapley_correctness.append(False)
            ablation_guesses.append("")
            ablation_correctness.append(False)
            if use_gradients:
                gradient_guesses.append("")
                gradient_correctness.append(False)
       
    # Add results to dataset
    xsb_dataset['shapley_guesses'] = shapley_guesses
    xsb_dataset['shapley_correctness'] = shapley_correctness
    xsb_dataset['ablation_guesses'] = ablation_guesses
    xsb_dataset['ablation_correctness'] = ablation_correctness
    if use_gradients:
        xsb_dataset['gradient_guesses'] = gradient_guesses
        xsb_dataset['gradient_correctness'] = gradient_correctness
    
    # Calculate and print results
    print("\n=== RESULTS ===")
    
    methods = ['shapley', 'ablation']
    if use_gradients:
        methods.append('gradient')
    
    for method in methods:
        correctness_col = f'{method}_correctness'
        
        # Overall accuracy
        overall_accuracy = xsb_dataset[correctness_col].mean() * 100
        print(f"\n{method.upper()} VALUES:")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        
        # Accuracy by label
        safe_mask = xsb_dataset['label'] == 'safe'
        unsafe_mask = xsb_dataset['label'] == 'unsafe'
        
        if safe_mask.sum() > 0:
            safe_accuracy = xsb_dataset[safe_mask][correctness_col].mean() * 100
            print(f"Safe samples accuracy: {safe_accuracy:.2f}%")
        
        if unsafe_mask.sum() > 0:
            unsafe_accuracy = xsb_dataset[unsafe_mask][correctness_col].mean() * 100
            print(f"Unsafe samples accuracy: {unsafe_accuracy:.2f}%")
    
    # Save results
    output_file = 'data/xsb_multi_attribution_results.csv'
    xsb_dataset.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Summary comparison
    print("\n=== METHOD COMPARISON ===")
    comparison_data = {
        'Method': methods,
        'Overall_Accuracy': [xsb_dataset[f'{method}_correctness'].mean() * 100 for method in methods],
    }
    
    if (xsb_dataset['label'] == 'safe').sum() > 0:
        comparison_data['Safe_Accuracy'] = [
            xsb_dataset[xsb_dataset['label'] == 'safe'][f'{method}_correctness'].mean() * 100 
            for method in methods
        ]
    
    if (xsb_dataset['label'] == 'unsafe').sum() > 0:
        comparison_data['Unsafe_Accuracy'] = [
            xsb_dataset[xsb_dataset['label'] == 'unsafe'][f'{method}_correctness'].mean() * 100 
            for method in methods
        ]
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(2))

if __name__ == "__main__":
    main()