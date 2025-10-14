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

# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

def load_model_with_quantization(model_name):
    """Load model with 4-bit quantization for memory efficiency"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    n_gpus = torch.cuda.device_count()
    max_memory = "20000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main():
    print("Loading XSB dataset...")
    xsb_dataset = pd.read_csv('data/XSB.csv', header='infer', usecols=['prompt', 'label', 'focus'])
    
    # Fill NaN values in focus column with empty strings
    xsb_dataset['focus'] = xsb_dataset['focus'].fillna('')
    
    # Sample for testing - uncomment if needed
    xsb_dataset = pd.concat([xsb_dataset[xsb_dataset['label'] == 'safe'].sample(n=5, random_state=42),
                             xsb_dataset[xsb_dataset['label'] == 'unsafe'].sample(n=5, random_state=42)], ignore_index=True)
    
    print(f"XSB dataset shape: {xsb_dataset.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading model: {model_name}")
    
    # Load model with quantization for memory efficiency
    model, tokenizer = load_model_with_quantization(model_name)
    
    # Initialize attribution methods
    print("Initializing attribution methods...")
    
    # 1. Shapley Values
    sv = ShapleyValues(model)
    sv_llm_attr = LLMAttribution(sv, tokenizer)
    
    # 2. Feature Ablation
    fa = FeatureAblation(model)
    fa_llm_attr = LLMAttribution(fa, tokenizer)
    
    # 3. Layer Integrated Gradients
    lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    lig_llm_attr = LLMGradientAttribution(lig, tokenizer)
    
    skip_tokens = [1]  # skip special tokens
    
    # Initialize result lists
    shapley_guesses = []
    shapley_correctness = []
    ablation_guesses = []
    ablation_correctness = []
    gradient_guesses = []
    gradient_correctness = []
    
    print("Processing attributions...")
    
    for idx, row in xsb_dataset.iterrows():
        prompt = row['prompt']
        label = row['label']
        focus = row['focus']
        
        print(f"Processing row {idx + 1}/{len(xsb_dataset)}: {prompt[:50]}...")
        
        # Create input for attribution
        inp = TextTokenInput(
            prompt, 
            tokenizer,
            skip_tokens=skip_tokens,
        )
        
        try:
            # 1. Shapley Values Attribution
            sv_attr_res = sv_llm_attr.attribute(inp, skip_tokens=skip_tokens)
            sv_attr_scores = sv_attr_res.seq_attr.cpu().numpy()
            sv_trigger_idx = np.argmax(np.abs(sv_attr_scores))
            sv_trigger_token = inp.values[sv_trigger_idx].strip()
            shapley_guesses.append(sv_trigger_token)
            shapley_correctness.append(bool(sv_trigger_token in focus))
            
            # 2. Feature Ablation Attribution
            fa_attr_res = fa_llm_attr.attribute(inp, skip_tokens=skip_tokens)
            fa_attr_scores = fa_attr_res.seq_attr.cpu().numpy()
            fa_trigger_idx = np.argmax(np.abs(fa_attr_scores))
            fa_trigger_token = inp.values[fa_trigger_idx].strip()
            ablation_guesses.append(fa_trigger_token)
            ablation_correctness.append(bool(fa_trigger_token in focus))
            
            # 3. Layer Integrated Gradients Attribution
            lig_attr_res = lig_llm_attr.attribute(inp, skip_tokens=skip_tokens)
            lig_attr_scores = lig_attr_res.seq_attr.cpu().numpy()
            lig_trigger_idx = np.argmax(np.abs(lig_attr_scores))
            lig_trigger_token = inp.values[lig_trigger_idx].strip()
            gradient_guesses.append(lig_trigger_token)
            gradient_correctness.append(bool(lig_trigger_token in focus))
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Add empty results for failed cases
            shapley_guesses.append("")
            shapley_correctness.append(False)
            ablation_guesses.append("")
            ablation_correctness.append(False)
            gradient_guesses.append("")
            gradient_correctness.append(False)
    
    # Add results to dataset
    xsb_dataset['shapley_guesses'] = shapley_guesses
    xsb_dataset['shapley_correctness'] = shapley_correctness
    xsb_dataset['ablation_guesses'] = ablation_guesses
    xsb_dataset['ablation_correctness'] = ablation_correctness
    xsb_dataset['gradient_guesses'] = gradient_guesses
    xsb_dataset['gradient_correctness'] = gradient_correctness
    
    # Calculate and print results
    print("\n=== RESULTS ===")
    
    methods = ['shapley', 'ablation', 'gradient']
    for method in methods:
        correctness_col = f'{method}_correctness'
        guesses_col = f'{method}_guesses'
        
        # Overall accuracy
        overall_accuracy = xsb_dataset[correctness_col].mean() * 100
        print(f"\n{method.upper()} VALUES:")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        
        # Accuracy by label
        safe_accuracy = xsb_dataset[xsb_dataset['label'] == 'safe'][correctness_col].mean() * 100
        unsafe_accuracy = xsb_dataset[xsb_dataset['label'] == 'unsafe'][correctness_col].mean() * 100
        
        print(f"Safe samples accuracy: {safe_accuracy:.2f}%")
        print(f"Unsafe samples accuracy: {unsafe_accuracy:.2f}%")
    
    # Save results
    output_file = 'data/xsb_multi_attribution_results.csv'
    xsb_dataset.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Summary comparison
    print("\n=== METHOD COMPARISON ===")
    comparison_df = pd.DataFrame({
        'Method': methods,
        'Overall_Accuracy': [xsb_dataset[f'{method}_correctness'].mean() * 100 for method in methods],
        'Safe_Accuracy': [xsb_dataset[xsb_dataset['label'] == 'safe'][f'{method}_correctness'].mean() * 100 for method in methods],
        'Unsafe_Accuracy': [xsb_dataset[xsb_dataset['label'] == 'unsafe'][f'{method}_correctness'].mean() * 100 for method in methods]
    })
    print(comparison_df.round(2))

if __name__ == "__main__":
    main()