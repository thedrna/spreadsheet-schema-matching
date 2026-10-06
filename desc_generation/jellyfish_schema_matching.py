import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import time

# ===== CONFIGURATION =====
# Configuration is read from environment variables set by the bash script
# If running standalone, default values will be used

CONFIG = {
    "model_id": os.environ.get("JELLYFISH_MODEL_ID", "NECOUDBFM/Jellyfish-7B"),
    "use_cot": os.environ.get("JELLYFISH_USE_COT", "False") == "True",
    "include_source_samples": os.environ.get("JELLYFISH_INCLUDE_SAMPLES", "True") == "True",
    "test_mode": os.environ.get("JELLYFISH_TEST_MODE", "False") == "True",
    "test_random": os.environ.get("JELLYFISH_TEST_RANDOM", "False") == "True",
    "test_sample_count": int(os.environ.get("JELLYFISH_TEST_SAMPLE_COUNT", "5")),
    "use_filtering": os.environ.get("JELLYFISH_USE_FILTERING", "True") == "True",
    "filter_top_k": int(os.environ.get("JELLYFISH_FILTER_TOP_K", "5")),
    "filter_threshold": float(os.environ.get("JELLYFISH_FILTER_THRESHOLD", "0.25")),
    "embedding_model": os.environ.get("JELLYFISH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "deduplicate_pairs": os.environ.get("JELLYFISH_DEDUPLICATE_PAIRS", "True") == "True",
}

# Automatically set max_new_tokens based on CoT
CONFIG["max_new_tokens"] = 150 if CONFIG["use_cot"] else 5

# INPUT_CSV = "outputs/generated_descriptions_df.csv"
INPUT_CSV = os.environ.get("JELLYFISH_INPUT_CSV", "outputs/generated_descriptions_df.csv")

# Use RUN_DIR if provided by bash script, otherwise create a new one
if "RUN_DIR" in os.environ:
    OUTPUT_DIR = os.environ["RUN_DIR"]
else:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"outputs/jellyfish_schema_matching/run_{ts}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate output filename based on config
model_size = "7B" if "7B" in CONFIG["model_id"] else "8B"
cot_label = "CoT" if CONFIG["use_cot"] else "noCoT"
samples_label = "withSamples" if CONFIG["include_source_samples"] else "noSamples"
filter_label = f"filtered_top{CONFIG['filter_top_k']}_th{CONFIG['filter_threshold']}" if CONFIG["use_filtering"] else "noFilter"
OUTPUT_CSV = f"{OUTPUT_DIR}/jellyfish_results_{model_size}_{cot_label}_{samples_label}_{filter_label}.csv"


def create_prompt(row, source_desc, use_cot=True, include_samples=True):
    """Create prompt for Jellyfish model
    
    Args:
        row: DataFrame row with column information
        source_desc: The source description to use (can be source_description_1, 2, or 3)
        use_cot: Whether to use chain of thought prompting
        include_samples: Whether to include sample values
    """
    
    source_col = row['source_column']
    target_col = row['target_column']
    target_desc = row['target_description']
    
    # Build sample values string if needed
    samples_text = ""
    if include_samples:
        sample_values = []
        for col in ['sample_1', 'sample_2', 'sample_3']:
            v = row[col] if col in row else None
            if str(v).strip() and str(v) != 'nan' and pd.notna(v):
                sample = str(v).strip()
                if len(sample) > 100:
                    sample = sample[:100] + "..."
                sample_values.append(sample)
        
        if sample_values:
            samples_text = f"\nSource Column Sample Values: {', '.join(sample_values)}"
    
    if use_cot:
        prompt = f"""Your task is to determine if the two attributes (columns) are semantically equivalent in the context of merging two tables.
Each attribute will be provided by its name and a brief description.
Your goal is to assess if they refer to the same information based on these names and descriptions provided.
While answering, provide detailed explanation and justify your answer.

Attribute A is [name: {source_col}, description: {source_desc}{samples_text}].
Attribute B is [name: {target_col}, description: {target_desc}].

Are Attribute A and Attribute B semantically equivalent? After your reasoning, finish your response in a separate line with and ONLY with your final answer. Choose your final answer from [Yes, No].
"""


    else:
        prompt = f"""Your task is to determine if the two attributes (columns) are semantically equivalent in the context of merging two tables.
Each attribute will be provided by its name and a brief description.
Your goal is to assess if they refer to the same information based on these names and descriptions provided.

Attribute A is [name: {source_col}, description: {source_desc}{samples_text}].
Attribute B is [name: {target_col}, description: {target_desc}].

Are Attribute A and Attribute B semantically equivalent? Choose your answer from: [Yes, No].
"""
    
    return prompt


def process_output(raw_output, use_cot=True):
    """Extract Yes/No from model output"""
    
    output_lower = raw_output.lower()
    
    if use_cot:
        # For CoT, look for final answer after reasoning
        if 'yes' in output_lower:
            return 'Yes'
        elif 'no' in output_lower:
            return 'No'
        else:
            return 'Unknown'
    else:
        # For non-CoT, look for direct Yes/No
        if 'yes' in output_lower:
            return 'Yes'
        elif 'no' in output_lower:
            return 'No'
        else:
            return 'Unknown'


def vote_outputs(output1, output2, output3):
    """Vote on 3 processed outputs using majority rule
    
    Args:
        output1, output2, output3: Processed outputs ('Yes', 'No', or 'Unknown')
    
    Returns:
        The majority vote result. In case of tie, prioritizes 'Yes' > 'No' > 'Unknown'
    """
    from collections import Counter
    
    votes = [output1, output2, output3]
    vote_counts = Counter(votes)
    
    # Get the most common vote
    most_common = vote_counts.most_common()
    
    # If there's a clear majority (2 or 3 votes)
    if most_common[0][1] >= 2:
        return most_common[0][0]
    
    # If all three are different (tie), prioritize Yes > No > Unknown
    if 'Yes' in votes:
        return 'Yes'
    elif 'No' in votes:
        return 'No'
    else:
        return 'Unknown'


def embedding_based_filtering(df, model_name='all-MiniLM-L6-v2', top_k=5, threshold=0.25, output_dir=None):
    """
    Pre-filter candidates using semantic embeddings.
    Reduces comparisons while catching semantic similarity (e.g., Document ↔ Section).
    With multiple source descriptions, computes embeddings for all 3 and uses the maximum similarity.
    
    Args:
        df: DataFrame with source_column, source_description_1/2/3, target_column, target_description
        model_name: Sentence transformer model to use
        top_k: Keep top K most similar targets per source column
        threshold: Minimum similarity score to keep (0-1 range)
        output_dir: Directory to save filtering results
    
    Returns:
        Filtered DataFrame with only candidate pairs to check
    """
    print(f"\n🔄 Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("✓ Embedding model loaded")
    
    # Get unique source and target columns
    print(f"\n🔄 Preparing unique source and target columns...")
    source_cols = ['source_column', 'source_description_1', 'source_description_2', 'source_description_3']
    source_data = df[source_cols].drop_duplicates().reset_index(drop=True)
    target_data = df[['target_column', 'target_description']].drop_duplicates().reset_index(drop=True)
    
    print(f"  Unique sources: {len(source_data)}")
    print(f"  Unique targets: {len(target_data)}")
    
    # Create text representations for all 3 source descriptions
    print(f"\n🔄 Creating text representations for 3 source descriptions...")
    source_texts_1 = source_data.apply(
        lambda x: f"{x['source_column']} {x['source_description_1']}", axis=1
    ).tolist()
    source_texts_2 = source_data.apply(
        lambda x: f"{x['source_column']} {x['source_description_2']}", axis=1
    ).tolist()
    source_texts_3 = source_data.apply(
        lambda x: f"{x['source_column']} {x['source_description_3']}", axis=1
    ).tolist()
    target_texts = target_data.apply(
        lambda x: f"{x['target_column']} {x['target_description']}", axis=1
    ).tolist()
    
    # Compute embeddings for all 3 source descriptions
    print(f"\n🔄 Computing embeddings...")
    print(f"  Encoding {len(source_texts_1)} source columns (description 1)...")
    source_embeddings_1 = model.encode(source_texts_1, show_progress_bar=True, batch_size=32)
    print(f"  Encoding {len(source_texts_2)} source columns (description 2)...")
    source_embeddings_2 = model.encode(source_texts_2, show_progress_bar=True, batch_size=32)
    print(f"  Encoding {len(source_texts_3)} source columns (description 3)...")
    source_embeddings_3 = model.encode(source_texts_3, show_progress_bar=True, batch_size=32)
    print(f"  Encoding {len(target_texts)} target columns...")
    target_embeddings = model.encode(target_texts, show_progress_bar=True, batch_size=32)
    
    # Compute similarity matrices for all 3 descriptions
    print(f"\n🔄 Computing similarity matrices...")
    similarity_matrix_1 = cosine_similarity(source_embeddings_1, target_embeddings)
    similarity_matrix_2 = cosine_similarity(source_embeddings_2, target_embeddings)
    similarity_matrix_3 = cosine_similarity(source_embeddings_3, target_embeddings)
    
    # Take maximum similarity across all 3 descriptions for each (source, target) pair
    print(f"\n🔄 Taking maximum similarity across 3 descriptions...")
    max_similarity_matrix = np.maximum(np.maximum(similarity_matrix_1, similarity_matrix_2), similarity_matrix_3)
    print(f"  Max similarity matrix shape: {max_similarity_matrix.shape}")
    
    # Filter candidates using max similarity
    print(f"\n🔄 Filtering candidates (top_k={top_k}, threshold={threshold})...")
    filtered_pairs = []
    
    for i in range(len(source_data)):
        source_row = source_data.iloc[i]
        similarities = max_similarity_matrix[i]
        
        # Get top-k most similar targets
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        for j in top_indices:
            sim_score = similarities[j]
            if sim_score >= threshold:
                target_row = target_data.iloc[j]
                
                # Determine which description gave the max score
                desc_scores = [
                    similarity_matrix_1[i, j],
                    similarity_matrix_2[i, j],
                    similarity_matrix_3[i, j]
                ]
                best_desc_idx = np.argmax(desc_scores) + 1
                
                filtered_pairs.append({
                    'source_column': source_row['source_column'],
                    'target_column': target_row['target_column'],
                    'similarity_score': round(float(sim_score), 4),
                    'best_description_num': best_desc_idx
                })
    
    print(f"  Generated {len(filtered_pairs)} candidate pair entries")
    
    # Create filtered candidates DataFrame
    candidates_df = pd.DataFrame(filtered_pairs)
    
    # CRITICAL: Deduplicate candidate pairs (keep highest similarity score)
    print(f"  Deduplicating candidate pairs...")
    candidates_df = candidates_df.sort_values('similarity_score', ascending=False).drop_duplicates(
        subset=['source_column', 'target_column'], keep='first'
    ).reset_index(drop=True)
    print(f"  Unique candidate pairs after deduplication: {len(candidates_df)}")
    
    # Check unique pairs in original data for comparison
    original_unique_pairs = df[['source_column', 'target_column']].drop_duplicates()
    print(f"  Original unique pairs in data: {len(original_unique_pairs)}")
    
    # Merge back to get full rows from original dataframe
    print(f"\n🔄 Merging with original data...")
    filtered_df = candidates_df.merge(
        df, 
        on=['source_column', 'target_column'], 
        how='inner'
    )
    print(f"  Merged to {len(filtered_df)} rows (same pairs may have multiple rows with different sample sets)")
    
    # Save filtering results if output directory provided
    if output_dir:
        candidates_file = f"{output_dir}/filtered_candidates_top{top_k}_th{threshold}.csv"
        print(f"\n💾 Saving filtered candidates to {candidates_file}...")
        candidates_df.to_csv(candidates_file, index=False)
        print("✓ Filtered candidates saved")
        
        # Also save full filtered data
        filtered_full_file = f"{output_dir}/filtered_full_data_top{top_k}_th{threshold}.csv"
        print(f"💾 Saving full filtered data to {filtered_full_file}...")
        filtered_df.to_csv(filtered_full_file, index=False)
        print("✓ Full filtered data saved")
    
    return filtered_df


def main():
    """Main processing function"""
    start_time = time.time()
    
    print("="*60)
    print("JELLYFISH SCHEMA MATCHING")
    print("="*60)
    print(f"\n📋 CONFIGURATION:")
    print(f"  Model: {CONFIG['model_id']}")
    print(f"  Chain of Thought: {CONFIG['use_cot']}")
    print(f"  Include Source Samples: {CONFIG['include_source_samples']}")
    print(f"  Max New Tokens: {CONFIG['max_new_tokens']}")
    print(f"  Test Mode: {CONFIG['test_mode']}")
    if CONFIG['test_mode']:
        print(f"    - Random Sampling: {CONFIG['test_random']}")
        print(f"    - Sample Count: {CONFIG['test_sample_count']}")
    print(f"  Use Filtering: {CONFIG['use_filtering']}")
    if CONFIG['use_filtering']:
        print(f"    - Embedding Model: {CONFIG['embedding_model']}")
        print(f"    - Top K: {CONFIG['filter_top_k']}")
        print(f"    - Threshold: {CONFIG['filter_threshold']}")
    print(f"  Deduplicate Pairs: {CONFIG['deduplicate_pairs']}")
    print(f"\n📂 Input: {INPUT_CSV}")
    print(f"📂 Output: {OUTPUT_CSV}")
    print("="*60)

    # Load data
    print("\n🔄 Loading input data...")
    full_df = pd.read_csv(INPUT_CSV)
    full_df = full_df[full_df['target_column'] != 'project'].reset_index(drop=True)
    print(f"✓ Filtered out rows with target_column='project'. Remaining rows: {len(full_df)}")

    # Apply test mode settings
    if CONFIG['test_mode']:
        if CONFIG['test_random']:
            df = full_df.sample(n=CONFIG['test_sample_count'], random_state=42).reset_index(drop=True)
            print(f"✓ Test mode: Using {len(df)} random samples from {len(full_df)} total rows")
        else:
            df = full_df.head(CONFIG['test_sample_count']).reset_index(drop=True)
            print(f"✓ Test mode: Using first {len(df)} rows from {len(full_df)} total rows")
    else:
        df = full_df
        print(f"✓ Production mode: Processing all {len(df)} rows")
    
    # Apply embedding-based filtering if enabled
    if CONFIG['use_filtering']:
        print(f"\n{'='*60}")
        print("EMBEDDING-BASED PRE-FILTERING")
        print(f"{'='*60}")
        print(f"Original pairs: {len(df)}")
        
        filtered_df = embedding_based_filtering(
            df,
            model_name=CONFIG['embedding_model'],
            top_k=CONFIG['filter_top_k'],
            threshold=CONFIG['filter_threshold'],
            output_dir=OUTPUT_DIR
        )
        
        reduction_pct = (1 - len(filtered_df) / len(df)) * 100
        time_saved_hours = (len(df) - len(filtered_df)) / 60  # Assuming 1 min per pair
        
        # Get unique pair counts for clarity
        original_unique = df[['source_column', 'target_column']].drop_duplicates()
        filtered_unique = filtered_df[['source_column', 'target_column']].drop_duplicates()
        
        print(f"\n📊 FILTERING SUMMARY:")
        print(f"  Original rows: {len(df)}")
        print(f"  Original unique pairs: {len(original_unique)}")
        print(f"  Filtered rows: {len(filtered_df)}")
        print(f"  Filtered unique pairs: {len(filtered_unique)}")
        print(f"  Row reduction: {reduction_pct:.1f}%")
        print(f"  Unique pair reduction: {(1 - len(filtered_unique)/len(original_unique))*100:.1f}%")
        print(f"  Estimated time saved: {time_saved_hours:.1f} hours")
        print(f"{'='*60}\n")
        
        df = filtered_df
    else:
        print(f"\n⚠️  Filtering disabled - processing all pairs")
    
    # Deduplicate column pairs if enabled (keep first occurrence, save all for later reconstruction)
    df_with_duplicates = df.copy()  # Save original with all duplicates for reconstruction
    
    if CONFIG['deduplicate_pairs']:
        original_rows = len(df)
        df = df.drop_duplicates(subset=['source_column', 'target_column'], keep='first').reset_index(drop=True)
        if len(df) < original_rows:
            print(f"\n🔄 Deduplicating for processing efficiency:")
            print(f"  Original rows: {original_rows}")
            print(f"  Unique pairs to process: {len(df)}")
            print(f"  Will process {len(df)} pairs, then propagate results to all {original_rows} rows")
            print(f"  Estimated time saved: {(original_rows - len(df))/60:.1f} hours")
    else:
        df_with_duplicates = None  # No deduplication needed
    
    
    # Load model
    print(f"\n🔄 Loading Jellyfish model: {CONFIG['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_id'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ Model loaded successfully")
    
    # Process each row
    print(f"\n🔄 Processing {len(df)} rows with 3 descriptions each...")
    results = []
    
    for idx, row in df.iterrows():
        row_start_time = time.time()
        
        print(f"\n--- Row {idx+1}/{len(df)} ---")
        print(f"Source: {row['source_column']} -> Target: {row['target_column']}")
        
        # Process with each of the 3 source descriptions
        prompts = []
        raw_outputs = []
        processed_outputs = []
        
        for desc_num in [1, 2, 3]:
            desc_col = f'source_description_{desc_num}'
            
            # Check if column exists
            if desc_col not in row or pd.isna(row[desc_col]):
                print(f"  ⚠️  Warning: {desc_col} not found or is NaN, skipping")
                prompts.append('')
                raw_outputs.append('N/A')
                processed_outputs.append('Unknown')
                continue
            
            source_desc = row[desc_col]
            
            print(f"\n  Processing with description {desc_num}...")
            
            # Create prompt
            prompt = create_prompt(
                row,
                source_desc=source_desc,
                use_cot=CONFIG['use_cot'],
                include_samples=CONFIG['include_source_samples']
            )
            
            print(f"  Prompt length: {len(prompt)} chars")
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            print(f"  Generating response (max {CONFIG['max_new_tokens']} tokens)...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG['max_new_tokens'],
                    do_sample=False,  # Deterministic output
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            raw_output = raw_output[len(prompt):].strip()
            
            # Process output
            processed_output = process_output(raw_output, use_cot=CONFIG['use_cot'])
            
            print(f"  Raw output: {raw_output[:100]}..." if len(raw_output) > 100 else f"  Raw output: {raw_output}")
            print(f"  Processed: {processed_output}")
            
            # Store outputs
            prompts.append(prompt)
            raw_outputs.append(raw_output)
            processed_outputs.append(processed_output)
        
        # Vote on the final output
        voted_output = vote_outputs(processed_outputs[0], processed_outputs[1], processed_outputs[2])
        
        print(f"\n  📊 Voting Results: [{processed_outputs[0]}, {processed_outputs[1]}, {processed_outputs[2]}] -> {voted_output}")
        
        row_time = time.time() - row_start_time
        print(f"⏱️  Row processing time: {row_time:.2f} seconds")
        
        # Store results
        result = row.to_dict()
        result['prompt_1'] = prompts[0]
        result['prompt_2'] = prompts[1]
        result['prompt_3'] = prompts[2]
        result['raw_output_1'] = raw_outputs[0]
        result['raw_output_2'] = raw_outputs[1]
        result['raw_output_3'] = raw_outputs[2]
        result['processed_output_1'] = processed_outputs[0]
        result['processed_output_2'] = processed_outputs[1]
        result['processed_output_3'] = processed_outputs[2]
        result['processed_output'] = voted_output
        result['processing_time_seconds'] = round(row_time, 2)
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Propagate results back to all duplicate rows if deduplication was used
    if df_with_duplicates is not None and len(df_with_duplicates) > len(results_df):
        print(f"\n🔄 Propagating results to all duplicate pairs...")
        print(f"  Processed unique pairs: {len(results_df)}")
        print(f"  Total rows to reconstruct: {len(df_with_duplicates)}")
        
        # Drop similarity_score before merging (it's from filtering, not needed in final output)
        df_with_duplicates_clean = df_with_duplicates.drop(columns=['similarity_score'], errors='ignore')
        
        # Create a lookup of results by (source_column, target_column)
        results_lookup = results_df[['source_column', 'target_column', 
                                     'prompt_1', 'prompt_2', 'prompt_3',
                                     'raw_output_1', 'raw_output_2', 'raw_output_3',
                                     'processed_output_1', 'processed_output_2', 'processed_output_3',
                                     'processed_output']].copy()
        
        # Merge results back to all original rows
        reconstructed_df = df_with_duplicates_clean.merge(
            results_lookup,
            on=['source_column', 'target_column'],
            how='left'
        )
        
        # Fill Unknown/N/A for any pairs that weren't processed
        for i in [1, 2, 3]:
            reconstructed_df[f'processed_output_{i}'] = reconstructed_df[f'processed_output_{i}'].fillna('Unknown')
            reconstructed_df[f'raw_output_{i}'] = reconstructed_df[f'raw_output_{i}'].fillna('Not processed (filtered out)')
            reconstructed_df[f'prompt_{i}'] = reconstructed_df[f'prompt_{i}'].fillna('Not generated (filtered out)')
        reconstructed_df['processed_output'] = reconstructed_df['processed_output'].fillna('Unknown')
        
        # Check for duplicates
        duplicate_check = reconstructed_df.groupby(['source_column', 'target_column', 'source_file']).size()
        if (duplicate_check > 1).any():
            print(f"  ⚠️  WARNING: Found {(duplicate_check > 1).sum()} unexpected duplicate triplets!")
            print(f"  This may indicate an issue in the input data.")
        
        print(f"  Reconstructed {len(reconstructed_df)} total rows")
        print(f"  Matched pairs: {(reconstructed_df['processed_output'] != 'Unknown').sum()}")
        print(f"  Filtered out (Unknown): {(reconstructed_df['processed_output'] == 'Unknown').sum()}")
        
        results_df = reconstructed_df
    
    # Save results
    print(f"\n💾 Saving results to {OUTPUT_CSV}...")
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("✓ Results saved successfully")
    
    # Save "Yes" results to separate file (based on voted output)
    yes_results = results_df[results_df['processed_output'] == 'Yes']
    if len(yes_results) > 0:
        yes_output_csv = OUTPUT_CSV.replace('.csv', '_yes_only.csv')
        print(f"\n💾 Saving 'Yes' results (voted) to {yes_output_csv}...")
        yes_results.to_csv(yes_output_csv, index=False)
        print(f"✓ Saved {len(yes_results)} 'Yes' results")
    else:
        print("\n⚠️ No 'Yes' results to save separately")
    
    # Save voting statistics
    voting_stats_file = OUTPUT_CSV.replace('.csv', '_voting_stats.csv')
    print(f"\n💾 Saving voting statistics to {voting_stats_file}...")
    
    voting_stats = []
    for _, row in results_df.iterrows():
        stat = {
            'source_column': row['source_column'],
            'target_column': row['target_column'],
            'output_1': row['processed_output_1'],
            'output_2': row['processed_output_2'],
            'output_3': row['processed_output_3'],
            'processed_output': row['processed_output'],
            'unanimous': row['processed_output_1'] == row['processed_output_2'] == row['processed_output_3'],
            'agreement_level': sum([row['processed_output_1'] == row['processed_output'],
                                   row['processed_output_2'] == row['processed_output'],
                                   row['processed_output_3'] == row['processed_output']])
        }
        voting_stats.append(stat)
    
    voting_stats_df = pd.DataFrame(voting_stats)
    voting_stats_df.to_csv(voting_stats_file, index=False)
    print(f"✓ Saved voting statistics")
    
    # Print summary
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    avg_time_per_row = total_time / len(results_df) if len(results_df) > 0 else 0
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total rows in output: {len(results_df)}")
    if df_with_duplicates is not None and len(df_with_duplicates) > len(results):
        print(f"  Unique pairs processed: {len(results)}")
    print(f"\n  Voted Results:")
    print(f"    Yes: {(results_df['processed_output'] == 'Yes').sum()}")
    print(f"    No: {(results_df['processed_output'] == 'No').sum()}")
    print(f"    Unknown: {(results_df['processed_output'] == 'Unknown').sum()}")
    
    # Calculate agreement statistics
    unanimous = sum([row['processed_output_1'] == row['processed_output_2'] == row['processed_output_3'] 
                     for _, row in results_df.iterrows()])
    majority = sum([sum([row['processed_output_1'] == row['processed_output'],
                        row['processed_output_2'] == row['processed_output'],
                        row['processed_output_3'] == row['processed_output']]) >= 2 
                   for _, row in results_df.iterrows()])
    
    print(f"\n  Agreement Statistics:")
    print(f"    Unanimous (3/3): {unanimous} ({unanimous/len(results_df)*100:.1f}%)")
    print(f"    Majority (2+/3): {majority} ({majority/len(results_df)*100:.1f}%)")
    print(f"    Split decisions: {len(results_df) - majority} ({(len(results_df)-majority)/len(results_df)*100:.1f}%)")
    
    print(f"\n⏱️  TIMING:")
    print(f"  Total runtime: {total_minutes:.2f} minutes ({total_time:.2f} seconds)")
    print(f"  Average per row: {avg_time_per_row:.2f} seconds")
    print(f"  Average per inference: {avg_time_per_row/3:.2f} seconds")
    print("\n✅ Done!")


if __name__ == "__main__":  
    main()