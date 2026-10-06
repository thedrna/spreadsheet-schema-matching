"""
Run Data Mapping Pipeline
========================

This script orchestrates the complete data mapping pipeline from Excel files
to standardized CSV format.

Usage:
    python run_mapping_pipeline.py --mode preprocess  # Generate jellyfish input
    python run_mapping_pipeline.py --mode process     # Process with existing mappings
    python run_mapping_pipeline.py --mode full        # Full pipeline (if jellyfish results exist)
    python scripts/run_mapping_pipeline.py --mode process --data-dir data --output-dir value_mapping_results_train --jellyfish-results jellyfish_7b_outputs/jellyfish_results_7B_CoT_withSamples.csv
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import List
import logging

from mapping import DataMappingPipeline, get_target_schema, ColumnMapping

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_excel_files(data_directory: str) -> List[str]:
    """Get all Excel files from data directories"""
    excel_files = []
    
    # Check both data and data_new directories
    if os.path.exists(data_directory):
            for file in os.listdir(data_directory):
                if file.endswith('.xlsx') and not file.startswith('~'):  # Ignore temp files
                    excel_files.append(os.path.join(data_directory, file))

    # for directory in [data_directory, data_directory.replace('data', 'data_new')]:
    #     if os.path.exists(directory):
    #         for file in os.listdir(directory):
    #             if file.endswith('.xlsx') and not file.startswith('~'):  # Ignore temp files
    #                 excel_files.append(os.path.join(directory, file))

    return sorted(excel_files)


def preprocess_mode(data_dir: str, output_dir: str):
    """Mode 1: Preprocess files and generate jellyfish input"""
    logger.info("Starting preprocessing mode...")
    
    # Get all Excel files
    excel_files = get_excel_files(data_dir)
    logger.info(f"Found {len(excel_files)} Excel files")
    
    # Initialize pipeline
    target_schema = get_target_schema()
    pipeline = DataMappingPipeline(target_schema)
    
    all_jellyfish_inputs = []
    
    for file_path in excel_files:
        try:
            logger.info(f"Preprocessing: {os.path.basename(file_path)}")
            
            # Simple preprocessing - just create jellyfish input from source columns
            jellyfish_input = pipeline.preprocess_for_jellyfish(file_path)
            
            # Add source file information
            jellyfish_input['source_file'] = os.path.basename(file_path)
            
            all_jellyfish_inputs.append(jellyfish_input)
            
        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {e}")
            continue
    
    if all_jellyfish_inputs:
        # Combine all jellyfish inputs
        combined_input = pd.concat(all_jellyfish_inputs, ignore_index=True)
        
        # Remove duplicates (same column pairs from the same file only)
        # Keep duplicates from different files as they may have different sample values
        combined_input = combined_input.drop_duplicates(
            subset=['source_column', 'target_column', 'source_file']
        ).reset_index(drop=True)
        
        # Save jellyfish input
        output_path = os.path.join(output_dir, 'jellyfish_input_random_samples.csv')
        combined_input.to_csv(output_path, index=False)
        
        logger.info(f"Jellyfish input saved to: {output_path}")
        logger.info(f"Total unique column pairs: {len(combined_input)}")
        
        # Create batch processing script
        # script_content = create_batch_script(output_path)
        # script_path = os.path.join(output_dir, 'run_jellyfish_batch.py')
        
        # with open(script_path, 'w') as f:
        #     f.write(script_content)
        
        # logger.info(f"Batch processing script created: {script_path}")
        logger.info("Next steps:")
        logger.info("1. Upload jellyfish_input_random_samples.csv to your server")
        logger.info("2. Run the jellyfish processing on the server")
        logger.info("3. Download the results and use process mode")
    
    else:
        logger.error("No files were successfully preprocessed")


def process_mode(data_dir: str, output_dir: str, jellyfish_results_path: str):
    """Mode 2: Process files with existing jellyfish mappings"""
    logger.info("Starting processing mode...")
    
    if not os.path.exists(jellyfish_results_path):
        logger.error(f"Jellyfish results file not found: {jellyfish_results_path}")
        return
    
    # Load jellyfish results
    jellyfish_results = pd.read_csv(jellyfish_results_path)
    logger.info(f"Loaded jellyfish results: {len(jellyfish_results)} mappings")
    
    # Initialize pipeline
    target_schema = get_target_schema()
    pipeline = DataMappingPipeline(target_schema, jellyfish_results_path)
    
    # Parse jellyfish results into ColumnMapping objects
    all_mappings = pipeline.column_mapper.parse_jellyfish_results(jellyfish_results)
    
    # Get all Excel files
    excel_files = get_excel_files(data_dir)
    
    all_processed_files = []
    
    for file_path in excel_files:
        try:
            logger.info(f"Processing: {os.path.basename(file_path)}")
            
            # Use the standard processing method that includes structure detection
            # Pass ALL parsed mappings (not only the globally filtered ones) so
            # structure detection and per-file disambiguation can make decisions
            # based on the full set of candidate mappings for that file.
            # Filter mappings to only those that originate from this source file
            basename = os.path.basename(file_path)
            file_mappings = [m for m in all_mappings if getattr(m, 'source_file', None) == basename]

            # If no mappings were found specifically for this file, fall back to
            # global mappings (optional). Here we pass only the file-specific
            # mappings as requested.
            result_df = pipeline.process_single_file(file_path, file_mappings)
            
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                all_processed_files.append(result_df)
                
                # Save individual file result
                output_filename = os.path.basename(file_path).replace('.xlsx', '_processed.csv')
                individual_output_path = os.path.join(output_dir, 'individual', output_filename)
                os.makedirs(os.path.dirname(individual_output_path), exist_ok=True)
                result_df.to_csv(individual_output_path, index=False)
                
                # Log structure detection results
                if 'data_structure' in result_df.columns:
                    structure_type = result_df['data_structure'].iloc[0]
                    logger.info(f"Detected {structure_type} format in {os.path.basename(file_path)}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if all_processed_files:
        # Combine all results
        combined_df = pd.concat(all_processed_files, ignore_index=True)
        
        # Save combined results
        combined_output_path = os.path.join(output_dir, 'combined_processed_data.csv')
        combined_df.to_csv(combined_output_path, index=False)
        
        logger.info(f"Combined results saved to: {combined_output_path}")
        logger.info(f"Total processed records: {len(combined_df)}")
        
        # Generate summary report
        generate_summary_report(combined_df, output_dir)
        
    else:
        logger.error("No files were successfully processed")


def full_mode(data_dir: str, output_dir: str, jellyfish_results_path: str):
    """Mode 3: Full pipeline (preprocess + process)"""
    logger.info("Starting full pipeline mode...")
    
    # First run preprocessing if jellyfish results don't exist
    if not os.path.exists(jellyfish_results_path):
        logger.info("Jellyfish results not found, running preprocessing first...")
        preprocess_mode(data_dir, output_dir)
        logger.info("Please run jellyfish processing on server and then use process mode")
        return
    
    # Run processing mode
    process_mode(data_dir, output_dir, jellyfish_results_path)


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a summary report of the processed data"""
    report_lines = [
        "Data Processing Summary Report",
        "=" * 40,
        f"Generated: {pd.Timestamp.now()}",
        "",
        f"Total records processed: {len(df):,}",
        f"Total unique projects: {df['project'].nunique() if 'project' in df.columns else 'N/A'}",
        f"Total unique agencies: {df['agency'].nunique() if 'agency' in df.columns else 'N/A'}",
        "",
        "Column Coverage:",
    ]
    
    # Analyze column coverage
    target_columns = [schema.name for schema in get_target_schema()]
    for col in target_columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            coverage = (non_null_count / len(df)) * 100
            report_lines.append(f"  {col}: {non_null_count:,}/{len(df):,} ({coverage:.1f}%)")
        else:
            report_lines.append(f"  {col}: Missing")
    
    if 'project' in df.columns:
        report_lines.extend([
            "",
            "Records by Project:",
        ])
        project_counts = df['project'].value_counts()
        for project, count in project_counts.head(10).items():
            report_lines.append(f"  {project}: {count:,}")
    
    if 'agency' in df.columns:
        report_lines.extend([
            "",
            "Records by Agency:",
        ])
        agency_counts = df['agency'].value_counts()
        for agency, count in agency_counts.head(10).items():
            report_lines.append(f"  {agency}: {count:,}")
    
    # Save report
    report_path = os.path.join(output_dir, 'processing_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Summary report saved to: {report_path}")


def create_batch_script(jellyfish_input_path: str) -> str:
    """Create batch processing script for the server"""
    return f'''#!/usr/bin/env python3
"""
Batch processing script for jellyfish server
Run this script on your server with the jellyfish model
"""

import pandas as pd
import sys
import os

# Add the path to your jellyfish interface if needed
# sys.path.append('/path/to/your/jellyfish/code')


def main():
    # Load the jellyfish input
    print("Loading jellyfish input...")
    jellyfish_input = pd.read_csv("{os.path.basename(jellyfish_input_path)}")
    print(f"Loaded {{len(jellyfish_input)}} column pairs to process")
    
    # Initialize server interface (adjust URL if using API, or modify for local processing)
    # For local processing, you might directly use your jellyfish model here
    print("Initializing processing...")
    
    # Process all column mappings
    print("Processing column mappings...")
    results = []
    
    for idx, row in jellyfish_input.iterrows():
        # Your jellyfish processing logic here
        # This is a placeholder - replace with your actual jellyfish code
        
        prompt = f"""Your task is to determine if the two attributes (columns) are semantically equivalent in the context of merging two tables.
Each attribute will be provided by its name and a brief description.
Your goal is to assess if they refer to the same information based on these names and descriptions provided.

Attribute A is [name: {{row['source_column']}}, description: {{row['source_description']}}].
Attribute B is [name: {{row['target_column']}}, description: {{row['target_description']}}].

Are Attribute A and Attribute B semantically equivalent? Choose your answer from: [Yes, No]."""
        
        # Process with your jellyfish model
        # response = your_jellyfish_model.process(prompt)
        response = "PLACEHOLDER - Replace with actual jellyfish processing"
        
        results.append({{
            'source_column': row['source_column'],
            'target_column': row['target_column'],
            'source_description': row['source_description'],
            'target_description': row['target_description'],
            'prompt': prompt,
            'response': response
        }})
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {{idx + 1}}/{{len(jellyfish_input)}} pairs")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = "jellyfish_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {{output_path}}")
    
    # Display summary
    yes_count = sum(1 for response in results_df['response'] if 'yes' in str(response).lower())
    total_count = len(results_df)
    print(f"Mappings with 'Yes': {{yes_count}}/{{total_count}} ({{yes_count/total_count*100:.1f}}%)")

if __name__ == "__main__":
    main()
'''


def main():
    parser = argparse.ArgumentParser(description='Run data mapping pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'process', 'full'],
                       required=True, help='Pipeline mode to run')
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing Excel files')
    parser.add_argument('--output-dir', default='outputs/mapping',
                       help='Output directory for results')
    parser.add_argument('--jellyfish-results', default='outputs/jellyfish_results.csv',
                       help='Path to jellyfish results CSV file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the appropriate mode
    if args.mode == 'preprocess':
        preprocess_mode(args.data_dir, args.output_dir)
    elif args.mode == 'process':
        process_mode(args.data_dir, args.output_dir, args.jellyfish_results)
    elif args.mode == 'full':
        full_mode(args.data_dir, args.output_dir, args.jellyfish_results)


if __name__ == "__main__":
    main()
