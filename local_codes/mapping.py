"""
Data Mapping Pipeline for EMLI Comments
======================================

This module provides a comprehensive pipeline for mapping data structures between 
source Excel files and target CSV format. The pipeline includes:

1. Preprocessing: Extract correct sheets and columns from Excel files
2. Column Mapping: Use jellyfish tool to map source to target columns
3. Value Transformation: Apply rule-based transformations for cell values


Usage:
    python scripts/run_mapping_pipeline.py --mode process --data-dir data --output-dir value_mapping_results/train_3dsc --jellyfish-results on_server_files/run_<TIMESTAMP>_7B_CoT_withSamples_filtered_top5_th0.25_full_traindata/jellyfish_results_7B_CoT_withSamples_filtered_top5_th0.25.csv
    
    python scripts/run_mapping_pipeline.py --mode process --data-dir data_new --output-dir value_mapping_results/test_3dsc --jellyfish-results on_server_files/run_<TIMESTAMP>_7B_CoT_withSamples_filtered_top5_th0.25_full_testdata/jellyfish_results_7B_CoT_withSamples_filtered_top5_th0.25.csv

"""

import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from difflib import SequenceMatcher

# Import configuration
from config import (
    ProcessingConfig, CustomTransformationRules, 
    HEADER_DETECTION_PATTERNS, FALLBACK_COLUMN_MAPPINGS,
    get_default_config
)


class DataStructureType(Enum):
    """Enum for different data structure types"""
    LONG = "long"   # One row = one comment/response pair
    WIDE = "wide"   # One row = multiple comment/response pairs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColumnMapping:
    """Represents a mapping between source and target columns"""
    source_column: str
    target_column: str
    confidence: str  # 'Yes' or 'No' from jellyfish
    transformation_rules: List[str] = None

    # Optional source file (filename) where this mapping was observed
    source_file: Optional[str] = None
    
    # Optional sample values from source column
    sample_1: Optional[str] = None
    sample_2: Optional[str] = None
    sample_3: Optional[str] = None
    
    def __post_init__(self):
        if self.transformation_rules is None:
            self.transformation_rules = []


@dataclass
class DataStructureInfo:
    """Information about the data structure detected"""
    structure_type: DataStructureType
    comment_columns: List[str]  # List of comment columns
    response_columns: List[str]  # List of response columns
    round_column: Optional[str] = None  # Round column (for long format)
    base_columns: List[str] = None  # Columns to repeat for each comment/response pair
    repeated_patterns: Dict[str, Dict[int, str]] = None  # Repeated column patterns for wide format
    
    def __post_init__(self):
        if self.base_columns is None:
            self.base_columns = []
        if self.repeated_patterns is None:
            self.repeated_patterns = {}


@dataclass
class SchemaDefinition:
    """Defines a schema with column names and descriptions"""
    name: str
    description: str


class ExcelPreprocessor:
    """Handles preprocessing of Excel files to extract clean DataFrames"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or get_default_config()
        # Get header indicators from config instead of hardcoding
        self.common_header_indicators = list(HEADER_DETECTION_PATTERNS.keys())
    
    def find_data_sheet(self, file_path: str) -> str:
        """Find the sheet containing actual data (not summary or metadata)"""
        #TODO: multiple sheets (new data)
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Use config settings for sheet detection
            data_indicators = self.config.prefer_sheets_with_keywords
            avoid_indicators = self.config.skip_sheets_with_keywords
            
            # Collect candidate sheets that meet the criteria
            candidate_sheets = []
            
            for sheet in sheet_names:
                sheet_lower = sheet.lower()
                
                # Avoid summary/metadata sheets
                if any(avoid in sheet_lower for avoid in avoid_indicators):
                    continue
                    
                # Prefer sheets with data indicators
                if any(indicator in sheet_lower for indicator in data_indicators):
                    candidate_sheets.append(sheet)
            
            # If we found candidate sheets, select the one with most rows
            if candidate_sheets:
                if len(candidate_sheets) == 1:
                    return candidate_sheets[0]
                else:
                    # Multiple candidates found - select the one with most rows
                    logger.info(f"Multiple candidate sheets found: {candidate_sheets}. Selecting based on row count.")
                    sheet_row_counts = []
                    
                    for sheet in candidate_sheets:
                        try:
                            df = pd.read_excel(file_path, sheet_name=sheet, header=None)
                            row_count = len(df)
                            sheet_row_counts.append((sheet, row_count))
                            logger.info(f"Sheet '{sheet}': {row_count} rows")
                        except Exception as e:
                            logger.warning(f"Could not read sheet '{sheet}': {e}")
                            sheet_row_counts.append((sheet, 0))  # Assign 0 rows if can't read
                    
                    # Sort by row count (descending) and return the sheet with most rows
                    sheet_row_counts.sort(key=lambda x: x[1], reverse=True)
                    selected_sheet = sheet_row_counts[0][0]
                    selected_rows = sheet_row_counts[0][1]
                    
                    logger.info(f"Selected sheet '{selected_sheet}' with {selected_rows} rows")
                    return selected_sheet
            
            # If no obvious data sheet, return the first one
            return sheet_names[0] if sheet_names else None
            
        except Exception as e:
            logger.error(f"Error finding data sheet in {file_path}: {e}")
            return None
    
    def find_header_row(self, df: pd.DataFrame) -> int:
        """Find the row containing column headers"""
        # First, collect all candidate rows that meet the minimum criteria
        candidate_rows = []
        
        for idx, row in df.iterrows():
            if idx > self.config.max_header_search_rows:  # Use config setting
                break
                
            # Convert row to string and check for header indicators
            row_str = ' '.join(str(cell).lower() for cell in row if pd.notna(cell))
            
            # Count how many header indicators we find
            header_count = sum(1 for indicator in self.common_header_indicators 
                             if indicator in row_str)
            
            # Count non-empty cells in this row
            non_empty_cells = sum(1 for cell in row if pd.notna(cell) and str(cell).strip())
            
            # Calculate average cell length for non-empty cells (headers should be concise)
            non_empty_cell_values = [str(cell).strip() for cell in row if pd.notna(cell) and str(cell).strip()]
            avg_cell_length = sum(len(cell) for cell in non_empty_cell_values) / len(non_empty_cell_values) if non_empty_cell_values else 0
            
            # Header quality score: prefer rows with shorter cell content (likely column names vs data)
            # Penalize rows with very long cell content (likely data rows)
            length_penalty = min(avg_cell_length / self.config.header_length_penalty_divisor, 
                               self.config.header_max_length_penalty)
            header_quality_score = header_count - length_penalty
            
            # Check both conditions: header indicators AND minimum number of non-empty cells
            if (header_count >= self.config.min_header_indicators and 
                non_empty_cells >= self.config.min_header_cells_with_values):
                candidate_rows.append((idx, header_count, non_empty_cells, avg_cell_length, header_quality_score))
        
        # If we found candidate rows, select the best one using multiple criteria
        if candidate_rows:
            # Sort by: header_quality_score (descending), row position (ascending), header_count (descending)
            candidate_rows.sort(key=lambda x: (-x[4], x[0], -x[1]))
            best_row = candidate_rows[0]
            
            logger.info(f"Found {len(candidate_rows)} candidate header rows. "
                       f"Selected row {best_row[0]} with {best_row[1]} indicators, {best_row[2]} non-empty cells, "
                       f"avg length {best_row[3]:.1f}, quality score {best_row[4]:.2f}.")
            
            return best_row[0]
        
        return 0  # Default to first row
    
    def clean_empty_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert placeholder empty values to actual empty values (NaN)"""
        # Get empty value placeholders from config
        empty_placeholders = self.config.empty_value_placeholders
        
        # Apply replacement to all cells in the DataFrame
        for placeholder in empty_placeholders:
            df = df.replace(placeholder, np.nan)
        
        # Also handle strings that are just whitespace
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        logger.info(f"Converted empty value placeholders to NaN: {empty_placeholders}")
        return df
    
    def clean_dataframe(self, df: pd.DataFrame, header_row: int) -> pd.DataFrame:
        """Clean the DataFrame by removing unnecessary rows and columns"""
        # Set the header row as column names

        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)

        # if header_row > 0:
        #     df.columns = df.iloc[header_row]
        #     df = df.iloc[header_row + 1:].reset_index(drop=True)
        # elif header_row == 0:
        #     # Header is in the first row - set column names and remove the header row from data
        #     df.columns = df.iloc[header_row]
        #     df = df.iloc[1:].reset_index(drop=True)
        # # If header_row is negative or not found, keep the data as-is with default column names
        
        # Convert empty value placeholders to actual NaN values
        df = self.clean_empty_values(df)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() if pd.notna(col) else f"Column_{i}" 
                     for i, col in enumerate(df.columns)]
        
        # Remove duplicate column names by adding suffix
        seen = {}
        new_columns = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df.columns = new_columns
        
        return df
    
    def preprocess_excel(self, file_path: str) -> pd.DataFrame:
        """Main preprocessing function for Excel files"""
        logger.info(f"Preprocessing Excel file: {file_path}")
        
        # Find the correct sheet
        sheet_name = self.find_data_sheet(file_path)
        if not sheet_name:
            raise ValueError(f"No suitable data sheet found in {file_path}")
        
        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find header row
        header_row = self.find_header_row(df)
        
        # Clean the DataFrame
        df_clean = self.clean_dataframe(df, header_row)
        
        logger.info(f"Preprocessed {file_path}: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean


class DataStructureDetector:
    """Detects whether data is in long or wide format using jellyfish mapping results"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or get_default_config()
    
    def detect_structure(self, df: pd.DataFrame, column_mappings: Optional[List[ColumnMapping]] = None) -> DataStructureInfo:
        """Detect the data structure using jellyfish mapping results"""
        logger.info("Detecting data structure using jellyfish mappings...")
        
        if column_mappings is None:
            # Fallback to pattern-based detection if no mappings available
            logger.warning("No column mappings provided, falling back to pattern-based detection")
            return self._detect_structure_by_patterns(df)
        
        # Use jellyfish mappings to determine structure
        return self._detect_structure_by_mappings(df, column_mappings)
    
    def _detect_structure_by_mappings(self, df: pd.DataFrame, column_mappings: List[ColumnMapping]) -> DataStructureInfo:
        """Detect structure using jellyfish mapping results"""
        
        # Group resolved mappings by target column
        target_mappings = {}
        for mapping in column_mappings:
            target_col = mapping.target_column
            if target_col not in target_mappings:
                target_mappings[target_col] = []
            target_mappings[target_col].append(mapping.source_column)
        
        # Find comment and response columns from mappings
        # ONLY look for exact matches to comment_text and response_text
        comment_columns = target_mappings.get('comment_text', [])
        response_columns = target_mappings.get('response_text', [])
        
        logger.info(f"Mapping detected {len(comment_columns)} comment column(s), {len(response_columns)} response column(s)")
        
        # Remove duplicates while preserving order
        comment_columns = list(dict.fromkeys(comment_columns))
        response_columns = list(dict.fromkeys(response_columns))
        
        # Determine structure type based on number of comment/response columns
        if len(comment_columns) <= 1 or len(response_columns) <= 1:
            structure_type = DataStructureType.LONG
            logger.info(f"Mapping detected LONG format: {len(comment_columns)} comment column(s), {len(response_columns)} response column(s)")
        else:
            structure_type = DataStructureType.WIDE
            logger.info(f"Mapping detected WIDE format: {len(comment_columns)} comment column(s), {len(response_columns)} response column(s)")

        # Find round column from mappings
        round_column = None
        round_columns = target_mappings.get('round', [])
        if round_columns:
            round_column = round_columns[0]  # Take the first one if multiple
        
        # Identify base columns (all columns not used for comment/response/round)
        used_columns = set(comment_columns + response_columns + ([round_column] if round_column else []))
        
        # For wide format, also exclude round-specific metadata columns
        if structure_type == DataStructureType.WIDE:
            # Detect repeated patterns to find all round-specific columns
            repeated_patterns = self._detect_repeated_column_patterns(df, column_mappings, use_fallback=False)
            
            # Add all round-specific columns to used_columns (except 'base' entries)
            for metadata_type, rounds in repeated_patterns.items():
                for round_num, col_name in rounds.items():
                    if isinstance(round_num, int):  # Skip 'base' key
                        used_columns.add(col_name)
        else:
            # For long format, no repeated patterns
            repeated_patterns = {}
        
        base_columns = [col for col in df.columns if col not in used_columns]
        
        return DataStructureInfo(
            structure_type=structure_type,
            comment_columns=comment_columns,
            response_columns=response_columns,
            round_column=round_column,
            base_columns=base_columns,
            repeated_patterns=repeated_patterns
        )
    
    def _detect_structure_by_patterns(self, df: pd.DataFrame) -> DataStructureInfo:
        """Fallback method: Detect structure using header patterns (original implementation)"""
        logger.info("Using pattern-based structure detection...")
        
        # Get column names
        columns = df.columns.tolist()
        
        # Find comment and response columns using patterns
        comment_columns = self._find_comment_columns(columns)
        response_columns = self._find_response_columns(columns)
        
        # Determine structure type
        if len(comment_columns) <= 1 or len(response_columns) <= 1:
            structure_type = DataStructureType.LONG
            round_column = self._find_round_column(columns)
            base_columns = self._identify_base_columns(columns, comment_columns + response_columns + ([round_column] if round_column else []))
            
            logger.info(f"Pattern detected LONG format: {len(comment_columns)} comment column(s), {len(response_columns)} response column(s)")
            
        else:
            structure_type = DataStructureType.WIDE
            round_column = None  # No explicit round column in wide format
            base_columns = self._identify_base_columns(columns, comment_columns + response_columns)

            logger.info(f"Pattern detected WIDE format: {len(comment_columns)} comment column(s), {len(response_columns)} response column(s)")

        return DataStructureInfo(
            structure_type=structure_type,
            comment_columns=comment_columns,
            response_columns=response_columns,
            round_column=round_column,
            base_columns=base_columns,
            repeated_patterns={}  # Pattern-based detection doesn't compute repeated patterns
        )
    
    def _find_comment_columns(self, columns: List[str]) -> List[str]:
        """Find columns that contain comments"""
        comment_patterns = HEADER_DETECTION_PATTERNS.get('comment', [])
        comment_columns = []
        
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in comment_patterns):
                comment_columns.append(col)
        
        # Also look for numbered comment columns (Comment 1, Comment 2, etc.)
        for col in columns:
            if re.search(r'comment\s*\d+', col.lower()) or re.search(r'comment_\d+', col.lower()):
                if col not in comment_columns:
                    comment_columns.append(col)
        
        return sorted(comment_columns)
    
    def _find_response_columns(self, columns: List[str]) -> List[str]:
        """Find columns that contain responses"""
        response_patterns = HEADER_DETECTION_PATTERNS.get('response', [])
        response_columns = []
        
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in response_patterns):
                response_columns.append(col)
        
        # Also look for numbered response columns
        for col in columns:
            if re.search(r'response\s*\d+', col.lower()) or re.search(r'response_\d+', col.lower()):
                if col not in response_columns:
                    response_columns.append(col)
        
        return sorted(response_columns)
    
    def _find_round_column(self, columns: List[str]) -> Optional[str]:
        """Find the column that contains round information"""
        round_patterns = HEADER_DETECTION_PATTERNS.get('round', [])
        
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in round_patterns):
                return col
        
        return None
    
    def _identify_base_columns(self, all_columns: List[str], exclude_columns: List[str]) -> List[str]:
        """Identify columns that should be repeated for each comment/response pair"""
        # Base columns are those that aren't comment, response, or round columns
        # These typically include: ID, section, author, agency, etc.
        base_columns = []
        
        for col in all_columns:
            if col not in exclude_columns:
                base_columns.append(col)
        
        return base_columns

    def _detect_repeated_column_patterns(self, df: pd.DataFrame, 
                                       column_mappings: Optional[List[ColumnMapping]] = None,
                                       use_fallback: bool = True) -> Dict[str, Dict[int, str]]:
        """Detect repeated column patterns for wide format transformation
        
        This method handles cases where columns repeat based on position rather than explicit round numbering.
        For example: id, author, date_received, comment, response, date_received, comment, response, ...
        First appearance = round 1, second appearance = round 2, etc.
        
        Args:
            df: DataFrame to analyze
            column_mappings: List of column mappings from jellyfish results
            use_fallback: Whether to use pattern-based fallback detection when mappings are insufficient
        
        Returns:
            Dictionary mapping metadata types to round-specific columns
        """
        
        round_metadata = {
            'date_received': {},
            'date_responded': {},
            'status': {},
            'comment_type': {},
            'comment_text': {},
            'response_text': {}
        }
        
        # Create mapping from target columns back to source column patterns
        target_to_patterns = {}
        if column_mappings:
            # Group source columns by target
            for mapping in column_mappings:
                if mapping.confidence.lower() == 'yes':
                    target = mapping.target_column
                    if target not in target_to_patterns:
                        target_to_patterns[target] = []
                    target_to_patterns[target].append(mapping.source_column)
        
        # For each target column type, find all matching source columns and assign round numbers based on position
        for target_type in round_metadata.keys():
            if target_type in target_to_patterns:
                source_columns = target_to_patterns[target_type]
                
                # Sort by column position to maintain order
                columns_with_pos = [(col, df.columns.get_loc(col)) for col in source_columns if col in df.columns]
                columns_with_pos.sort(key=lambda x: x[1])  # Sort by position
                
                # Assign round numbers based on position (1-indexed)
                for i, (col_name, position) in enumerate(columns_with_pos):
                    round_num = i + 1
                    round_metadata[target_type][round_num] = col_name
                    
                    # Also store the first occurrence as 'base' for fallback
                    if i == 0:
                        round_metadata[target_type]['base'] = col_name
        
        # Handle case where we don't have mappings - use pattern-based detection
        if use_fallback and (not column_mappings or not any(round_metadata.values())):
            logger.info("No column mappings available, using pattern-based detection for repeated columns")
            self._detect_patterns_by_repetition(df, round_metadata)
        
        return round_metadata

    def _detect_patterns_by_repetition(self, df: pd.DataFrame, round_metadata: Dict[str, Dict[int, str]]):
        """Fallback method to detect repeated patterns when mappings are not available"""
        
        # Define patterns for each metadata type
        patterns = {
            'date_received': ['date', 'received', 'created'],
            'date_responded': ['date', 'respond', 'reply'],
            'status': ['status', 'state'],
            'comment_type': ['type', 'comment'],
            'comment_text': ['comment', 'feedback', 'text'],
            'response_text': ['response', 'reply', 'answer']
        }
        
        # Find columns that match each pattern and assign rounds based on position
        for target_type, pattern_keywords in patterns.items():
            matching_columns = []
            
            for col in df.columns:
                col_lower = col.lower()
                # Check if column matches the pattern (all keywords should be present)
                if all(keyword in col_lower for keyword in pattern_keywords):
                    matching_columns.append((col, df.columns.get_loc(col)))
            
            # Sort by position and assign round numbers
            matching_columns.sort(key=lambda x: x[1])
            
            for i, (col_name, position) in enumerate(matching_columns):
                round_num = i + 1
                round_metadata[target_type][round_num] = col_name
                
                # Also store the first occurrence as 'base' for fallback
                if i == 0:
                    round_metadata[target_type]['base'] = col_name


class ColumnMapper:
    """Handles column mapping using jellyfish tool results"""
    
    def __init__(self, jellyfish_results_path: Optional[str] = None, config: Optional[ProcessingConfig] = None):
        self.config = config or get_default_config()
        self.jellyfish_results = None
        if jellyfish_results_path and os.path.exists(jellyfish_results_path):
            self.jellyfish_results = pd.read_csv(jellyfish_results_path)
    
    def create_jellyfish_input(self, source_columns: List[str], 
                              target_schema: List[SchemaDefinition],
                              source_df: pd.DataFrame = None,
                              sample_count: int = 3) -> pd.DataFrame:
        """Create input for jellyfish tool with source columns and target schema"""
        pairs = []
        
        for source_col in source_columns:
            # Create a description for source column based on its name
            source_desc = self._infer_column_description(source_col)
            
            # Extract sample values if source_df is provided
            sample_values = {}
            if source_df is not None and source_col in source_df.columns:
                # Get non-null values from the column
                non_null_values = source_df[source_col].dropna()
                
                # Get unique values (to avoid duplicate samples)
                unique_values = non_null_values.drop_duplicates()
                
                # Take first 'sample_count' values
                # samples = unique_values.head(sample_count).tolist()

                # Take random 'sample_count' values (or all if fewer available)
                if len(unique_values) > sample_count:
                    samples = unique_values.sample(n=sample_count, random_state=42).tolist()
                else:
                    samples = unique_values.tolist()
                
                # Create sample columns
                for i in range(sample_count):
                    sample_key = f'sample_{i+1}'
                    sample_values[sample_key] = samples[i] if i < len(samples) else ''
            else:
                # If no source_df provided, create empty sample columns
                for i in range(sample_count):
                    sample_values[f'sample_{i+1}'] = ''
            
            for target_schema_item in target_schema:
                pair_data = {
                    'source_column': source_col,
                    'source_description': source_desc,
                    **sample_values,  # Add sample values
                    'target_column': target_schema_item.name,
                    'target_description': target_schema_item.description
                }
                pairs.append(pair_data)
        
        return pd.DataFrame(pairs)
    
    def _infer_column_description(self, column_name: str) -> str:
        """Infer description for a column based on its name using config patterns"""
        column_lower = column_name.lower()
        
        # Use patterns from config instead of hardcoded mapping
        for target_type, patterns in HEADER_DETECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in column_lower:
                    # Create description based on the target type
                    descriptions = {
                        'id': 'Unique identifier for the record',
                        'comment': 'Text content of the comment or feedback',
                        'response': 'Reply or answer to the comment',
                        'author': 'Person or entity who created the content',
                        'date_received': 'Date when the item was created or received',
                        'date_responded': 'Date when response was sent',
                        'status': 'Current state or condition of the item',
                        'section': 'Document section or part being referenced',
                        'topic': 'Subject matter or theme of the comment',
                        'round': 'Review round or phase number',
                        'type': 'Category or classification of the item',
                        'agency': 'Organization or department involved'
                    }
                    return descriptions.get(target_type, f"Data column containing {target_type} information")
        
        return f"Data column containing {column_name.lower()} information"
    
    def parse_jellyfish_results(self, jellyfish_output: pd.DataFrame) -> List[ColumnMapping]:
        """Parse jellyfish tool results into ColumnMapping objects
        
        For source columns that map to multiple target columns, uses string similarity
        to select the best match.
        """
        mappings = []
        
        for _, row in jellyfish_output.iterrows():
            # Safely extract fields from the jellyfish output row
            src_col = row.get('source_column') if 'source_column' in row.index else None
            tgt_col = row.get('target_column') if 'target_column' in row.index else None
            best_mappings = row.get('best_description_num') if 'best_description_num' in row.index else None
            raw_out_col = f'raw_output_{best_mappings}' if best_mappings is not None else 'raw_output'
            
            raw_out = row.get(raw_out_col) if raw_out_col in row.index else None
            src_file = row.get('source_file') if 'source_file' in row.index else None
            
            # Extract sample values if available
            sample_1 = row.get('sample_1') if 'sample_1' in row.index else None
            sample_2 = row.get('sample_2') if 'sample_2' in row.index else None
            sample_3 = row.get('sample_3') if 'sample_3' in row.index else None
            
            # Process raw_output: take first 5 characters and determine Yes/No
            processed_output = ''
            if raw_out is not None:
                first_5_chars = str(raw_out)[:5].strip().lower()
                if 'yes' in first_5_chars:
                    processed_output = 'Yes'
                elif 'no' in first_5_chars:
                    processed_output = 'No'
                else:
                    processed_output = ''

            mapping = ColumnMapping(
                source_column=src_col,
                target_column=tgt_col,
                confidence=processed_output,
                source_file=src_file,
                sample_1=str(sample_1) if pd.notna(sample_1) else None,
                sample_2=str(sample_2) if pd.notna(sample_2) else None,
                sample_3=str(sample_3) if pd.notna(sample_3) else None
            )
            mappings.append(mapping)
        
        # Filter to only 'Yes' mappings
        yes_mappings = [m for m in mappings if m.confidence == 'Yes']
        
        # Disambiguate: for each source column, if multiple target columns exist,
        # choose the one with highest string similarity
        best_mappings = self._disambiguate_mappings_by_similarity(yes_mappings)
        
        # Special handling: keep only first source column per project for certain target columns
        target_columns_to_deduplicate = ['comment_id', 'agency', 'author']
        best_mappings = self._keep_first_mapping_per_project(best_mappings, target_columns_to_deduplicate)

        return best_mappings
    
    def _disambiguate_mappings_by_similarity(self, mappings: List[ColumnMapping]) -> List[ColumnMapping]:
        """For source columns with multiple target mappings, choose best match by string similarity
        
        Args:
            mappings: List of ColumnMapping objects
            
        Returns:
            List of ColumnMapping objects with one mapping per source column
        """
        # Group mappings by source_column, source_file, and sample values
        grouped = {}
        for mapping in mappings:
            # Create a key that includes source_file and samples
            key = (
                mapping.source_column,
                mapping.source_file if mapping.source_file else '',
                mapping.sample_1 if mapping.sample_1 else '',
                mapping.sample_2 if mapping.sample_2 else '',
                mapping.sample_3 if mapping.sample_3 else ''
            )
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(mapping)
        
        best_mappings = []
        
        for key, candidates in grouped.items():
            source_col = key[0]  # First element is source_column
            if len(candidates) == 1:
                # Only one target column, keep it
                best_mappings.append(candidates[0])
            else:
                # Multiple target columns, use string similarity to choose best
                best_mapping = self._select_best_mapping_by_similarity(source_col, candidates)
                best_mappings.append(best_mapping)
                
                logger.info(f"Source column '{source_col}' mapped to {len(candidates)} targets. "
                          f"Selected '{best_mapping.target_column}' based on string similarity.")
        
        return best_mappings
    
    def _select_best_mapping_by_similarity(self, source_column: str, 
                                          candidates: List[ColumnMapping]) -> ColumnMapping:
        """Select the best target column based on string similarity to source column
        
        Args:
            source_column: Name of the source column
            candidates: List of candidate ColumnMapping objects with different target columns
            
        Returns:
            The ColumnMapping with the highest string similarity score
        """
        # Normalize source column name for comparison
        source_normalized = self._normalize_column_name(source_column)
        
        best_score = -1
        best_mapping = candidates[0]  # Fallback to first if all scores are equal
        
        # Log all similarity scores for debugging
        logger.info(f"\n--- Similarity scoring for source column '{source_column}' (normalized: '{source_normalized}') in project {candidates[0].source_file if candidates and candidates[0].source_file else 'unknown_project'} ---")
        similarity_scores = []
        
        for candidate in candidates:
            target_normalized = self._normalize_column_name(candidate.target_column)
            
            # Calculate string similarity using SequenceMatcher
            similarity = SequenceMatcher(None, source_normalized, target_normalized).ratio()
            similarity_scores.append((candidate.target_column, target_normalized, similarity))
            
            logger.info(f"  Target: '{candidate.target_column}' (normalized: '{target_normalized}') - Similarity: {similarity:.4f}")
            
            if similarity > best_score:
                best_score = similarity
                best_mapping = candidate
        
        logger.info(f"  BEST MATCH: '{best_mapping.target_column}' with score {best_score:.4f}")
        logger.info(f"--- End similarity scoring ---\n")
        
        return best_mapping
    
    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column name for similarity comparison
        
        Converts to lowercase, removes special characters, and collapses whitespace
        """
        if not column_name:
            return ''
        
        # Convert to lowercase
        normalized = column_name.lower()
        
        # Remove special characters and replace with spaces
        normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
        
        # Collapse multiple spaces into single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _keep_first_mapping_per_project(self, mappings: List[ColumnMapping], 
                                        target_columns: List[str]) -> List[ColumnMapping]:
        """For specified target columns, keep only the first source column per project
        
        This is useful for columns that should have only one mapping per project,
        such as comment_id, agency, or author.
        
        Args:
            mappings: List of ColumnMapping objects
            target_columns: List of target column names to deduplicate
            
        Returns:
            List of ColumnMapping objects with only first mapping per project for specified targets
        """
        # Separate mappings for target columns from other mappings
        target_mappings = [m for m in mappings if m.target_column in target_columns]
        other_mappings = [m for m in mappings if m.target_column not in target_columns]
        
        if not target_mappings:
            return mappings
        
        # Group target mappings by (target_column, source_file)
        mappings_by_target_and_project = {}
        for mapping in target_mappings:
            project_key = mapping.source_file if mapping.source_file else 'unknown_project'
            key = (mapping.target_column, project_key)
            if key not in mappings_by_target_and_project:
                mappings_by_target_and_project[key] = []
            mappings_by_target_and_project[key].append(mapping)
        
        # Keep only the first mapping per (target_column, project)
        kept_target_mappings = []
        for (target_col, project_key), project_mappings in mappings_by_target_and_project.items():
            if len(project_mappings) > 1:
                logger.info(f"Project '{project_key}': Found {len(project_mappings)} '{target_col}' mappings. "
                          f"Keeping first: '{project_mappings[0].source_column}'")
            kept_target_mappings.append(project_mappings[0])
        
        # Combine back with other mappings
        return other_mappings + kept_target_mappings

    
    
    def get_fallback_mappings(self, source_columns: List[str]) -> List[ColumnMapping]:
        """Get fallback column mappings from config when jellyfish is not available"""
        mappings = []
        
        for source_col in source_columns:
            if source_col in FALLBACK_COLUMN_MAPPINGS:
                target_col = FALLBACK_COLUMN_MAPPINGS[source_col]
                mapping = ColumnMapping(
                    source_column=source_col,
                    target_column=target_col,
                    confidence="Fallback"
                )
                mappings.append(mapping)
                logger.info(f"Fallback mapping: {source_col} -> {target_col}")
        
        return mappings


class DataStructureTransformer:
    """Transforms data from wide/long format to standardized long format"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or get_default_config()
    
    def transform_to_long_format(self, df: pd.DataFrame, 
                               structure_info: DataStructureInfo,
                               column_mappings: Optional[List[ColumnMapping]] = None) -> pd.DataFrame:
        """Transform data to long format (one row per comment/response pair)"""
        
        if structure_info.structure_type == DataStructureType.LONG:
            logger.info("Data is already in long format, processing rounds and deduplicating columns")
            return self._process_long_format(df, structure_info, column_mappings)
        
        elif structure_info.structure_type == DataStructureType.WIDE:
            logger.info("Transforming wide format to long format")
            return self._transform_wide_to_long(df, structure_info, column_mappings)
        
        else:
            logger.warning("Unknown data structure type, returning data as-is")
            return df

    def _process_long_format(self, df: pd.DataFrame,
                           structure_info: DataStructureInfo,
                           column_mappings: Optional[List[ColumnMapping]] = None) -> pd.DataFrame:
        """Process long format data to handle rounds and multiple column mappings
        
        Issues addressed:
        1. Generate round numbers based on sequence of rows with same comment_id
        2. When multiple source columns map to same target (comment_text/response_text),
           select the longest non-empty value
        """
        result_df = df.copy()
        
        if not column_mappings:
            logger.warning("No column mappings provided for long format processing")
            return result_df
        
        # Find comment_id mapping to group rows
        comment_id_col = None
        for mapping in column_mappings:
            if mapping.target_column == 'comment_id' and mapping.confidence.lower() == 'yes':
                comment_id_col = mapping.source_column
                break
        
        # Issue 1: Generate round numbers based on row sequence with same comment_id
        if comment_id_col and comment_id_col in result_df.columns:
            logger.info(f"Generating round numbers based on '{comment_id_col}' grouping")
            
            # Check if round information is embedded in comment_id (e.g., "ABC-123-R1", "AGY_R1.01")
            def extract_base_id_and_round(comment_id_value):
                """Extract base ID and round number from comment_id if pattern exists
                
                Patterns: R1, R2, Round 1, Round 2, etc. (can be at end or in middle)
                Returns: (base_id, round_num) or (comment_id_value, None) if no pattern found
                """
                if pd.isna(comment_id_value):
                    return str(comment_id_value), None
                
                comment_id_str = str(comment_id_value).strip()
                
                # Patterns to match: R1, R2, Round 1, Round 2, etc.
                # Updated to match rounds at the end OR in the middle
                patterns = [
                    (r'[-_\s]R(\d+)(?=[-_\s.]|$)', 'R'),      # Matches: ABC-123-R1, AGY_R1.01, ABC-123_R1
                    (r'[-_\s]Round\s*(\d+)(?=[-_\s.]|$)', 'Round'),  # Matches: ABC-123-Round 1, AGY_Round1.01
                    (r'[-_\s]r(\d+)(?=[-_\s.]|$)', 'r'),      # Matches: ABC-123-r1 (lowercase)
                    (r'[-_\s]round\s*(\d+)(?=[-_\s.]|$)', 'round'),  # Matches: ABC-123-round 1 (lowercase)
                ]
                
                for pattern, prefix in patterns:
                    match = re.search(pattern, comment_id_str, re.IGNORECASE)
                    if match:
                        round_num = int(match.group(1))
                        # Remove the round part to get base ID
                        # Need to reconstruct the full match including separator and prefix
                        full_match = match.group(0)  # e.g., "_R1" or "-Round 2"
                        base_id = comment_id_str.replace(full_match, '', 1).strip()
                        return base_id, round_num
                
                # No pattern found
                return comment_id_str, None
            
            # Extract base IDs and rounds
            extracted = result_df[comment_id_col].apply(extract_base_id_and_round)
            result_df['_base_id'] = extracted.apply(lambda x: x[0])
            result_df['_extracted_round'] = extracted.apply(lambda x: x[1])
            
            # Check if any rounds were extracted
            has_extracted_rounds = result_df['_extracted_round'].notna().any()
            
            if has_extracted_rounds:
                logger.info(f"Detected round information embedded in '{comment_id_col}' column")
                
                # Use extracted rounds where available, otherwise generate sequential rounds per base_id
                def assign_round(group):
                    """Assign rounds within a group (base_id)"""
                    rounds = []
                    for idx, row in group.iterrows():
                        if pd.notna(row['_extracted_round']):
                            # Use extracted round
                            rounds.append(str(int(row['_extracted_round'])))
                        else:
                            # Generate sequential round based on position in group
                            # Start after the max extracted round if any
                            max_extracted = group['_extracted_round'].max()
                            if pd.notna(max_extracted):
                                # Count rows without extracted rounds before this one
                                rows_before = group.loc[:idx]
                                unassigned_before = rows_before['_extracted_round'].isna().sum()
                                rounds.append(str(int(max_extracted) + unassigned_before))
                            else:
                                # No extracted rounds in this group, use sequential
                                rounds.append(str(len(rounds) + 1))
                    return rounds
                
                # Apply round assignment per base_id group
                result_df['round'] = result_df.groupby('_base_id', group_keys=False).apply(
                    lambda g: pd.Series(assign_round(g), index=g.index)
                )
            else:
                # No embedded rounds, use simple sequential numbering per comment_id
                result_df['round'] = result_df.groupby(comment_id_col).cumcount() + 1
                result_df['round'] = result_df['round'].astype(str)
            
            # Clean up temporary columns
            result_df.drop(columns=['_base_id', '_extracted_round'], inplace=True)
            
            logger.info(f"Generated rounds for {len(result_df)} rows")
        else:
            logger.warning("No comment_id mapping found, cannot generate round numbers")
            # Assign round 1 to all rows as fallback
            result_df['round'] = '1'
        
        # Issue 2: Handle multiple source columns mapping to same target
        # Group mappings by target column
        target_to_sources = {}
        for mapping in column_mappings:
            if mapping.confidence.lower() == 'yes':
                target = mapping.target_column
                if target not in target_to_sources:
                    target_to_sources[target] = []
                target_to_sources[target].append(mapping.source_column)
        
        # For comment_text and response_text, if multiple sources exist, select longest
        for target_col in ['comment_text', 'response_text']:
            if target_col in target_to_sources:
                source_cols = target_to_sources[target_col]
                
                if len(source_cols) > 1:
                    logger.info(f"Multiple source columns for '{target_col}': {source_cols}")
                    logger.info(f"Selecting longest non-empty value among them")
                    
                    # Create new column by selecting longest value from source columns
                    def select_longest(*values):
                        """Select the longest non-empty value from multiple columns"""
                        valid_values = []
                        for val in values:
                            if pd.notna(val) and str(val).strip():
                                valid_values.append(str(val).strip())
                        
                        if not valid_values:
                            return ''
                        
                        # Return the longest value
                        return max(valid_values, key=len)
                    
                    # Apply selection across all source columns
                    available_sources = [col for col in source_cols if col in result_df.columns]
                    if available_sources:
                        result_df[target_col] = result_df[available_sources].apply(
                            lambda row: select_longest(*row), axis=1
                        )
                        logger.info(f"Created '{target_col}' by selecting longest from {available_sources}")
                elif len(source_cols) == 1 and source_cols[0] in result_df.columns:
                    # Single source, just rename
                    if source_cols[0] != target_col:
                        result_df[target_col] = result_df[source_cols[0]]
        
        return result_df

    def _transform_wide_to_long(self, df: pd.DataFrame, 
                              structure_info: DataStructureInfo,
                              column_mappings: Optional[List[ColumnMapping]] = None) -> pd.DataFrame:
        """Transform wide format data to long format using repeated column patterns"""
        
        # Get target column names from mappings or use defaults
        comment_target = 'comment_text'
        response_target = 'response_text'
        round_target = 'round'
        
        if column_mappings:
            for mapping in column_mappings:
                if mapping.target_column == 'comment_text' and mapping.confidence == 'Yes':
                    comment_target = 'comment_text'
                elif mapping.target_column == 'response_text' and mapping.confidence == 'Yes':
                    response_target = 'response_text'
                elif mapping.target_column == 'round' and mapping.confidence == 'Yes':
                    round_target = 'round'
        
        # Use repeated column patterns from structure_info (already computed during structure detection)
        repeated_patterns = structure_info.repeated_patterns
        
        # Patterns should always be available since they're computed during structure detection
        if not repeated_patterns or not any(repeated_patterns.values()):
            logger.warning("No repeated patterns available in structure_info. This might indicate an issue with structure detection.")
            logger.warning("Returning data as-is without wide-to-long transformation.")
            return df
        
        # Add logging that was in the original method
        logger.info(f"Using repeated column patterns: {repeated_patterns}")
        
        transformed_rows = []
        
        # Determine the number of rounds based on the maximum round number found
        max_rounds = 0
        for metadata_type, rounds in repeated_patterns.items():
            for round_num in rounds.keys():
                if isinstance(round_num, int):  # Skip 'base' key
                    max_rounds = max(max_rounds, round_num)
        
        if max_rounds == 0:
            logger.warning("No repeated column patterns found")
            return df
        
        logger.info(f"Found {max_rounds} rounds based on repeated column patterns")
        
        # Identify base columns (columns that don't repeat and aren't part of round-specific data)
        all_round_columns = set()
        for metadata_type, rounds in repeated_patterns.items():
            for round_num, col_name in rounds.items():
                if isinstance(round_num, int):  # Skip 'base' key
                    all_round_columns.add(col_name)
        
        base_columns = [col for col in df.columns if col not in all_round_columns]
        
        # Process each row in the original DataFrame
        for idx, row in df.iterrows():
            # Extract base information (columns that don't change per round)
            base_info = {}
            for col in base_columns:
                base_info[col] = row[col]
            
            # Create one new row for each round
            for round_num in range(1, max_rounds + 1):
                new_row = base_info.copy()
                
                # Check if this round has any content
                has_content = False
                
                # Add comment text for this round
                comment_col = repeated_patterns['comment_text'].get(round_num)
                if comment_col and comment_col in df.columns:
                    comment_value = row[comment_col]
                    if pd.notna(comment_value) and str(comment_value).strip():
                        new_row[comment_target] = str(comment_value).strip()
                        has_content = True
                
                # Add response text for this round
                response_col = repeated_patterns['response_text'].get(round_num)
                if response_col and response_col in df.columns:
                    response_value = row[response_col]
                    if pd.notna(response_value) and str(response_value).strip():
                        new_row[response_target] = str(response_value).strip()
                        has_content = True
                    else:
                        new_row[response_target] = ''
                else:
                    new_row[response_target] = ''
                
                # Skip this round if no comment content
                if not has_content:
                    continue
                
                # Add round number
                new_row[round_target] = str(round_num)
                
                # Add metadata for this round (date_received, date_responded, status, comment_type)
                for metadata_type in ['date_received', 'date_responded', 'status', 'comment_type']:
                    metadata_cols = repeated_patterns.get(metadata_type, {})
                    
                    # Try to get round-specific value first
                    metadata_value = None
                    metadata_col = metadata_cols.get(round_num)
                    if metadata_col and metadata_col in df.columns:
                        metadata_value = row[metadata_col]
                    
                    # Fall back to base value if no round-specific value or if round-specific is empty
                    if (pd.isna(metadata_value) or str(metadata_value).strip() == '' or str(metadata_value).strip() == '-'):
                        base_col = metadata_cols.get('base')
                        if base_col and base_col in df.columns:
                            metadata_value = row[base_col]
                    
                    # Add to new row if we have a valid value
                    if pd.notna(metadata_value) and str(metadata_value).strip() and str(metadata_value).strip() != '-':
                        new_row[metadata_type] = metadata_value
                
                # Add the row
                transformed_rows.append(new_row)
        
        # Create new DataFrame
        if transformed_rows:
            result_df = pd.DataFrame(transformed_rows)
            logger.info(f"Transformed {len(df)} wide-format rows to {len(result_df)} long-format rows")
            return result_df
        else:
            logger.warning("No valid comment/response pairs found after transformation")
            return pd.DataFrame()


class ValueTransformer:
    """Handles rule-based transformation of cell values"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or get_default_config()
        self.transformation_rules = self._initialize_transformation_rules()
    
    def _initialize_transformation_rules(self) -> Dict[str, List[callable]]:
        """Initialize transformation rules for different column types"""
        return {
            'agency': [self._extract_agency_code, self._normalize_agency_name],
            'date_received': [self._standardize_date],
            'date_responded': [self._standardize_date],
            'status': [self._normalize_status],
            'comment_type': [self._normalize_comment_type],
            'round': [self._extract_round_number],
            'author': [self._clean_author_name],
        }
    
    def _split_agency_id(self, value: str) -> Dict[str, str]:
        """Extract agency from ID while preserving the original value as comment_id
        
        Examples:
            'EMLI-001' -> {'agency': 'EMLI', 'comment_id': 'EMLI-001'}
            '123' -> {'agency': '', 'comment_id': '123'}
        """
        if pd.isna(value):
            return {'agency': '', 'comment_id': ''}
        
        # Convert to string if it's a number
        if isinstance(value, (int, float)):
            value_clean = str(int(value))  # Convert to string, removing .0 for floats
        elif isinstance(value, str):
            value_clean = value.strip()
        else:
            return {'agency': '', 'comment_id': ''}
        
        # Store the original value as comment_id
        original_value = value_clean
        
        # If the value is just numbers, no agency to extract
        if value_clean.isdigit():
            return {'agency': '', 'comment_id': original_value}
        
        # Use patterns from config for structured IDs like EMLI-001
        patterns = CustomTransformationRules.id_splitting_patterns()
        
        for pattern in patterns:
            match = re.match(pattern, value_clean)
            if match:
                # Extract agency but keep original value as comment_id
                return {'agency': match.group(1), 'comment_id': original_value}
        
        # If no pattern matches, keep the whole value as comment_id with no agency
        return {'agency': '', 'comment_id': original_value}
    
    def _extract_agency_code(self, value: str) -> str:
        """Extract agency abbreviation from full names using config mappings"""
        if pd.isna(value) or not isinstance(value, str):
            return ''
        
        # Use agency mapping from config
        agency_mapping = CustomTransformationRules.agency_name_mapping()
        value_clean = value.strip()
        
        # Check if value is already an abbreviated agency code
        # (appears as an abbreviation in the mapping values)
        all_abbreviations = set(agency_mapping.values())
        if value_clean in all_abbreviations:
            # Already abbreviated, return as-is
            return value_clean
        
        # Direct mapping (full name -> abbreviation)
        if value_clean in agency_mapping:
            return agency_mapping[value_clean]
        
        # Partial matching with full names
        for full_name, abbreviation in agency_mapping.items():
            if full_name.lower() in value_clean.lower():
                return abbreviation
        
        # Partial matching with abbreviations (check if known abbreviation appears as word in value)
        # This handles cases like "EMLI Geotechnical" -> "EMLI"
        for abbreviation in all_abbreviations:
            # Use word boundaries to match whole abbreviations
            pattern = r'\b' + re.escape(abbreviation) + r'\b'
            if re.search(pattern, value_clean, re.IGNORECASE):
                return abbreviation
        
        # Check if it looks like an agency code (2-4 uppercase letters)
        # If so, assume it's already an abbreviation
        if len(value_clean) <= 4 and value_clean.isupper() and value_clean.isalpha():
            return value_clean
        
        # If no match, create abbreviation from first letters of significant words
        words = value_clean.split()
        significant_words = [w for w in words if len(w) > 2 and w.lower() not in 
                           ['of', 'and', 'the', 'for', 'in', 'on', 'at']]
        if significant_words:
            abbreviation = ''.join(word[0].upper() for word in significant_words[:self.config.agency_abbreviation_max_length])
            # If abbreviation is only one character, keep the original value
            if len(abbreviation) == 1:
                return value_clean
            return abbreviation
        #TODO: add new abbreviation to agency_mapping

        # Last resort: truncate to max length
        truncated = value_clean[:self.config.agency_abbreviation_max_length].upper()
        # If truncation results in one character, keep the original value
        if len(truncated) == 1:
            return value_clean
        return truncated
    
    def _normalize_agency_name(self, value: str) -> str:
        """Normalize agency names to standard abbreviations"""
        return self._extract_agency_code(value)
    
    def _extract_agency_from_author(self, value: str) -> str:
        """Extract agency abbreviation from author name by finding known abbreviations
        
        This method searches for known agency abbreviations (from config) within the
        author string, rather than creating new abbreviations from the text.
        
        Example:
            'Jane Doe, EMLI - Geoscience' -> 'EMLI'
        """
        if pd.isna(value) or not isinstance(value, str):
            return ''
        
        # Get all known abbreviations from config
        agency_mapping = CustomTransformationRules.agency_name_mapping()
        all_abbreviations = set(agency_mapping.values())
        
        # Clean the value
        value_clean = value.strip()
        
        # Search for each known abbreviation in the author string
        # Use word boundaries to match whole abbreviations
        for abbreviation in sorted(all_abbreviations, key=len, reverse=True):  # Try longer abbreviations first
            # Create a pattern that matches the abbreviation as a whole word
            # This handles cases like "EMLI - Geoscience" or "EMLI," or "EMLI "
            pattern = r'\b' + re.escape(abbreviation) + r'\b'
            if re.search(pattern, value_clean, re.IGNORECASE):
                return abbreviation
        
        return ''
    
    def _standardize_date(self, value: Any) -> str:
        """Standardize date formats to YYYY-MM-DD using config date formats"""
        if pd.isna(value):
            return ''
        
        # If already a datetime object
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')
        
        # If it's a string, try to parse it using config date formats
        if isinstance(value, str):
            try:
                # Try date formats from config
                for fmt in self.config.date_formats:
                    try:
                        dt = datetime.strptime(value.strip(), fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                
                # If pandas can parse it
                dt = pd.to_datetime(value)
                return dt.strftime('%Y-%m-%d')
                
            except:
                logger.warning(f"Could not parse date: {value}")
                return str(value)
        
        return str(value)
    
    def _normalize_status(self, value: str) -> str:
        """Normalize status values using config mappings"""
        if pd.isna(value) or not isinstance(value, str):
            return ''
        
        value_lower = value.strip().lower()
        
        # Use status mapping from config
        status_mapping = CustomTransformationRules.status_mapping()
        
        # Direct lookup
        if value_lower in status_mapping:
            return status_mapping[value_lower]
        
        # Partial matching for complex status descriptions
        for key, normalized in status_mapping.items():
            if key in value_lower:
                return normalized
        
        return value.strip()
    
    def _normalize_comment_type(self, value: str) -> str:
        """Normalize comment type values using config mappings"""
        if pd.isna(value) or not isinstance(value, str):
            return ''
        
        value_lower = value.strip().lower()
        
        # Use comment type mapping from config
        type_mapping = CustomTransformationRules.comment_type_mapping()
        
        # Direct lookup
        if value_lower in type_mapping:
            return type_mapping[value_lower]
        
        # Partial matching
        for key, normalized in type_mapping.items():
            if key in value_lower:
                return normalized
        return value.strip()
    
    def _extract_round_number(self, value: str) -> str:
        """Extract round number from text using config patterns"""
        if pd.isna(value) or not isinstance(value, str):
            return ''
        
        # Use round extraction patterns from config
        patterns = CustomTransformationRules.round_extraction_patterns()
        
        for pattern in patterns:
            match = re.search(pattern, value.lower())
            if match:
                return match.group(1)
        #TODO: rounds should be int
        return value.strip()
    
    def _clean_author_name(self, value: str) -> str:
        """Clean author names"""
        #TODO: not necessary
        if pd.isna(value) or not isinstance(value, str):
            return ''
        
        # Remove extra whitespace and normalize
        cleaned = ' '.join(value.strip().split())
        
        # Remove common prefixes/suffixes
        prefixes = ['Mr.', 'Ms.', 'Dr.', 'Prof.']
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        return cleaned
    
    def apply_transformations(self, df: pd.DataFrame, 
                            mappings: List[ColumnMapping],
                            preserve_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply transformations to create target DataFrame
        
        Args:
            df: Source DataFrame
            mappings: Column mappings
            preserve_columns: Columns already in target format that should be preserved (e.g., from long format processing)
        """
        result_df = pd.DataFrame()
        
        # Preserve columns that are already in target format (e.g., round, deduplicated comment_text/response_text)
        if preserve_columns:
            for col in preserve_columns:
                if col in df.columns:
                    result_df[col] = df[col]
                    logger.info(f"Preserving pre-transformed column '{col}'")
        
        # Create mapping dictionary for easy lookup
        mapping_dict = {m.source_column: m.target_column for m in mappings}
        
        # Check if there's a direct mapping to agency column
        agency_source = None
        comment_id_source = None
        author_source = None
        
        for source_col, target_col in mapping_dict.items():
            if target_col == 'agency':
                agency_source = source_col
            elif target_col == 'comment_id':
                comment_id_source = source_col
            elif target_col == 'author':
                author_source = source_col
        
        # Process comment_id column to preserve original value
        if comment_id_source and comment_id_source in df.columns:
            # Simply copy the comment_id values directly (no transformation)
            comment_id_values = []
            for value in df[comment_id_source]:
                if pd.isna(value):
                    comment_id_values.append('')
                elif isinstance(value, (int, float)):
                    comment_id_values.append(str(int(value)))
                else:
                    comment_id_values.append(str(value).strip())
            
            result_df['comment_id'] = pd.Series(comment_id_values, dtype='string')
        
        # Handle agency extraction with priority logic
        if agency_source and agency_source in df.columns:
            # Priority 1: Use the dedicated agency column mapping if it exists
            logger.info(f"Using dedicated agency column mapping from '{agency_source}'")
            rules = self.transformation_rules.get('agency', [])
            if rules:
                temp_result = df[agency_source]
                for rule in rules:
                    temp_result = temp_result.apply(rule)
                result_df['agency'] = temp_result
            else:
                result_df['agency'] = df[agency_source]
        elif comment_id_source and comment_id_source in df.columns:
            # Priority 2: Extract agency from comment_id if no dedicated agency mapping
            logger.info(f"No dedicated agency mapping found. Extracting agency from comment_id column '{comment_id_source}'")
            split_results = df[comment_id_source].apply(self._split_agency_id)
            
            # Extract only agency values
            agency_values = []
            for result in split_results:
                if isinstance(result, dict):
                    agency_values.append(str(result.get('agency', '')))
                else:
                    agency_values.append('')
            
            result_df['agency'] = pd.Series(agency_values, dtype='string')
            
            # Priority 3: If agency is still empty and author column exists, extract from author
            if author_source and author_source in df.columns:
                # Check if we have empty agency values
                empty_agency_mask = result_df['agency'].isna() | (result_df['agency'] == '')
                if empty_agency_mask.any():
                    logger.info(f"Some agency values are empty. Attempting to extract from author column '{author_source}'")
                    # Extract agency from author column for rows with empty agency
                    # Use specialized method that finds known abbreviations in author names
                    author_agencies = df.loc[empty_agency_mask, author_source].apply(self._extract_agency_from_author)
                    result_df.loc[empty_agency_mask, 'agency'] = author_agencies
        
        # Process all other columns
        for source_col, target_col in mapping_dict.items():
            if source_col not in df.columns:
                logger.warning(f"Source column {source_col} not found in DataFrame")
                continue
            
            # Skip columns that are already processed (comment_id, agency, or preserved columns)
            skip_columns = ['comment_id', 'agency']
            if preserve_columns:
                skip_columns.extend(preserve_columns)
            if target_col in skip_columns:
                continue
            
            # Get transformation rules for this target column
            rules = self.transformation_rules.get(target_col, [])
            
            if rules:
                # Apply transformations
                for rule in rules:
                    # Regular transformation
                    result_df[target_col] = df[source_col].apply(rule)
            else:
                # No transformation rules, copy as-is
                result_df[target_col] = df[source_col]
        
        # Ensure comment_id remains as string type if it exists  
        if 'comment_id' in result_df.columns:
            # Force pandas StringDtype and ensure no accidental numeric interpretation
            result_df['comment_id'] = (
                result_df['comment_id']
                    .astype('string')
                    .str.strip()
            )
        
        return result_df



class DataMappingPipeline:
    """Main pipeline class that orchestrates the entire mapping process"""
    
    def __init__(self, target_schema: List[SchemaDefinition], 
                 jellyfish_results_path: Optional[str] = None,
                 config: Optional[ProcessingConfig] = None):
        self.target_schema = target_schema
        self.config = config or get_default_config()
        self.preprocessor = ExcelPreprocessor(self.config)
        self.structure_detector = DataStructureDetector(self.config)
        self.structure_transformer = DataStructureTransformer(self.config)
        self.column_mapper = ColumnMapper(jellyfish_results_path, self.config)
        self.value_transformer = ValueTransformer(self.config)
    
    def _ensure_comment_id_mapping(self, source_df: pd.DataFrame, 
                                    column_mappings: List[ColumnMapping]) -> Tuple[pd.DataFrame, List[ColumnMapping]]:
        """Ensure comment_id mapping exists by auto-generating ID column if needed
        
        Args:
            source_df: Source DataFrame
            column_mappings: List of column mappings
            
        Returns:
            Tuple of (modified DataFrame, modified column mappings)
        """
        has_comment_id_mapping = any(m.target_column == 'comment_id' for m in column_mappings)
        
        if not has_comment_id_mapping:
            logger.warning("No comment_id mapping found. Creating auto-generated ID column.")
            # Add a new 'id' column with sequential numbers starting from 1
            source_df.insert(0, 'id', range(1, len(source_df) + 1))
            # Add a mapping for this new column
            auto_id_mapping = ColumnMapping(
                source_column='id',
                target_column='comment_id',
                confidence='Yes'
            )
            column_mappings = [auto_id_mapping] + column_mappings
            logger.info(f"Added auto-generated ID column with {len(source_df)} rows (1 to {len(source_df)})")
        
        return source_df, column_mappings
    
    def process_single_file(self, file_path: str, 
                          column_mappings: Optional[List[ColumnMapping]] = None) -> pd.DataFrame:
        """Process a single Excel file through the complete pipeline"""
        logger.info(f"Processing file: {file_path}")
        
        # Step 1: Preprocess Excel file
        source_df = self.preprocessor.preprocess_excel(file_path)
        
        # Step 2: Get column mappings (if not provided)
        if column_mappings is None:
            # Create jellyfish input from raw source columns - no structure detection needed
            jellyfish_input = self.column_mapper.create_jellyfish_input(
                source_df.columns.tolist(), self.target_schema, source_df,
                sample_count=self.config.sample_values_count
            )
            
            logger.info("Column mappings not provided. You need to run jellyfish tool on the server.")
            logger.info(f"Jellyfish input shape: {jellyfish_input.shape}")
            
            # Return the input for jellyfish processing
            return jellyfish_input
        
        # Step 2.5: Ensure comment_id mapping exists
        source_df, column_mappings = self._ensure_comment_id_mapping(source_df, column_mappings)
        
        # Step 3: Detect data structure using jellyfish mappings
        structure_info = self.structure_detector.detect_structure(source_df, column_mappings)
        
        # Step 4: Transform to standardized long format if needed (now using target column names directly)
        standardized_df = self.structure_transformer.transform_to_long_format(source_df, structure_info, column_mappings)
        
        # Step 5: Apply value transformations to remaining columns (preserved columns only)
        # The comment/response/round columns already have target names, so only transform preserved columns
        if structure_info.structure_type == DataStructureType.WIDE:
            # For wide data that was transformed, only apply transformations to preserved columns
            preserved_mappings = [m for m in column_mappings 
                                if m.source_column in structure_info.base_columns]
            if preserved_mappings:
                result_df = self.value_transformer.apply_transformations(standardized_df, preserved_mappings)
                
                # Add columns from standardized_df that are in the target schema but not yet in result_df
                target_column_names = [schema.name for schema in self.target_schema]
                for col in standardized_df.columns:
                    if col in target_column_names and col not in result_df.columns:
                        result_df[col] = standardized_df[col]
            else:
                result_df = standardized_df
        else:
            # For long data, preserve columns created by _process_long_format (round, deduplicated comment_text/response_text)
            preserve_columns = []
            if 'round' in standardized_df.columns:
                preserve_columns.append('round')
            if 'comment_text' in standardized_df.columns:
                preserve_columns.append('comment_text')
            if 'response_text' in standardized_df.columns:
                preserve_columns.append('response_text')
            
            result_df = self.value_transformer.apply_transformations(standardized_df, column_mappings, preserve_columns=preserve_columns)
        
        # Add metadata
        result_df['source_file'] = os.path.basename(file_path)
        result_df['project'] = self._extract_project_name(file_path)
        result_df['data_structure'] = structure_info.structure_type.value
        
        # Ensure all target schema columns are present (add missing columns with null values)
        target_column_names = [schema.name for schema in self.target_schema]
        for col in target_column_names:
            if col not in result_df.columns:
                result_df[col] = pd.NA
                logger.debug(f"Added missing target column '{col}' with null values")
        
        
        logger.info(f"Processed {file_path}: {len(result_df)} final rows from {structure_info.structure_type.value} format")
        
        return result_df
    
    def preprocess_for_jellyfish(self, file_path: str) -> pd.DataFrame:
        """Preprocess file and return jellyfish input - no structure detection needed"""
        logger.info(f"Preprocessing for jellyfish: {file_path}")
        
        # Step 1: Preprocess Excel file
        source_df = self.preprocessor.preprocess_excel(file_path)
        
        # Step 2: Create jellyfish input using all source columns with sample values
        # Structure detection happens later when we have jellyfish results
        jellyfish_input = self.column_mapper.create_jellyfish_input(
            source_df.columns.tolist(), self.target_schema, source_df, 
            sample_count=self.config.sample_values_count
        )
        
        return jellyfish_input
    
    def _extract_project_name(self, file_path: str) -> str:
        """Extract project name from file path"""
        filename = os.path.basename(file_path)
        # Remove .xlsx extension and _ITT suffix
        project_name = filename.replace('.xlsx', '')
        return project_name
    
    def process_multiple_files(self, file_paths: List[str], 
                             column_mappings: Optional[List[ColumnMapping]] = None) -> pd.DataFrame:
        """Process multiple files and combine results"""
        all_results = []
        
        for file_path in file_paths:
            try:
                result = self.process_single_file(file_path, column_mappings)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()


# Example usage and schema definitions
def get_target_schema() -> List[SchemaDefinition]:
    """Define the target schema based on your merged CSV structure"""
    return [
        SchemaDefinition("comment_id", "Unique identifier for the comment"),
        SchemaDefinition("section", "Document section being referenced"),
        SchemaDefinition("topic", "Topic or theme of the comment"),
        SchemaDefinition("round", "Review round number"),
        SchemaDefinition("comment_type", "Type of comment (e.g., Information Requirement)"),
        SchemaDefinition("comment_text", "Full text of the comment"),
        SchemaDefinition("response_text", "Proponent's response to the comment"),
        SchemaDefinition("author", "Author of the comment"),
        SchemaDefinition("date_received", "Date comment was received"),
        SchemaDefinition("date_responded", "Date response was sent"),
        SchemaDefinition("status", "Current status of the comment"),
        SchemaDefinition("agency", "Agency (e.g., EMLI, First Nations)")
    ]


def example_usage():
    """Example of how to use the pipeline"""
    # Define target schema
    target_schema = get_target_schema()
    
    # Initialize pipeline
    pipeline = DataMappingPipeline(target_schema)
    
    # Process a single file (this will return jellyfish input for server processing)
    file_path = "data/example_project_ITT.xlsx"
    
    # Step 1: Get jellyfish input
    jellyfish_input = pipeline.process_single_file(file_path)
    print("Jellyfish input ready for server processing")
    print(jellyfish_input.head())
    
    # Step 2: After getting jellyfish results from server, create mappings
    # (This is a placeholder - you'll replace with actual jellyfish results)
    example_mappings = [
        ColumnMapping("ID", "comment_id", "Yes"),
        ColumnMapping("Comment", "comment_text", "Yes"),
        ColumnMapping("Response", "response_text", "Yes"),
        ColumnMapping("Date Received", "date_received", "Yes"),
        ColumnMapping("Source Author", "author", "Yes"),
    ]
    
    # Step 3: Process with mappings
    result_df = pipeline.process_single_file(file_path, example_mappings)
    print("\nProcessed result:")
    print(result_df.head())


if __name__ == "__main__":
    example_usage()



