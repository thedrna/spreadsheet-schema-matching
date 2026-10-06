"""
Configuration for Data Mapping Pipeline
======================================

This file contains configuration settings and customization options
for the data mapping pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable
import re


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    
    # Excel preprocessing settings
    max_header_search_rows: int = 20
    min_header_indicators: int = 3
    min_header_cells_with_values: int = 5  # Minimum number of non-empty cells required for header row
    
    # Header quality scoring parameters
    header_length_penalty_divisor: float = 50.0  # Divisor for average cell length penalty calculation
    header_max_length_penalty: float = 3.0  # Maximum penalty for very long cells
    
    skip_sheets_with_keywords: List[str] = None
    prefer_sheets_with_keywords: List[str] = None
    
    # Column mapping settings
    jellyfish_confidence_threshold: str = "Yes"  # Only accept "Yes" mappings
    allow_multiple_mappings: bool = False  # Allow one source to map to multiple targets
    sample_values_count: int = 3  # Number of sample values to include for each source column
    
    # Value transformation settings
    date_formats: List[str] = None
    agency_abbreviation_max_length: int = 4
    empty_value_placeholders: List[str] = None
    
    def __post_init__(self):
        if self.skip_sheets_with_keywords is None:
            self.skip_sheets_with_keywords = ['summary', 'metadata', 'legend', 'cover', 'readme']
        
        if self.prefer_sheets_with_keywords is None:
            self.prefer_sheets_with_keywords = ['comment', 'data', 'main', 'table', 'itt']
        
        if self.date_formats is None:
            self.date_formats = [
                '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', 
                '%Y/%m/%d', '%d-%m-%Y', '%Y%m%d'
            ]
            
        if self.empty_value_placeholders is None:
            self.empty_value_placeholders = ["-", "x", "X", "nan", "NaN", "NAN", "~", "", " "]


# Custom transformation rules
class CustomTransformationRules:
    """Custom transformation rules that can be modified by users"""
    @staticmethod
    def agency_name_mapping() -> Dict[str, str]:
        """Mapping from full agency names to abbreviations"""
        return {
            # Federal Agencies
            'Environment and Climate Change Canada': 'ECCC',
            'Fisheries and Oceans Canada': 'DFO',
            'Indigenous Services Canada': 'ISC',
            'Transport Canada': 'TC',
            'Health Canada': 'HC',
            'Natural Resources Canada': 'NRCAN',
            'Parks Canada': 'PC',
            'Impact Assessment Agency of Canada': 'IAAC',
            
            # Provincial Agencies (BC)
            'Energy, Mines and Low Carbon Innovation': 'EMLI',
            'Ministry of Energy Mines and Low Carbon Innovation': 'EMLI',
            'Ministry of Energy, Mines and Low Carbon Innovation': 'EMLI',
            'Environmental Assessment Office': 'EAO',
            'Ministry of Environment and Climate Change Strategy': 'ENV',
            'Ministry of Forests, Lands, Natural Resource Operations and Rural Development': 'FOR',
            'Ministry of Agriculture': 'AGRI',
            'Ministry of Transportation and Infrastructure': 'MOTI',
            'British Columbia Oil and Gas Commission': 'BCOGC',
            'BC Oil and Gas Commission': 'BCOGC',
            
            # First Nations
            # NOTE: the specific Nations that appeared in the (private) dataset
            # have been replaced with placeholder entries for this public copy.
            'Example First Nation A': 'EFNA',
            'Example First Nation B': 'EFNB',
            'Example National Government C': 'ENGC',
            'Example Band D': 'EBD',
            'First Nations Health Authority': 'FNHA',
            'First Nations': 'FN',
            
            # Other Organizations
            'Fraser Basin Council': 'FBC',
            'Canadian Environmental Assessment Agency': 'CEAA',
        }
    
    @staticmethod
    def comment_type_mapping() -> Dict[str, str]:
        """Mapping for standardizing comment types"""
        return {
            'information requirement': 'Information Requirement',
            'info req': 'Information Requirement',
            'information request': 'Information Requirement',
            'ir': 'Information Requirement',
            'general observation': 'General Observation',
            'general comment': 'General Comment',
            'comment': 'General Comment',
            'clarification': 'Clarification',
            'recommendation': 'Recommendation',
            'concern': 'Concern',
            'question': 'Question',
            'follow-up': 'Follow-up',
            'technical comment': 'Technical Comment',
            'editorial': 'Editorial'
        }
    
    @staticmethod
    def status_mapping() -> Dict[str, str]:
        """Mapping for standardizing status values"""
        return {
            'open': 'Open',
            'pending': 'Open',
            'active': 'Open',
            'in progress': 'In Progress',
            'ongoing': 'In Progress',
            'under review': 'In Progress',
            'closed': 'Closed',
            'resolved': 'Closed',
            'complete': 'Closed',
            'completed': 'Closed',
            'satisfied': 'Closed',
            'addressed': 'Closed'
        }
    
    @staticmethod
    def round_extraction_patterns() -> List[str]:
        """Regex patterns for extracting round numbers"""
        return [
            r'round\s*(\d+)',
            r'r(\d+)',
            r'phase\s*(\d+)',
            r'p(\d+)',
            r'screening\s*r?(\d+)',
            r'review\s*(\d+)',
            r'\b(\d+)\b'  # Fallback: any standalone number
        ]
    
    @staticmethod
    def id_splitting_patterns() -> List[str]:
        """Patterns for splitting compound IDs like EMLI-001
        
        Patterns are ordered from most specific to most general.
        Each pattern captures: (agency_code, rest_of_id)
        The rest_of_id can contain letters, numbers, dots, hyphens, etc.
        """
        return [
            # Patterns with separators (hyphen, dot, underscore, space)
            r'^([A-Z]+)-(.+)$',              # AGY-001, AGY-R1.01, ABC-R1.02, AB-CD-12
            r'^([A-Z]+)\.([A-Z]+.*)$',      # ABC.DEFG-001a(i), ABC.DEFG-002
            r'^([A-Z]+)_(.+)$',              # AGY_001, AGY_R1.01, AGY_SCR1.01
            r'^([A-Z]+)\s+(.+)$',            # AGY 001, AGY R1.01
            # Patterns without separators (but still need 2+ letter agency code)
            r'^([A-Z]{2,})(\d+)$',           # AGY001 (no separator, numeric ID only)
        ]


# Column header detection patterns
HEADER_DETECTION_PATTERNS = {
    'id': [
        'id', 'identifier', 'comment id', 'comment_id', 'ref', 'reference',
        'number', 'no', '#', 'item'
    ],
    'comment': [
        'comment', 'comments', 'text', 'description', 'content', 'detail',
        'observation', 'requirement', 'request'
    ],
    'response': [
        'response', 'reply', 'answer', 'proponent response', 'proponent reply',
        'company response', 'resolution'
    ],
    'author': [
        'author', 'reviewer', 'source', 'from', 'submitted by', 'name',
        'source author', 'reviewer name'
    ],
    'date_received': [
        'date received', 'received', 'date', 'submission date', 'submit date',
        'received date', 'date submitted'
    ],
    'date_responded': [
        'date responded', 'response date', 'reply date', 'date sent',
        'response sent', 'date response sent'
    ],
    'status': [
        'status', 'state', 'condition', 'resolution status', 'comment status'
    ],
    'section': [
        'section', 'chapter', 'part', 'reference section', 'document section'
    ],
    'topic': [
        'topic', 'subject', 'theme', 'category', 'topic label', 'subject area'
    ],
    'round': [
        'round', 'phase', 'review round', 'review phase', 'screening round'
    ],
    'type': [
        'type', 'comment type', 'category', 'classification', 'kind'
    ],
    'agency': [
        'agency', 'organization', 'dept', 'department', 'ministry',
        'reviewer agency', 'source agency'
    ]
}


# Data quality rules
DATA_QUALITY_RULES = {
    'required_columns': [
        'comment_id', 'comment_text', 'project'
    ],
    'recommended_columns': [
        'author', 'date_received', 'response_text', 'status', 'agency'
    ],
    'date_validation': {
        'min_year': 2000,
        'max_year': 2030
    },
    'text_validation': {
        'min_comment_length': 10,
        'max_comment_length': 10000
    }
}


# File naming conventions
FILE_NAMING = {
    'processed_suffix': '_processed',
    'jellyfish_input': 'jellyfish_input.csv',
    'jellyfish_results': 'jellyfish_results.csv',
    'combined_output': 'combined_processed_data.csv',
    'summary_report': 'processing_summary.txt',
    'error_log': 'processing_errors.log'
}


def get_default_config() -> ProcessingConfig:
    """Get default processing configuration"""
    return ProcessingConfig()


def validate_config(config: ProcessingConfig) -> bool:
    """Validate configuration settings"""
    if config.max_header_search_rows < 1:
        raise ValueError("max_header_search_rows must be at least 1")
    
    if config.min_header_indicators < 1:
        raise ValueError("min_header_indicators must be at least 1")
    
    if config.agency_abbreviation_max_length < 2:
        raise ValueError("agency_abbreviation_max_length must be at least 2")
    
    return True


# Custom column mapping rules (for when jellyfish is not available)
FALLBACK_COLUMN_MAPPINGS = {
    'ID': 'comment_id',
    'Comment ID': 'comment_id',
    'Comment': 'comment_text',
    'Comments': 'comment_text',
    'Response': 'response_text',
    'Proponent Response': 'response_text',
    'Author': 'author',
    'Source Author': 'author',
    'Reviewer': 'author',
    'Date Received': 'date_received',
    'Date': 'date_received',
    'Date Response Sent': 'date_responded',
    'Response Date': 'date_responded',
    'Status': 'status',
    'Comment Status': 'status',
    'Section': 'section',
    'Reference Section': 'section',
    'Topic': 'topic',
    'Subject': 'topic',
    'Round': 'round',
    'Phase': 'round',
    'Review Round': 'round',
    'Type': 'comment_type',
    'Comment Type': 'comment_type',
    'Agency': 'agency',
    'Reviewer Agency': 'agency'
}
