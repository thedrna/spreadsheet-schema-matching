import pandas as pd
import json
import requests
import time
import os
import pathlib

# ===== CONFIGURATION =====
# Configuration is read from environment variables set by the bash script
# If running standalone, default values will be used

CONFIG = {
    "model_name": os.environ.get("OLLAMA_MODEL_NAME", "mistral"),
    "test_mode": os.environ.get("OLLAMA_TEST_MODE", "False") == "True",
    "test_random": os.environ.get("OLLAMA_TEST_RANDOM", "False") == "True",
    "test_sample_count": int(os.environ.get("OLLAMA_TEST_SAMPLE_COUNT", "5")),
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11500"),
    "input_file": os.environ.get("OLLAMA_INPUT_CSV", "source_target_with_random_samples_columns.csv"),
    "test_data": os.environ.get("OLLAMA_TEST_DATA", "False") == "True",
    "num_source_descriptions": int(os.environ.get("OLLAMA_NUM_SOURCE_DESCRIPTIONS", "3")),
}


# Use RUN_DIR if provided by bash script, otherwise create a new one
if "RUN_DIR" in os.environ:
    OUTPUT_DIR = os.environ["RUN_DIR"]
else:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"outputs/ollama_desc_generation/run_{ts}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OLLAMA_BASE_URL = CONFIG["ollama_host"]
OLLAMA_MODEL = CONFIG["model_name"]
OUTPUT_CSV = f"{OUTPUT_DIR}/ollama_desc_results_{OLLAMA_MODEL}.csv"
INPUT_CSV = CONFIG["input_file"]
TEST_DATA = CONFIG["test_data"]



class OllamaDescriptionGenerator:
    """Generate column descriptions using Ollama LLM"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, batch_size=5):
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.session.keep_alive = True
        
    def get_model_name(self):
        """Return the name of the current model"""
        return self.model
        
    def test_connection(self):
        """Test if Ollama is available"""
        try:
            print(f"Testing connection to Ollama at {self.base_url}")
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✓ Connection successful")
                print(f"Available models: {response.json()}")
                return True
            else:
                print(f"✗ Server returned status code: {response.status_code}")
                print(f"Response text: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Connection error: {str(e)}")
            return False
    
    def _clean_sample_values(self, sample_values):
        """Clean and format sample values for prompt"""
        cleaned_samples = []
        for v in sample_values:
            if str(v).strip() and str(v) != 'nan' and pd.notna(v):
                sample = str(v).strip()
                if len(sample) > 100:
                    sample = sample[:100] + "..."
                cleaned_samples.append(sample)
        return ", ".join(cleaned_samples)
    
    def _build_prompt(self, column_name, samples_str=None, existing_desc=None):
        """Build the prompt for description generation
        
        Args:
            column_name: Name of the column
            samples_str: Sample values string (for source columns)
            existing_desc: Existing description (for target columns)
        """
        # Determine which type of prompt to build
        if samples_str is not None:
            # Source column prompt with sample values
            context_line = f"Sample values: {samples_str}"
        elif existing_desc is not None:
            # Target column prompt with existing description
            context_line = f"Current description: {existing_desc}"
        else:
            # Default fallback
            context_line = "Sample values: (none provided)"
        
        return f"""Task: Write one concise, technical sentence describing this database column.

**Table Context:**
The table contains back-and-forth comments from various reviewers to the proponent for a major mine project.

**Inputs:**
1.  Column Name: {column_name}
2.  {context_line}

**Guidelines:**
1.  **Start with:** "This column"
2.  **Function:** Describe the column's specific purpose *based on the Table Context*. Infer the entity (e.g., 'comment', 'reviewer', 'response') from the `Column Name` and `Context`.
3.  **Format:** If `Context` contains sample values, infer the data type (e.g., "numeric identifier," "timestamp," "text").
4.  **Length:** Strictly 25 words or less.
5.  **Style:** Be precise and technical. Do not add justifications (e.g., "is important for," "facilitating traceability"). Focus on *what it is*.
6.  **End with:** A period.

**Examples (Source Column with Samples):**
* **Inputs:**
    * Column Name: `ID #`
    * Context: `Sample values: 1, 2, 3`
* **Target Response:** This column is a unique numeric identifier for individual comments in the permit review process.

**Example (Target Column with Description):**
* **Inputs:**
    * Column Name: `Comment_Text`
    * Context: `Current description: The reviewer's feedback`
* **Target Response:** This column contains the full text of the comment submitted by a reviewer.

Response: This column"""
    
    def _clean_response(self, response_text):
        """Clean up the generated description"""
        description = response_text.strip()
        description = description.replace("Complete this exactly: This column", "").strip()
        return description
    
    def generate_description(self, column_name, sample_values=None, existing_desc=None):
        """Generate description for a column using its name and sample values or existing description
        
        Args:
            column_name: Name of the column
            sample_values: List of sample values (for source columns)
            existing_desc: Existing description (for target columns)
        """
        samples_str = self._clean_sample_values(sample_values) if sample_values else None
        prompt = self._build_prompt(column_name, samples_str=samples_str, existing_desc=existing_desc)

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 2048,
                        "num_predict": 50,
                        "stop": ["\n"],
                        "num_gpu": 999  # Load all layers to GPU
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                description = self._clean_response(result.get("response", ""))
                return description if description else f"Column containing {column_name} data"
            else:
                print(f"Ollama API error: {response.status_code}")
                return f"Column containing {column_name} data"
                
        except Exception as e:
            print(f"Error generating description for {column_name}: {e}")
            return f"Column containing {column_name} data"


def _load_description_target(target_file):
    """Load cached target descriptions from file"""
    try:
        with open(target_file, 'r') as f:
            descriptions = json.load(f)
        print(f"✓ Loaded {len(descriptions)} cached target descriptions")
        return descriptions
    except FileNotFoundError:
        print("No existing target description cache found. Creating new target description cache.")
        return {}


def _save_description_target(target_file, descriptions):
    """Save target descriptions to cache file"""
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, 'w') as f:
        json.dump(descriptions, f, indent=2)


def _create_samples_tuple(row, sample_columns):
    """Create a tuple of non-empty sample values"""
    return tuple(
        str(x) for x in row[sample_columns] 
        if pd.notna(x) and str(x).strip() and str(x) != 'nan'
    )


def _generate_source_descriptions(df, description_generator, sample_columns, num_descriptions=3):
    """Generate multiple descriptions for source columns based on their samples"""
    df['samples_tuple'] = df[sample_columns].apply(
        lambda row: _create_samples_tuple(row, sample_columns), axis=1
    )
    
    unique_source_samples = df[['source_column', 'samples_tuple']].drop_duplicates()
    print(f"Found {len(unique_source_samples)} unique source column and samples combinations")
    print(f"Generating {num_descriptions} description variants for each source column")
    
    source_descriptions = {}
    for _, row in unique_source_samples.iterrows():
        source_name = row['source_column']
        samples = list(row['samples_tuple'])
        key = (source_name, row['samples_tuple'])
        
        print(f"Processing source column: {source_name} with its specific samples")
        descriptions = []
        for i in range(num_descriptions):
            description = description_generator.generate_description(
                source_name, sample_values=samples
            )
            descriptions.append(description)
            print(f"  Generated description {i+1}/{num_descriptions}")
        
        source_descriptions[key] = descriptions
    
    return source_descriptions


def _generate_target_descriptions(df, description_generator, target_file):
    """Generate or retrieve descriptions for target columns"""
    target_descriptions = _load_description_target(target_file)
    unique_target_cols = df['target_column'].unique()
    print(f"Found {len(unique_target_cols)} unique target columns")
    
    for target_col in unique_target_cols:
        if target_col not in target_descriptions:
            print(f"Generating description for target column: {target_col}")
            
            existing_desc = df[df['target_column'] == target_col]['target_description'].iloc[0]
            
            description = description_generator.generate_description(
                target_col, existing_desc=existing_desc
            )
            target_descriptions[target_col] = description

            _save_description_target(target_file, target_descriptions)
            time.sleep(0.1)
    
    return target_descriptions


def generate_missing_descriptions(df, description_generator, sample_columns):
    """Generate comprehensive descriptions for both source and target columns"""
    print("Checking for missing descriptions...")

    target_file = f"{OUTPUT_DIR}/generated_target_descriptions.json"
    num_source_descriptions = CONFIG["num_source_descriptions"]

    # Generate source descriptions (multiple per source)
    source_descriptions = _generate_source_descriptions(
        df, description_generator, sample_columns, num_descriptions=num_source_descriptions
    )
    
    # Apply source descriptions to dataframe (one column per description variant)
    print("Applying descriptions to all rows...")
    for i in range(num_source_descriptions):
        col_name = f'source_description_{i+1}'
        df[col_name] = df.apply(
            lambda row: source_descriptions[(row['source_column'], row['samples_tuple'])][i], 
            axis=1
        )
    
    # Generate target descriptions (single description per target)
    target_descriptions = _generate_target_descriptions(df, description_generator, target_file)
    
    # Apply target descriptions to dataframe
    df['target_description'] = df['target_column'].map(target_descriptions)
    
    # Save the DataFrame with generated descriptions
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved DataFrame with generated descriptions to {OUTPUT_CSV}")

    return df


def _load_and_filter_data(input_file):
    """Load CSV data and filter out unwanted rows"""
    full_df = pd.read_csv(input_file)
    full_df = full_df[full_df['target_column'] != 'project'].reset_index(drop=True)
    print(f"✓ Filtered out rows with target_column='project'. Remaining rows: {len(full_df)}")
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
    print(f"✓ Loaded {len(df)} rows from {input_file}")
    return df


def _detect_sample_columns(df):
    """Detect columns that contain sample data"""
    sample_columns = [col for col in df.columns if col.startswith('sample_')]
    print(f"✓ Detected {len(sample_columns)} sample columns: {sample_columns}")
    return sample_columns



def main():
    """Main processing function"""
    print("=== Ollama Description Generation ===")
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print()

    # Load input data
    print("Loading input data...")
    try:
        df = _load_and_filter_data(INPUT_CSV)
        sample_columns = _detect_sample_columns(df)
    except Exception as e:
        print(f"✗ Error loading input file: {e}")
        return

    # Initialize description generator
    description_generator = OllamaDescriptionGenerator()

    # Test Ollama connection
    print("\nTesting Ollama connection...")
    if description_generator.test_connection():
        print("✓ Ollama is available")
            
        df = generate_missing_descriptions(df, description_generator, sample_columns)
        
        print("✓ Description generation completed")
    else:
        print("⚠ Ollama not available - using existing descriptions only")
        print("  Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()