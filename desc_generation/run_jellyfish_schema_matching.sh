#!/bin/bash
#SBATCH --job-name=jellyfish_schema_matching
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Jellyfish Processing Script
# =============================================================================
# This script processes CSV input with column comparisons using:
# - Jellyfish model for semantic equivalence assessment
# =============================================================================

# Configure PyTorch CUDA memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# CONFIGURATION - Edit these settings for different runs
# =============================================================================
MODEL_ID="NECOUDBFM/Jellyfish-8B"  # Options: "NECOUDBFM/Jellyfish-7B" or "NECOUDBFM/Jellyfish-8B"
USE_COT="True"                      # Options: "True" or "False"
INCLUDE_SAMPLES="True"              # Options: "True" or "False"
TEST_DATA="False"                  # Options: "True" or "False" - If True, use test data file

# Test mode configuration
TEST_MODE="True"                    # Options: "True" or "False" - If True, process only a subset
TEST_RANDOM="False"                 # Options: "True" or "False" - If True, random sample; if False, head
TEST_SAMPLE_COUNT="100"               # Number of rows to process in test mode

# Filtering configuration
USE_FILTERING="False"                 # Options: "True" or "False" - Enable embedding-based pre-filtering
FILTER_TOP_K="5"                     # Keep top K most similar targets per source column
FILTER_THRESHOLD="0.25"              # Minimum similarity score to keep (0-1 range)
EMBEDDING_MODEL="all-MiniLM-L6-v2"   # Sentence transformer model for embeddings
DEDUPLICATE_PAIRS="True"             # Options: "True" or "False" - Process unique pairs only, propagate results back to all duplicates

# Export configuration as environment variables for Python script
export JELLYFISH_MODEL_ID="$MODEL_ID"
export JELLYFISH_USE_COT="$USE_COT"
export JELLYFISH_INCLUDE_SAMPLES="$INCLUDE_SAMPLES"
export JELLYFISH_TEST_DATA="$TEST_DATA"
export JELLYFISH_TEST_MODE="$TEST_MODE"
export JELLYFISH_TEST_RANDOM="$TEST_RANDOM"
export JELLYFISH_TEST_SAMPLE_COUNT="$TEST_SAMPLE_COUNT"
export JELLYFISH_USE_FILTERING="$USE_FILTERING"
export JELLYFISH_FILTER_TOP_K="$FILTER_TOP_K"
export JELLYFISH_FILTER_THRESHOLD="$FILTER_THRESHOLD"
export JELLYFISH_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export JELLYFISH_DEDUPLICATE_PAIRS="$DEDUPLICATE_PAIRS"

# Configuration Variables
PROJECT_DIR="/path/to/desc_generation"
PYTHON_SCRIPT="jellyfish_schema_matching.py"
VENV_PATH="$PROJECT_DIR/venv"
if [ "$TEST_DATA" = "True" ]; then
    INPUT_FILE="/path/to/desc_generation/outputs/ollama_desc_generation/run_<TIMESTAMP>_mistral_full__testdata_src3/ollama_desc_results_mistral.csv"
else
    INPUT_FILE="/path/to/desc_generation/outputs/ollama_desc_generation/run_<TIMESTAMP>_mistral_full__traindata_src3/ollama_desc_results_mistral.csv"
fi

export JELLYFISH_INPUT_CSV="$INPUT_FILE"

# Navigate to project directory
cd "$PROJECT_DIR"

# Extract short labels for directory name
MODEL_SIZE=$(echo "$MODEL_ID" | sed -E 's/.*Jellyfish-([0-9]+B).*/\1/')
COT_LABEL=$( [ "$USE_COT" = "True" ] && echo "CoT" || echo "noCoT" )
SAMPLES_LABEL=$( [ "$INCLUDE_SAMPLES" = "True" ] && echo "withSamples" || echo "noSamples" )
FILTER_LABEL=$( [ "$USE_FILTERING" = "True" ] && echo "filtered_top${FILTER_TOP_K}_th${FILTER_THRESHOLD}" || echo "noFilter" )
TEST_LABEL=$( [ "$TEST_MODE" = "True" ] && echo "test" || echo "full" )
SAMPLING_METHOD=$( [ "$TEST_MODE" = "True" ] && [ "$TEST_RANDOM" = "True" ] && echo "random" || echo "" )
SAMPLE_NUM=$( [ "$TEST_MODE" = "True" ] && echo "_${TEST_SAMPLE_COUNT}" || echo "" )
DATA_LABEL=$( [ "$TEST_DATA" = "True" ] && echo "testdata" || echo "traindata" )

# Create unique run directory with timestamp and config
TIMESTAMP=$(date +"%Y%m%d_%H%M")
RUN_DIR="$PROJECT_DIR/outputs/jellyfish_schema_matching/run_${TIMESTAMP}_${MODEL_SIZE}_${COT_LABEL}_${SAMPLES_LABEL}_${FILTER_LABEL}_${TEST_LABEL}${SAMPLING_METHOD}${SAMPLE_NUM}_${DATA_LABEL}"
mkdir -p "$RUN_DIR"

# Export RUN_DIR so Python script can use it
export RUN_DIR

# Redirect output and error to run directory
exec 1>"$RUN_DIR/jellyfish_schema_matching_${SLURM_JOB_ID}.out"
exec 2>"$RUN_DIR/jellyfish_schema_matching_${SLURM_JOB_ID}.err"

echo "=== Project Setup ==="
echo "Project directory: $(pwd)"
echo "Python script: $PYTHON_SCRIPT"
echo "Input file: $INPUT_FILE"
echo ""
echo "=== Configuration ==="
echo "Model: $MODEL_ID"
echo "Chain of Thought: $USE_COT"
echo "Include Source Samples: $INCLUDE_SAMPLES"
echo "Test Mode: $TEST_MODE"
if [ "$TEST_MODE" = "True" ]; then
    echo "  Test Random: $TEST_RANDOM"
    echo "  Test Sample Count: $TEST_SAMPLE_COUNT"
fi
echo "Use Filtering: $USE_FILTERING"
if [ "$USE_FILTERING" = "True" ]; then
    echo "  Embedding Model: $EMBEDDING_MODEL"
    echo "  Top K: $FILTER_TOP_K"
    echo "  Threshold: $FILTER_THRESHOLD"
fi
echo "Test Data: $TEST_DATA"
echo ""
echo "Run directory: $RUN_DIR"
echo "Timestamp: $(date)"
echo ""

# Verify input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "✗ Input file '$INPUT_FILE' not found!"
    exit 1
fi
echo "✓ Input file found: $INPUT_FILE ($(wc -l < $INPUT_FILE) lines)"

# Set environment variables for large model downloads using scratch directory
echo "=== Setting up Scratch Directory for Model Cache ==="
export HF_HOME=/path/to/scratch/huggingface_cache
export TRANSFORMERS_CACHE=/path/to/scratch/huggingface_cache
export TORCH_HOME=/path/to/scratch/torch_cache
echo "✓ HF_HOME set to: $HF_HOME"
echo "✓ TRANSFORMERS_CACHE set to: $TRANSFORMERS_CACHE"
echo "✓ TORCH_HOME set to: $TORCH_HOME"
echo ""

# Create cache directories if they don't exist
mkdir -p /path/to/scratch/huggingface_cache
mkdir -p /path/to/scratch/torch_cache
echo "✓ Cache and output directories created"
echo ""

echo "=== Environment Check ==="
echo "Current working directory: $(pwd)"
echo "Available Python: $(which python3) - $(python3 --version)"
echo "Available disk space in scratch: $(df -h /scratch | tail -1 | awk '{print $4}')"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "CPU cores allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo ""

# Virtual Environment Management
echo "=== Virtual Environment Setup ==="
# Activate Virtual Environment
echo "=== Activating Virtual Environment ==="
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Activated virtual environment: $VENV_PATH"
else
    echo "✗ Virtual environment '$VENV_PATH' not found!"
    exit 1
fi
echo "✓ Active Python: $(which python3) - $(python3 --version)"
echo ""

# Install required packages
echo "=== Installing Required Packages ==="
"$VENV_PATH/bin/pip" install --upgrade pip
"$VENV_PATH/bin/pip" install pandas requests transformers torch accelerate jellyfish datasets sentence-transformers scikit-learn

echo "=== Verifying Installation ==="

# Verify script exists
echo "=== Script Verification ==="
if [ -f "$PYTHON_SCRIPT" ]; then
    echo "✓ Script file '$PYTHON_SCRIPT' found"
    echo "Script size: $(wc -l < $PYTHON_SCRIPT) lines"
else
    echo "✗ Script file '$PYTHON_SCRIPT' not found!"
    echo "Available Python files in directory:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    exit 1
fi
echo ""



# Run Python script
echo "=== Running Jellyfish Schema Matching ==="
echo "Executing: python $PYTHON_SCRIPT"
echo "Started at: $(date)"
echo ""

echo "Running Python script..."
python "$PYTHON_SCRIPT"
SCRIPT_EXIT_CODE=$?

if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Job Complete ==="
    echo "Check the output files for results and any error messages."
    echo ""
    echo "All output files saved in: $RUN_DIR"
    echo "Output files:"
    ls -lh "$RUN_DIR"/* 2>/dev/null || echo "No output files found"
else
    echo "✗ Script failed with exit code: $SCRIPT_EXIT_CODE"
    exit $SCRIPT_EXIT_CODE
fi

echo ""
echo "=== Job Summary ==="
echo "Run directory: $RUN_DIR"
echo "Model cache location: $HF_HOME"
echo "Completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Done."
