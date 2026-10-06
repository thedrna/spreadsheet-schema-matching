#!/bin/bash
#SBATCH --job-name=ollama_desc_generation
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=END,FAIL

# Configure PyTorch CUDA memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# =============================================================================
# Ollama Description Generation Script
# =============================================================================
# This script processes CSV input and generate descriptions using:
# - Ollama for description generation (small LLM)
# =============================================================================

# Configuration Variables
PROJECT_DIR="/path/to/desc_generation"
PYTHON_SCRIPT="ollama_description_generation.py"
VENV_PATH="$PROJECT_DIR/venv"
# OLLAMA_HOST="http://<REMOTE_OLLAMA_HOST>:11500"
OLLAMA_HOST="http://127.0.0.1:11500"
# Respect a pre-set OLLAMA_HOST; fall back to localhost if nothing provided.
# : "${OLLAMA_HOST:=http://127.0.0.1:11500}"
# INPUT_FILE="source_target_columns.csv"
# INPUT_FILE="source_target_with_random_samples_columns.csv"
INPUT_FILE="testdata_source_target_with_random_samples_columns.csv"

# Navigate to project directory
cd "$PROJECT_DIR"

# Test mode configuration
MODEL_NAME="mistral"
TEST_MODE="False"                    # Options: "True" or "False" - If True, process only a subset
TEST_RANDOM="False"                 # Options: "True" or "False" - If True, random sample; if False, head
TEST_SAMPLE_COUNT="300"               # Number of rows to process in test mode
TEST_DATA="True"                  # Options: "True" or "False" - If True, use test data file
NUM_SOURCE_DESCRIPTIONS="3"         # Number of description variants to generate for each source column

export OLLAMA_MODEL_NAME="$MODEL_NAME"
export OLLAMA_TEST_MODE="$TEST_MODE"
export OLLAMA_TEST_RANDOM="$TEST_RANDOM"
export OLLAMA_TEST_SAMPLE_COUNT="$TEST_SAMPLE_COUNT"
export OLLAMA_TEST_DATA="$TEST_DATA"
export OLLAMA_INPUT_CSV="$INPUT_FILE"
export OLLAMA_NUM_SOURCE_DESCRIPTIONS="$NUM_SOURCE_DESCRIPTIONS"

MODEL_NAME_LABEL=$( echo "$MODEL_NAME" )
TEST_LABEL=$( [ "$TEST_MODE" = "True" ] && echo "test" || echo "full" )
SAMPLING_METHOD=$( [ "$TEST_MODE" = "True" ] && [ "$TEST_RANDOM" = "True" ] && echo "random" || echo "" )
SAMPLE_NUM=$( [ "$TEST_MODE" = "True" ] && echo "_${TEST_SAMPLE_COUNT}" || echo "" )
DATA_LABEL=$( [ "$TEST_DATA" = "True" ] && echo "testdata" || echo "traindata" )
SOURCE_DESC_NUM="src${NUM_SOURCE_DESCRIPTIONS}"

TIMESTAMP=$(date +"%Y%m%d_%H%M")
RUN_DIR="$PROJECT_DIR/outputs/ollama_desc_generation/run_${TIMESTAMP}_${MODEL_NAME_LABEL}_${TEST_LABEL}_${SAMPLING_METHOD}${SAMPLE_NUM}_${DATA_LABEL}_${SOURCE_DESC_NUM}"
mkdir -p "$RUN_DIR"

# Create outputs directory
# Export RUN_DIR so Python script can use it
export RUN_DIR

# Redirect output and error to run directory
exec 1>"$RUN_DIR/ollama_desc_generation_${SLURM_JOB_ID}.out"
exec 2>"$RUN_DIR/ollama_desc_generation_${SLURM_JOB_ID}.err"




echo "=== Project Setup ==="
echo "Project directory: $(pwd)"
echo "Python script: $PYTHON_SCRIPT"
echo "Input file: $INPUT_FILE"
echo ""
echo "=== Configuration ==="
echo "Model: $MODEL_NAME"
echo "Test Mode: $TEST_MODE"
if [ "$TEST_MODE" = "True" ]; then
    echo "  Test Random: $TEST_RANDOM"
    echo "  Test Sample Count: $TEST_SAMPLE_COUNT"
fi
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
"$VENV_PATH/bin/pip" install pandas requests transformers torch accelerate

# Start Ollama server on this compute node
echo "=== Starting Ollama Server ==="
if command -v ollama &> /dev/null; then
    echo "✓ Ollama command found"
    
    # Set Ollama environment variables
    export OLLAMA_HOST="$OLLAMA_HOST"
    export OLLAMA_DEBUG=1       # Enable debug logging
    
    # Start Ollama in the background
    echo "Starting Ollama server on $(hostname)..."
    ollama serve &
    OLLAMA_PID=$!
    echo "✓ Ollama server started with PID: $OLLAMA_PID"
    
    # Wait for Ollama to be ready (max 30 seconds)
    echo "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
            echo "✓ Ollama server is ready!"
            break
        fi
        echo "  Attempt $i/30: waiting..."
        sleep 1
    done
    
    # Final check
    if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
        echo "✓ Ollama server is responding"
    else
        echo "⚠ Ollama server may not be responding - continuing anyway"
    fi
else
    echo "⚠ Ollama not found - script will use existing descriptions only"
    OLLAMA_PID=""
fi
echo ""

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
echo "=== Running Description Generation ==="
echo "Executing: python $PYTHON_SCRIPT"
echo "Started at: $(date)"
echo ""

echo "Running Python script..."
OLLAMA_HOST="$OLLAMA_HOST" python3 "$PYTHON_SCRIPT"
SCRIPT_EXIT_CODE=$?

if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Job Complete ==="
    echo "Check the output files for results and any error messages."
    echo ""
    echo "Output files:"
    ls -lh "$RUN_DIR"/* 2>/dev/null || echo "No output files found"
else
    echo "✗ Script failed with exit code: $SCRIPT_EXIT_CODE"
fi

# Clean up: Stop Ollama server if we started it
if [ -n "$OLLAMA_PID" ]; then
    echo ""
    echo "=== Stopping Ollama Server ==="
    if kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "Stopping Ollama server (PID: $OLLAMA_PID)..."
        kill $OLLAMA_PID
        sleep 2
        # Force kill if still running
        if kill -0 $OLLAMA_PID 2>/dev/null; then
            kill -9 $OLLAMA_PID 2>/dev/null
        fi
        echo "✓ Ollama server stopped"
    else
        echo "Ollama server already stopped"
    fi
fi

echo ""
echo "=== Job Summary ==="
echo "Run directory: $RUN_DIR"
echo "Model cache location: $HF_HOME"
echo "Completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Done."

# Exit with the script's exit code
exit $SCRIPT_EXIT_CODE
