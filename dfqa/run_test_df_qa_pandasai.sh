#!/bin/bash
#SBATCH --job-name=test_df_qa_pandasai
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=END,FAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="deepseek-coder:6.7b"
# MODEL_NAME="qwen2.5-coder:7b"
PROJECT_DIR="/path/to/dfqa"
PYTHON_SCRIPT="test_df_qa_pandasai.py"
VENV_PATH="$PROJECT_DIR/venv"
OLLAMA_HOST="http://127.0.0.1:11500"
INPUT_FILE="combined_processed_data_test_1dsc.csv"

cd "$PROJECT_DIR"

export OLLAMA_MODEL="$MODEL_NAME"
export OLLAMA_HOST="$OLLAMA_HOST"
export DF_QA_INPUT_CSV="$INPUT_FILE"
export OLLAMA_HOST_API="$OLLAMA_HOST"

TIMESTAMP=$(date +"%Y%m%d_%H%M")
RUN_DIR="$PROJECT_DIR/outputs/test_df_qa_pandasai/run_${TIMESTAMP}_${MODEL_NAME//:/_}"
mkdir -p "$RUN_DIR"
export RUN_DIR

exec 1>"$RUN_DIR/test_df_qa_${SLURM_JOB_ID}.out"
exec 2>"$RUN_DIR/test_df_qa_${SLURM_JOB_ID}.err"

echo "=== Project Setup ==="
echo "Project directory: $(pwd)"
echo "Python script: $PYTHON_SCRIPT"
echo "Input file: $INPUT_FILE"
echo ""
echo "=== Configuration ==="
echo "Model: $MODEL_NAME"
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

echo "=== Setting up Scratch Directory for Model Cache ==="
export HF_HOME=/path/to/scratch/huggingface_cache
export TRANSFORMERS_CACHE=/path/to/scratch/huggingface_cache
export TORCH_HOME=/path/to/scratch/torch_cache
echo "✓ HF_HOME set to: $HF_HOME"
echo "✓ TRANSFORMERS_CACHE set to: $TRANSFORMERS_CACHE"
echo "✓ TORCH_HOME set to: $TORCH_HOME"
echo ""
mkdir -p /path/to/scratch/huggingface_cache
mkdir -p /path/to/scratch/torch_cache
echo "✓ Cache and output directories created"
echo ""


# Virtual Environment Management
echo "=== Virtual Environment Setup ==="
if [ -d "$VENV_PATH" ]; then
    echo "✓ Using existing virtual environment at: $VENV_PATH"
else
    echo "Creating new virtual environment at: $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    if [ $? -eq 0 ]; then
        echo "✓ Virtual environment created successfully"
    else
        echo "✗ Failed to create virtual environment"
        exit 1
    fi
fi
echo ""


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
# Install numpy first to avoid binary incompatibility issues
"$VENV_PATH/bin/pip" install numpy==1.26.4
# Then install pandas and other packages (PyYAML not yaml!)
"$VENV_PATH/bin/pip" install pandas requests PyYAML pandasai

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
        
        # Check if model exists and pull if necessary
        if [ -n "$MODEL_NAME" ]; then
            echo ""
            echo "=== Checking Model Availability ==="
            echo "Checking if model '$MODEL_NAME' is available..."
            
            # List available models and check if our model exists
            if ollama list | grep -q "^$MODEL_NAME"; then
                echo "✓ Model '$MODEL_NAME' is already available"
            else
                echo "⚠ Model '$MODEL_NAME' not found locally"
                echo "Pulling model '$MODEL_NAME'..."
                if ollama pull "$MODEL_NAME"; then
                    echo "✓ Model '$MODEL_NAME' pulled successfully"
                else
                    echo "✗ Failed to pull model '$MODEL_NAME'"
                    echo "⚠ Continuing anyway - script may fail if model is required"
                fi
            fi
        else
            echo "⚠ MODEL_NAME is empty - skipping model check"
        fi
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