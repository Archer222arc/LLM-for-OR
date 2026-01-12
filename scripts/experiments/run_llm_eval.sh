#!/bin/bash
#
# LLM Evaluation Runner Script
#
# This script provides a structured way to run LLM evaluations with:
# - Automatic experiment directory creation
# - Configuration logging for reproducibility
# - Support for parallel model evaluation
# - SQLite-based result storage
#
# Usage:
#   ./scripts/experiments/run_llm_eval.sh [OPTIONS]
#
# Examples:
#   # Run single model evaluation
#   ./scripts/experiments/run_llm_eval.sh --model gpt-5.2-chat --samples 200
#
#   # Run all models in parallel
#   ./scripts/experiments/run_llm_eval.sh --all --samples 200 --parallel
#
#   # Resume interrupted evaluation
#   ./scripts/experiments/run_llm_eval.sh --model o4-mini --samples 500 --resume
#
# Author: Claude Code
# Created: 2026-01-12
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default values
SAMPLES=200
MAX_STEPS=15
WORKERS=4
PARALLEL=false
RESUME=false
ALL_MODELS=false
DRY_RUN=false

# Model definitions with recommended worker counts
declare -A MODEL_WORKERS=(
    ["o4-mini"]=4
    ["o1"]=4
    ["gpt-5.2-chat"]=4
    ["gpt-5-mini"]=4
    ["gpt-4.1-mini"]=4
    ["gpt-5-nano"]=4
    ["Kimi-K2-Thinking"]=2    # Rate limit sensitive
    ["Llama-3.3-70B-Instruct"]=2  # Rate limit sensitive
)

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/evaluation/evaluate_llm.py"

# =============================================================================
# Helper Functions
# =============================================================================

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --model MODEL       Single model to evaluate
    --all               Evaluate all 8 models
    --samples N         Number of samples (default: 200)
    --max-steps N       Max steps per episode (default: 15)
    --workers N         Number of parallel workers (default: 4, or model-specific)
    --parallel          Run models in parallel (with --all)
    --resume            Resume from existing results
    --exp-name NAME     Experiment name (default: auto-generated)
    --dry-run           Print commands without executing
    -h, --help          Show this help message

Examples:
    $(basename "$0") --model gpt-5.2-chat --samples 200
    $(basename "$0") --all --samples 200 --parallel
    $(basename "$0") --model o4-mini --samples 500 --resume

EOF
    exit 0
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# =============================================================================
# Argument Parsing
# =============================================================================

MODEL=""
EXP_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --all)
            ALL_MODELS=true
            shift
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate arguments
if [[ -z "$MODEL" ]] && [[ "$ALL_MODELS" != "true" ]]; then
    error "Must specify --model or --all"
fi

# =============================================================================
# Setup Experiment Directory
# =============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
DATE_DIR=$(date '+%Y-%m-%d')

if [[ -z "$EXP_NAME" ]]; then
    if [[ "$ALL_MODELS" == "true" ]]; then
        EXP_NAME="all_models_${SAMPLES}samples"
    else
        EXP_NAME="${MODEL}_${SAMPLES}samples"
    fi
fi

# Create structured output directory
OUTPUT_DIR="${PROJECT_ROOT}/outputs/experiments/${DATE_DIR}/${EXP_NAME}"
DB_PATH="${OUTPUT_DIR}/results.db"
LOG_DIR="${OUTPUT_DIR}/logs"

if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

    # Save experiment configuration
    cat > "${OUTPUT_DIR}/config.yaml" << EOF
# Experiment Configuration
# Generated: $(date '+%Y-%m-%d %H:%M:%S')

experiment:
  name: ${EXP_NAME}
  timestamp: ${TIMESTAMP}

evaluation:
  samples: ${SAMPLES}
  max_steps: ${MAX_STEPS}
  default_workers: ${WORKERS}
  resume: ${RESUME}

models:
$(if [[ "$ALL_MODELS" == "true" ]]; then
    for m in "${!MODEL_WORKERS[@]}"; do
        echo "  - name: $m"
        echo "    workers: ${MODEL_WORKERS[$m]}"
    done
else
    echo "  - name: ${MODEL}"
    echo "    workers: ${MODEL_WORKERS[$MODEL]:-$WORKERS}"
fi)

paths:
  output_dir: ${OUTPUT_DIR}
  database: ${DB_PATH}
  logs: ${LOG_DIR}
EOF

    # Save git hash for reproducibility
    cd "$PROJECT_ROOT"
    git rev-parse HEAD > "${OUTPUT_DIR}/git_hash.txt" 2>/dev/null || echo "not a git repo" > "${OUTPUT_DIR}/git_hash.txt"

    log "Experiment directory: ${OUTPUT_DIR}"
fi

# =============================================================================
# Run Evaluation
# =============================================================================

run_model() {
    local model=$1
    local workers=${MODEL_WORKERS[$model]:-$WORKERS}
    local log_file="${LOG_DIR}/${model}.log"

    local cmd="python ${EVAL_SCRIPT} \
        --model ${model} \
        --limit ${SAMPLES} \
        --max_steps ${MAX_STEPS} \
        --workers ${workers} \
        --db ${DB_PATH}"

    if [[ "$RESUME" == "true" ]]; then
        cmd="${cmd} --resume"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] ${cmd}"
    else
        log "Starting: ${model} (workers=${workers})"
        if [[ "$PARALLEL" == "true" ]]; then
            nohup bash -c "${cmd}" > "${log_file}" 2>&1 &
            echo $! > "${LOG_DIR}/${model}.pid"
            log "  PID: $! -> ${log_file}"
        else
            ${cmd} 2>&1 | tee "${log_file}"
        fi
    fi
}

if [[ "$ALL_MODELS" == "true" ]]; then
    log "Running all models (parallel=${PARALLEL})"
    for model in "${!MODEL_WORKERS[@]}"; do
        run_model "$model"
    done

    if [[ "$PARALLEL" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
        log "All models started in background"
        log "Monitor progress: sqlite3 ${DB_PATH} \"SELECT model_name, COUNT(*) FROM evaluation_results GROUP BY model_name\""
        log "Check logs: tail -f ${LOG_DIR}/*.log"
    fi
else
    run_model "$MODEL"
fi

# =============================================================================
# Post-run Summary
# =============================================================================

if [[ "$DRY_RUN" != "true" ]] && [[ "$PARALLEL" != "true" ]]; then
    log "Evaluation complete"
    log "Results: ${DB_PATH}"

    # Export to JSON
    JSON_PATH="${OUTPUT_DIR}/results.json"
    python "${EVAL_SCRIPT}" --export-json "${JSON_PATH}" --db "${DB_PATH}"
    log "Exported: ${JSON_PATH}"
fi
