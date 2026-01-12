#!/bin/bash
#
# Evaluation Monitor Script
#
# Monitors running LLM evaluations and displays real-time progress.
#
# Usage:
#   ./scripts/experiments/monitor_eval.sh [OPTIONS]
#
# Examples:
#   # Monitor default database
#   ./scripts/experiments/monitor_eval.sh
#
#   # Monitor specific database
#   ./scripts/experiments/monitor_eval.sh --db outputs/experiments/2026-01-12/exp1/results.db
#
#   # Continuous monitoring (refresh every 10s)
#   ./scripts/experiments/monitor_eval.sh --watch
#
# Author: Claude Code
# Created: 2026-01-12
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default database path
DB_PATH="${PROJECT_ROOT}/outputs/results.db"
WATCH=false
INTERVAL=10

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --db)
            DB_PATH="$2"
            shift 2
            ;;
        --watch|-w)
            WATCH=true
            shift
            ;;
        --interval|-i)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--db PATH] [--watch] [--interval N]"
            echo ""
            echo "Options:"
            echo "  --db PATH      Database path (default: outputs/results.db)"
            echo "  --watch, -w    Continuous monitoring"
            echo "  --interval N   Refresh interval in seconds (default: 10)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# =============================================================================
# Monitor Functions
# =============================================================================

check_db() {
    if [[ ! -f "$DB_PATH" ]]; then
        echo "Database not found: $DB_PATH"
        exit 1
    fi
}

show_progress() {
    clear
    echo "============================================================"
    echo "LLM Evaluation Progress Monitor"
    echo "Database: ${DB_PATH}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo ""

    # Summary table
    echo "Model                    | Completed | Success | RR%    | Status"
    echo "-------------------------|-----------|---------|--------|--------"

    sqlite3 -separator '|' "$DB_PATH" "
        SELECT
            model_name,
            COUNT(*) as n,
            SUM(success) as ok,
            ROUND(100.0*SUM(success)/COUNT(*), 1) as rr
        FROM evaluation_results
        GROUP BY model_name
        ORDER BY n DESC
    " | while IFS='|' read -r model n ok rr; do
        # Determine status
        if [[ "$n" -ge 200 ]]; then
            status="DONE"
        else
            status="Running"
        fi
        printf "%-24s | %9s | %7s | %5s%% | %s\n" "$model" "$n" "$ok" "$rr" "$status"
    done

    echo ""
    echo "============================================================"

    # Total stats
    total=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM evaluation_results")
    models=$(sqlite3 "$DB_PATH" "SELECT COUNT(DISTINCT model_name) FROM evaluation_results")
    echo "Total: ${total} results across ${models} models"

    # Check for running processes
    running=$(pgrep -f "evaluate_llm.py" | wc -l | tr -d ' ')
    echo "Running processes: ${running}"

    if [[ "$WATCH" == "true" ]]; then
        echo ""
        echo "Refreshing every ${INTERVAL}s... (Ctrl+C to stop)"
    fi
}

# =============================================================================
# Main
# =============================================================================

check_db

if [[ "$WATCH" == "true" ]]; then
    while true; do
        show_progress
        sleep "$INTERVAL"
    done
else
    show_progress
fi
