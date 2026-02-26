#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# 4-Hour Paper Trading Session with Auto-Tuning
# ──────────────────────────────────────────────────────────────────────
# Runs the live paper trader in background, then executes the auto-tuner
# every 30 minutes for 4 hours (8 tuning cycles).
#
# Usage: bash scripts/run_4h_session.sh
# ──────────────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON=".venv/bin/python"
DATA_DIR="$PROJECT_DIR/data"
LOG_DIR="$DATA_DIR/logs"
TUNER_INTERVAL=1800  # 30 minutes in seconds
TOTAL_DURATION=14400  # 4 hours in seconds
TUNER_CYCLES=$((TOTAL_DURATION / TUNER_INTERVAL))  # 8 cycles

mkdir -p "$DATA_DIR" "$LOG_DIR"

SESSION_ID=$(date +%Y%m%d_%H%M%S)
TRADER_LOG="$LOG_DIR/trader_${SESSION_ID}.log"
TUNER_SESSION_LOG="$LOG_DIR/tuner_session_${SESSION_ID}.log"

echo "============================================================"
echo "  4-HOUR PAPER TRADING SESSION"
echo "  Session ID: $SESSION_ID"
echo "  Started:    $(date)"
echo "  Duration:   4 hours ($TUNER_CYCLES tuning cycles)"
echo "  Tuner:      every 30 minutes"
echo "============================================================"
echo ""

# Clear previous trade/signal logs for fresh session
> "$DATA_DIR/trade_history.jsonl" 2>/dev/null || true
> "$DATA_DIR/signal_history.jsonl" 2>/dev/null || true
rm -f "$DATA_DIR/tuned_params.json" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] Starting paper trader with live CEX data..."
echo "[$(date +%H:%M:%S)] Trader log: $TRADER_LOG"
echo "[$(date +%H:%M:%S)] Web dashboard: http://localhost:8080"
echo ""

# Start paper trader in background
$PYTHON -m src.simulator.run_paper_live > "$TRADER_LOG" 2>&1 &
TRADER_PID=$!
echo "[$(date +%H:%M:%S)] Paper trader PID: $TRADER_PID"

# Verify it started
sleep 5
if ! kill -0 $TRADER_PID 2>/dev/null; then
    echo "[ERROR] Paper trader failed to start. Check $TRADER_LOG"
    tail -20 "$TRADER_LOG"
    exit 1
fi
echo "[$(date +%H:%M:%S)] Paper trader running."
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] Shutting down..."
    kill $TRADER_PID 2>/dev/null
    wait $TRADER_PID 2>/dev/null
    echo "[$(date +%H:%M:%S)] Session complete."

    # Print final summary
    echo ""
    echo "============================================================"
    echo "  SESSION SUMMARY"
    echo "============================================================"
    if [ -f "$DATA_DIR/trade_history.jsonl" ]; then
        TRADE_COUNT=$(wc -l < "$DATA_DIR/trade_history.jsonl" | tr -d ' ')
        echo "  Total trades logged: $TRADE_COUNT"
    fi
    if [ -f "$DATA_DIR/signal_history.jsonl" ]; then
        SIGNAL_COUNT=$(wc -l < "$DATA_DIR/signal_history.jsonl" | tr -d ' ')
        echo "  Total signals logged: $SIGNAL_COUNT"
    fi
    if [ -f "$DATA_DIR/tuner_log.jsonl" ]; then
        TUNER_COUNT=$(wc -l < "$DATA_DIR/tuner_log.jsonl" | tr -d ' ')
        echo "  Tuning cycles run: $TUNER_COUNT"
    fi
    echo "  Trader log: $TRADER_LOG"
    echo "  Tuner log:  $TUNER_SESSION_LOG"
    echo "============================================================"
}
trap cleanup EXIT INT TERM

# Wait 5 minutes before first tuning (let some trades accumulate)
echo "[$(date +%H:%M:%S)] Waiting 5 minutes before first tuning cycle..."
sleep 300

# Run tuning cycles
START_TIME=$(date +%s)
CYCLE=0

while true; do
    CYCLE=$((CYCLE + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))

    if [ $ELAPSED -ge $TOTAL_DURATION ]; then
        echo "[$(date +%H:%M:%S)] 4-hour session complete."
        break
    fi

    # Check trader is still running
    if ! kill -0 $TRADER_PID 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] Paper trader stopped unexpectedly. Restarting..."
        $PYTHON -m src.simulator.run_paper_live >> "$TRADER_LOG" 2>&1 &
        TRADER_PID=$!
        sleep 5
    fi

    REMAINING=$(( (TOTAL_DURATION - ELAPSED) / 60 ))
    echo ""
    echo "[$(date +%H:%M:%S)] ── TUNING CYCLE $CYCLE/$TUNER_CYCLES ($REMAINING min remaining) ──"

    # Run auto-tuner
    $PYTHON -m src.simulator.auto_tuner 2>&1 | tee -a "$TUNER_SESSION_LOG"

    echo "[$(date +%H:%M:%S)] Next tuning in 30 minutes..."

    # Sleep until next cycle (but check if session is over)
    for i in $(seq 1 $TUNER_INTERVAL); do
        ELAPSED=$(( $(date +%s) - START_TIME ))
        if [ $ELAPSED -ge $TOTAL_DURATION ]; then
            break 2
        fi
        sleep 1
    done
done
