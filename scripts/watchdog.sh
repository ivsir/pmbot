#!/bin/bash
# Watchdog script — keeps the arbitrage bot running for 24 hours.
# Usage: nohup bash scripts/watchdog.sh > /tmp/watchdog.log 2>&1 &

set -euo pipefail

BOT_DIR="/home/rash/Documents/arbitragemarkets"
BOT_LOG="/tmp/arb_bot.log"
WATCHDOG_LOG="/tmp/watchdog.log"
RUNTIME_HOURS=24
CHECK_INTERVAL=30  # seconds between health checks
MAX_RESTARTS=50

start_time=$(date +%s)
end_time=$((start_time + RUNTIME_HOURS * 3600))
restart_count=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

start_bot() {
    cd "$BOT_DIR"
    # Clear stale caches
    find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    # Free port 8080 if needed
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 1
    # Start bot
    nohup python -m src.main > "$BOT_LOG" 2>&1 &
    local pid=$!
    log "Bot started with PID $pid"
    echo "$pid"
}

is_bot_running() {
    pgrep -f "python -m src.main" > /dev/null 2>&1
}

get_bot_pid() {
    pgrep -f "python -m src.main" 2>/dev/null | head -1
}

check_bot_health() {
    local pid=$(get_bot_pid)
    if [ -z "$pid" ]; then
        return 1
    fi
    # Check if process is responsive (not zombie)
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi
    # Check if log has been updated in the last 60 seconds
    if [ -f "$BOT_LOG" ]; then
        local log_age=$(( $(date +%s) - $(stat -c %Y "$BOT_LOG") ))
        if [ "$log_age" -gt 60 ]; then
            log "WARNING: Bot log hasn't been updated in ${log_age}s"
            return 1
        fi
    fi
    return 0
}

# ── Main loop ──
log "Watchdog started. Will run for ${RUNTIME_HOURS}h until $(date -d @$end_time '+%Y-%m-%d %H:%M:%S')"

# Ensure bot is running at start
if ! is_bot_running; then
    log "Bot not running, starting..."
    start_bot
    restart_count=$((restart_count + 1))
    sleep 10  # Give it time to initialize
fi

while [ "$(date +%s)" -lt "$end_time" ]; do
    sleep "$CHECK_INTERVAL"

    if ! check_bot_health; then
        if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
            log "ERROR: Max restarts ($MAX_RESTARTS) reached. Giving up."
            exit 1
        fi

        log "Bot is down! Restarting... (restart #$((restart_count + 1)))"

        # Kill any zombie processes
        pkill -9 -f "python -m src.main" 2>/dev/null || true
        sleep 3

        start_bot
        restart_count=$((restart_count + 1))
        sleep 15  # Wait for initialization
    fi

    # Periodic status log (every 5 minutes)
    elapsed=$(( $(date +%s) - start_time ))
    remaining=$(( end_time - $(date +%s) ))
    if [ $((elapsed % 300)) -lt "$CHECK_INTERVAL" ]; then
        pid=$(get_bot_pid)
        hours_left=$((remaining / 3600))
        mins_left=$(( (remaining % 3600) / 60 ))
        log "Status: PID=$pid, restarts=$restart_count, remaining=${hours_left}h${mins_left}m"
    fi
done

log "24-hour runtime complete. Restarts: $restart_count"
