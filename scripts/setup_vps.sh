#!/usr/bin/env bash
# =============================================================================
# Polymarket Arbitrage Bot — VPS Setup Script
# Tested on Ubuntu 22.04 LTS (Vultr/DO/Hetzner)
#
# Usage:
#   1. SSH into your VPS: ssh root@<your-vps-ip>
#   2. Upload this script: scp scripts/setup_vps.sh root@<your-vps-ip>:~/
#   3. Run: chmod +x setup_vps.sh && ./setup_vps.sh
# =============================================================================

set -e

REPO_URL="https://github.com/rasheemtrq/arbitragemarkets.git"
APP_DIR="/opt/arbbot"
SERVICE_NAME="arbbot"
PYTHON_VERSION="3.11"

echo "============================================"
echo " Polymarket Arbitrage Bot — VPS Setup"
echo "============================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    wget \
    htop \
    tmux \
    ufw

# ── 2. Firewall ───────────────────────────────────────────────────────────────
echo "[2/7] Configuring firewall..."
ufw allow OpenSSH
ufw allow 8080/tcp   # dashboard
ufw --force enable

# ── 3. Clone repo ────────────────────────────────────────────────────────────
echo "[3/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "  Directory exists — pulling latest..."
    cd "$APP_DIR" && git pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# ── 4. Python venv + deps ─────────────────────────────────────────────────────
echo "[4/7] Setting up Python environment..."
python${PYTHON_VERSION} -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
echo "  Dependencies installed."

# ── 5. .env file ─────────────────────────────────────────────────────────────
echo "[5/7] Setting up .env..."
if [ ! -f "$APP_DIR/.env" ]; then
    echo ""
    echo "  ⚠️  No .env file found. You need to create one."
    echo "  Copy your local .env to the server:"
    echo "    scp .env root@<your-vps-ip>:${APP_DIR}/.env"
    echo ""
    echo "  Or paste contents now (Ctrl+D when done):"
    cat > "$APP_DIR/.env"
else
    echo "  .env already exists — skipping."
fi

# ── 6. Data directories ──────────────────────────────────────────────────────
echo "[6/7] Creating data directories..."
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/logs"
mkdir -p "$APP_DIR/models"
chown -R root:root "$APP_DIR"

# ── 7. systemd service ───────────────────────────────────────────────────────
echo "[7/7] Installing systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=Polymarket Arbitrage Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/logs/bot.log
StandardError=append:${APP_DIR}/logs/bot.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Make sure .env is at ${APP_DIR}/.env"
echo "  2. Start the bot:  systemctl start ${SERVICE_NAME}"
echo "  3. Check status:   systemctl status ${SERVICE_NAME}"
echo "  4. View logs:      tail -f ${APP_DIR}/logs/bot.log"
echo "  5. Dashboard:      http://<your-vps-ip>:8080"
echo ""
echo "Other commands:"
echo "  Stop bot:    systemctl stop ${SERVICE_NAME}"
echo "  Restart bot: systemctl restart ${SERVICE_NAME}"
echo "  Auto-update: cd ${APP_DIR} && git pull && systemctl restart ${SERVICE_NAME}"
echo ""
