# Oracle Cloud Always Free Deployment Guide

This guide walks you through deploying the Polymarket arbitrage bot to Oracle Cloud Always Free tier for 24/7 operation.

## Why Oracle Cloud Always Free?

- **Truly free forever**: 4 ARM vCPUs, 24GB RAM, 200GB storage
- **No credit card needed** after registration
- **Auto-restart on crash**: Systemd handles failure recovery
- **Only requirement**: Log in once per month to keep account active

---

## Step 1: Create Oracle Cloud Account

1. Go to [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Click **"Sign Up"** (no credit card required)
3. Enter email, create account
4. **Skip** adding payment method (it's free)
5. Verify email

---

## Step 2: Create Compute Instance

1. Log into [Oracle Cloud Console](https://www.oracle.com/cloud/sign-in/)
2. Navigate to **Compute → Instances**
3. Click **"Create Instance"**
4. Configure:
   - **Name**: `polybot` (or any name)
   - **Image**: Ubuntu 22.04 (or 24.04)
   - **Shape**: Ampere (ARM) A1 Compute - **Always Free Eligible**
     - 4 OCPUs (ARM-based, very capable)
     - 24GB RAM
   - **VCN**: Create new or select default
   - **Public IP**: Assign one
   - **SSH Key**: Download and save `.key` file (you'll need this)
5. Click **"Create"** and wait 2-3 minutes

**Note the Instance IP Address** once it starts

---

## Step 3: Connect to Instance via SSH

```bash
# From your local machine, change key permissions
chmod 600 /path/to/your-instance-key.key

# SSH into instance (replace IP)
ssh -i /path/to/your-instance-key.key ubuntu@<INSTANCE_IP>
```

---

## Step 4: Deploy Bot (Run these commands on the instance)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git curl wget

# Clone your repo (use HTTPS, or upload if private)
cd ~ && git clone https://github.com/yourusername/windsurf-project.git bot
cd ~/bot

# Create venv and install requirements
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 5: Set Up Environment Variables

```bash
# Copy your .env file to the server
# Option A: Upload via scp (from your local machine)
scp -i /path/to/key.key /path/to/.env ubuntu@<INSTANCE_IP>:~/bot/.env

# Option B: Create it manually (less secure but works)
nano ~/bot/.env
# Paste your .env contents
# Press Ctrl+O, Enter, Ctrl+X to save
```

**Critical**: Make sure these env vars are set:
- `POLYGON_PRIVATE_KEY` - Your wallet's private key
- `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`
- `LIVE_TRADING_ENABLED=true`
- `MARKET_MODE=5min_updown`

---

## Step 6: Create Systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/polybot.service
```

Paste this content:

```ini
[Unit]
Description=Polymarket Arbitrage Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bot
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=/home/ubuntu/bot/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/ubuntu/bot/venv/bin/python -m src.main
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/ubuntu/bot/bot.log
StandardError=append:/home/ubuntu/bot/bot.log
KillMode=process
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Save with `Ctrl+O`, `Enter`, `Ctrl+X`

---

## Step 7: Enable and Start the Service

```bash
# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable polybot

# Start the service now
sudo systemctl start polybot

# Check status
sudo systemctl status polybot

# View real-time logs
sudo journalctl -u polybot -f

# Or tail the log file
tail -f ~/bot/bot.log
```

---

## Step 8: Verify Bot is Running

```bash
# Check if process is alive
ps aux | grep "python.*main"

# Check if log file is being written to
tail -20 ~/bot/bot.log

# Check if dashboard is accessible (if running on localhost:8080)
curl -s http://localhost:8080/api/health | jq .
```

---

## Step 9: Set Up Log Rotation (Optional but Recommended)

To prevent logs from growing too large:

```bash
sudo nano /etc/logrotate.d/polybot
```

Paste:

```
/home/ubuntu/bot/bot.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 ubuntu ubuntu
}
```

---

## Step 10: Keep Account Active

To prevent Oracle from shutting down your instance after 3 months of inactivity:

- **Set calendar reminder** to log in once per month
- OR use a cron job on the instance to periodically touch a file:

```bash
# Add to crontab (on the instance)
crontab -e

# Add this line (runs every Monday at 9am UTC)
0 9 * * 1 /bin/true
```

---

## Monitoring & Maintenance

### Check bot status:
```bash
sudo systemctl status polybot
```

### Restart bot:
```bash
sudo systemctl restart polybot
```

### View last 50 lines of logs:
```bash
tail -50 ~/bot/bot.log
```

### Update code from GitHub:
```bash
cd ~/bot && git pull
# May need to restart if code changed
sudo systemctl restart polybot
```

### Stop bot:
```bash
sudo systemctl stop polybot
```

---

## Troubleshooting

**Bot not starting?**
```bash
# Check detailed error
sudo journalctl -u polybot -n 100

# Check env vars are loaded
sudo systemctl show-environment polybot
```

**Out of disk space?**
```bash
# Check disk usage
df -h

# Compress old logs
gzip /home/ubuntu/bot/*.log
```

**Need to edit .env?**
```bash
nano ~/bot/.env
sudo systemctl restart polybot
```

---

## Cost Summary

- **Monthly cost**: $0
- **No credit card needed** (stays free tier)
- **Only requirement**: Log in once per month

---

## Next Steps

1. Create your Oracle Cloud account (free, no CC)
2. Create the Ampere A1 instance
3. SSH in and run the deployment commands
4. Monitor the logs to verify bot is trading

Estimated setup time: **15-20 minutes**
