#!/usr/bin/env bash
# One-time / repeatable setup of ResearchOS on an Ubuntu EC2 instance.
#   usage: ./setup_ec2.sh <git-tag> <domain> <email-for-letsencrypt>
# Expects /opt/researchos/.env to exist already (copied with scp; it is never in git).
set -euo pipefail

TAG="${1:-v1.0.0}"
DOMAIN="${2:-researchos.duckdns.org}"
EMAIL="${3:-}"
REPO="https://github.com/vaibhavparmarog/ResearchOS.git"
DIR=/opt/researchos

# --- packages: docker, nginx, certbot -------------------------------------------------
if ! command -v docker >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl git nginx certbot python3-certbot-nginx
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi
sudo apt-get install -y git nginx certbot python3-certbot-nginx >/dev/null

# --- 2 GB swap: headroom for OCR / large PDFs on small instances --------------------------
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# --- code at the requested release -------------------------------------------------------
sudo mkdir -p "$DIR" && sudo chown "$USER":"$USER" "$DIR"
if [ ! -d "$DIR/src/.git" ]; then git clone "$REPO" "$DIR/src"; fi
cd "$DIR/src"
git fetch --tags --force origin
git checkout -q "$TAG"
test -f "$DIR/.env" || { echo "missing $DIR/.env"; exit 1; }
chmod 600 "$DIR/.env"
ln -sf "$DIR/.env" .env

# --- build + run ---------------------------------------------------------------------------
sudo docker compose up -d --build
for i in $(seq 1 30); do curl -fs http://127.0.0.1:8000/api/health && break; sleep 2; done
echo

# --- nginx reverse proxy + HTTPS -----------------------------------------------------------
sudo cp deploy/nginx-researchos.conf /etc/nginx/sites-available/researchos
sudo cp deploy/researchos_proxy.conf /etc/nginx/researchos_proxy.conf
sudo sed -i "s/server_name .*/server_name $DOMAIN;/" /etc/nginx/sites-available/researchos
sudo ln -sf /etc/nginx/sites-available/researchos /etc/nginx/sites-enabled/researchos
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
if [ -n "$EMAIL" ]; then
  sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
else
  sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect
fi
sudo systemctl reload nginx
echo "ResearchOS $TAG is live at https://$DOMAIN"
