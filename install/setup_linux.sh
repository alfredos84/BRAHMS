#!/usr/bin/env bash
# ============================================================
# BRAHMS — Linux installer (Ubuntu/Debian)
#
# What this does:
#   1. Installs system dependencies (apt)
#   2. Creates a Python virtual environment and installs requirements.txt
#   3. Builds the CPU engine (engine_omp) — always
#   4. Builds the GPU engine (engine_gpu) — only if nvcc is found
#   5. Installs a desktop launcher (brahms.desktop) + icon, and a
#      shortcut on the Desktop, so the app opens with a double-click
#      like any other installed application.
#
# Usage:
#   cd BRAHMS/            (the cloned repo root)
#   bash install/setup_linux.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Installing system dependencies (apt)..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential \
    libfftw3-dev \
    libhdf5-dev \
    nlohmann-json3-dev \
    fonts-liberation

echo ""
echo "==> Creating Python virtual environment: venv/"
python3 -m venv venv
source venv/bin/activate

echo "==> Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "==> Building CPU engine (engine_omp)..."
make -C engine_omp -B

if command -v nvcc >/dev/null 2>&1; then
    echo ""
    echo "==> NVIDIA CUDA compiler found — building GPU engine (engine_gpu)..."
    ARCH="sm_75"
    if command -v nvidia-smi >/dev/null 2>&1; then
        CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
        if [ -n "$CAP" ]; then
            ARCH="sm_${CAP}"
        fi
    fi
    echo "    Compute capability: $ARCH"
    make -C engine_gpu -B ARCH="$ARCH" || \
        echo "    [WARN] GPU engine build failed — the app still works with the CPU engine."
else
    echo ""
    echo "==> No NVIDIA CUDA compiler (nvcc) found — skipping GPU engine."
    echo "    The app will run fine with the CPU (OpenMP) engine only."
fi

echo ""
echo "==> Installing desktop launcher..."

ICON_SRC="$ROOT_DIR/icon/icon.png"
ICON_DST_DIR="$HOME/.local/share/icons/hicolor/512x512/apps"
mkdir -p "$ICON_DST_DIR"
cp -f "$ICON_SRC" "$ICON_DST_DIR/brahms.png"

LAUNCHER="$ROOT_DIR/install/run_brahms.sh"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
cd "$ROOT_DIR"
source venv/bin/activate
exec python main.py
EOF
chmod +x "$LAUNCHER"

DESKTOP_FILE_CONTENT="[Desktop Entry]
Type=Application
Name=BRAHMS
Comment=Three-Wave Mixing Simulator
Exec=$LAUNCHER
Icon=brahms
Terminal=false
Categories=Science;Education;
StartupWMClass=brahms"

APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
echo "$DESKTOP_FILE_CONTENT" > "$APPS_DIR/brahms.desktop"
chmod +x "$APPS_DIR/brahms.desktop"
update-desktop-database "$APPS_DIR" 2>/dev/null || true

if [ -d "$HOME/Desktop" ]; then
    echo "$DESKTOP_FILE_CONTENT" > "$HOME/Desktop/brahms.desktop"
    chmod +x "$HOME/Desktop/brahms.desktop"
    # Mark as trusted so GNOME/Nautilus shows it as a launcher, not a text file
    gio set "$HOME/Desktop/brahms.desktop" metadata::trusted true 2>/dev/null || true
fi

gtk-update-icon-cache -f "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo ""
echo "==> Done!"
echo "    BRAHMS is now in your applications menu and on your Desktop."
echo "    You can also launch it manually with:"
echo "        bash install/run_brahms.sh"
