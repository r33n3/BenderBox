#!/bin/bash
# BenderBox Installation Launcher (Linux/macOS)
# Run: chmod +x install.sh && ./install.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  BenderBox Installation Launcher${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Unknown"
fi

echo -e "[*] Detected OS: ${CYAN}$OS${NC}"
echo ""

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[ERROR] Python not found!${NC}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install Python via Homebrew:"
        echo "  brew install python@3.11"
    else
        echo "Install Python via package manager:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "  Fedora/RHEL:   sudo dnf install python3 python3-pip"
        echo "  Arch:          sudo pacman -S python python-pip"
    fi
    echo ""
    exit 1
fi

# Check Python version
PYVER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYMAJOR=$(echo $PYVER | cut -d. -f1)
PYMINOR=$(echo $PYVER | cut -d. -f2)

if [ "$PYMAJOR" -lt 3 ] || ([ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 9 ]); then
    echo -e "${YELLOW}[WARNING] Python $PYVER found, but 3.9+ is required${NC}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Upgrade via: brew install python@3.11"
    else
        echo "Please install Python 3.9 or later"
    fi
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Python $PYVER found"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if setup_wizard.py exists
if [ ! -f "$SCRIPT_DIR/setup_wizard.py" ]; then
    echo -e "${RED}[ERROR] setup_wizard.py not found!${NC}"
    echo "Make sure you're running this from the BenderBox directory."
    exit 1
fi

# Run the setup wizard
echo "Starting BenderBox Setup Wizard..."
echo ""
$PYTHON_CMD "$SCRIPT_DIR/setup_wizard.py"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Setup wizard failed!${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Installation Complete!${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo "To start BenderBox, run:"
echo -e "  ${GREEN}python3 bb.py -i${NC}"
echo ""
echo "Or add an alias to your ~/.bashrc or ~/.zshrc:"
echo -e "  ${YELLOW}alias bb='python3 $(pwd)/bb.py'${NC}"
echo ""
