#!/bin/bash
# BenderBox Prerequisites Installation Script (Linux/macOS)
# This script installs required and optional dependencies for BenderBox

set -e

# Configuration
TOOLS_DIR="$HOME/.benderbox/tools"
LLAMA_RELEASE_URL="https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
INSTALL_ALL=false
INSTALL_REQUIRED=false
INSTALL_LLAMA=false
FORCE=false

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/debian_version ]; then
            DISTRO="debian"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="redhat"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="darwin"
        DISTRO="macos"
    else
        OS="unknown"
        DISTRO="unknown"
    fi
}

# Output functions
print_header() {
    echo ""
    echo -e "${CYAN}================================================${NC}"
    echo -e "${CYAN}  BenderBox Prerequisites Installer ($OS)${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[X]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[*]${NC} $1"
}

show_help() {
    print_header
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all         Install all prerequisites"
    echo "  --required    Install only required prerequisites"
    echo "  --llama-cli   Install llama-cli (for GGUF analysis)"
    echo "  --force       Force reinstall even if already installed"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all"
    echo "  $0 --llama-cli"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get installed version
get_version() {
    local cmd=$1
    local pattern=$2
    local output=$($cmd 2>&1) || true
    echo "$output" | grep -oE "$pattern" | head -1
}

# Check Python
check_python() {
    print_info "Checking Python..."

    if command_exists python3; then
        version=$(python3 --version 2>&1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)

        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            print_success "Python $version is installed"
            return 0
        else
            print_warning "Python $version found, but 3.9+ is required"
            return 1
        fi
    elif command_exists python; then
        version=$(python --version 2>&1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
        major=$(echo $version | cut -d. -f1)

        if [ "$major" -ge 3 ]; then
            print_success "Python $version is installed"
            return 0
        fi
    fi

    print_warning "Python 3.9+ not found"
    if [ "$OS" == "darwin" ]; then
        print_info "Install via: brew install python@3.11"
    elif [ "$DISTRO" == "debian" ]; then
        print_info "Install via: sudo apt-get install python3"
    elif [ "$DISTRO" == "redhat" ]; then
        print_info "Install via: sudo dnf install python3"
    fi
    return 1
}

# Check Git
check_git() {
    print_info "Checking Git..."

    if command_exists git; then
        version=$(git --version | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
        print_success "Git $version is installed"
        return 0
    fi

    print_warning "Git not found"
    if [ "$OS" == "darwin" ]; then
        print_info "Install via: brew install git"
    elif [ "$DISTRO" == "debian" ]; then
        print_info "Install via: sudo apt-get install git"
    elif [ "$DISTRO" == "redhat" ]; then
        print_info "Install via: sudo dnf install git"
    fi
    return 1
}

# Install llama-cli
install_llama_cli() {
    print_info "Checking llama-cli..."

    local llama_path="$TOOLS_DIR/llama-cli"

    # Check if already installed
    if [ -f "$llama_path" ] && [ "$FORCE" != true ]; then
        print_success "llama-cli is already installed at $llama_path"
        return 0
    fi

    # Also check PATH
    if command_exists llama-cli && [ "$FORCE" != true ]; then
        local existing_path=$(which llama-cli)
        print_success "llama-cli found at $existing_path"
        return 0
    fi

    print_info "Installing llama-cli..."

    # Create tools directory
    mkdir -p "$TOOLS_DIR"

    # Check for required tools
    if ! command_exists curl; then
        print_error "curl is required but not installed"
        return 1
    fi

    if ! command_exists unzip; then
        print_error "unzip is required but not installed"
        return 1
    fi

    # Get latest release info
    print_info "Fetching latest release information..."
    local release_info=$(curl -s "$LLAMA_RELEASE_URL")
    local tag_name=$(echo "$release_info" | grep -oP '"tag_name":\s*"\K[^"]+' | head -1)

    if [ -z "$tag_name" ]; then
        print_error "Could not determine latest release"
        return 1
    fi

    tag_name="${tag_name#v}"  # Remove 'v' prefix
    print_info "Latest release: $tag_name"

    # Determine download URL based on OS
    local download_url=""
    local arch=$(uname -m)

    if [ "$OS" == "linux" ]; then
        download_url="https://github.com/ggerganov/llama.cpp/releases/download/b${tag_name}/llama-b${tag_name}-bin-ubuntu-x64.zip"
    elif [ "$OS" == "darwin" ]; then
        if [ "$arch" == "arm64" ]; then
            download_url="https://github.com/ggerganov/llama.cpp/releases/download/b${tag_name}/llama-b${tag_name}-bin-macos-arm64.zip"
        else
            download_url="https://github.com/ggerganov/llama.cpp/releases/download/b${tag_name}/llama-b${tag_name}-bin-macos-x64.zip"
        fi
    fi

    if [ -z "$download_url" ]; then
        print_error "No binary available for $OS $arch"
        return 1
    fi

    print_info "Downloading from $download_url..."

    # Create temp directory
    local temp_dir=$(mktemp -d)
    local zip_path="$temp_dir/llama.zip"
    local extract_path="$temp_dir/extracted"

    # Download
    if ! curl -L -o "$zip_path" "$download_url"; then
        print_error "Download failed"
        rm -rf "$temp_dir"
        return 1
    fi

    print_info "Extracting archive..."
    mkdir -p "$extract_path"
    unzip -q "$zip_path" -d "$extract_path"

    # Find llama-cli
    local llama_exe=$(find "$extract_path" -name "llama-cli" -o -name "main" | head -1)

    if [ -z "$llama_exe" ]; then
        print_error "Could not find llama-cli in archive"
        print_info "Archive contents:"
        find "$extract_path" -type f
        rm -rf "$temp_dir"
        return 1
    fi

    # Copy to tools directory
    cp "$llama_exe" "$llama_path"
    chmod +x "$llama_path"

    # Clean up
    rm -rf "$temp_dir"

    print_success "llama-cli installed to $llama_path"

    # Add to PATH suggestion
    print_info "To add to PATH, add this to your ~/.bashrc or ~/.zshrc:"
    echo -e "  ${YELLOW}export PATH=\"\$PATH:$TOOLS_DIR\"${NC}"
    echo ""

    return 0
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."

    if command_exists python3; then
        python3 -m pip install --upgrade pip
        python3 -m pip install -e ".[all,dev]"
    else
        python -m pip install --upgrade pip
        python -m pip install -e ".[all,dev]"
    fi

    print_success "Python dependencies installed"
    return 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --required)
            INSTALL_REQUIRED=true
            shift
            ;;
        --llama-cli)
            INSTALL_LLAMA=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            detect_os
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
detect_os
print_header

declare -A results

# Always check Python (required)
if check_python; then
    results["python"]=0
else
    results["python"]=1
fi

# Check Git
if check_git; then
    results["git"]=0
else
    results["git"]=1
fi

# Install llama-cli if requested
if [ "$INSTALL_ALL" = true ] || [ "$INSTALL_LLAMA" = true ]; then
    if install_llama_cli; then
        results["llama-cli"]=0
    else
        results["llama-cli"]=1
    fi
fi

# Install Python dependencies if Python is available
if [ "$INSTALL_ALL" = true ] || [ "$INSTALL_REQUIRED" = true ]; then
    if [ "${results["python"]}" = 0 ]; then
        if install_python_deps; then
            results["pip-deps"]=0
        else
            results["pip-deps"]=1
        fi
    fi
fi

# Summary
echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Installation Summary${NC}"
echo -e "${CYAN}================================================${NC}"

success=0
failed=0

for key in "${!results[@]}"; do
    if [ "${results[$key]}" = 0 ]; then
        print_success "$key"
        ((success++))
    else
        print_error "$key"
        ((failed++))
    fi
done

echo ""
if [ $failed -eq 0 ]; then
    print_success "All prerequisites installed successfully!"
else
    print_warning "$failed prerequisite(s) need attention"
fi

echo ""
print_info "BenderBox tools directory: $TOOLS_DIR"
echo ""
