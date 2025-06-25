#!/bin/bash
# FMP Package Installation Script
# This script tries multiple methods to install the FMP package

echo "🚀 Installing FMP package..."

# Method 1: Try local installation first
echo "📂 Trying local installation..."
if pip install -e /home/noufal/automation/fmp 2>/dev/null; then
    echo "✅ FMP package installed from local directory"
    exit 0
fi

# Method 2: Try SSH clone (recommended)
echo "🔑 Trying SSH clone from GitHub..."
if git clone git@github.com:noufal85/fmp.git 2>/dev/null; then
    echo "📦 SSH clone successful, installing..."
    if cd fmp && pip install . 2>/dev/null; then
        echo "✅ FMP package installed from GitHub via SSH"
        cd .. > /dev/null
        rm -rf fmp
        exit 0
    fi
    # Clean up if install failed
    cd .. > /dev/null 2>&1
    rm -rf fmp 2>/dev/null
fi

# All methods failed
echo "❌ Automatic installation failed. Manual installation required:"
echo ""
echo "📝 Choose one of the following methods:"
echo ""
echo "1️⃣  If you have SSH keys configured (recommended):"
echo "   git clone git@github.com:noufal85/fmp.git && cd fmp && pip install . && cd .. && rm -rf fmp"
echo ""
echo "2️⃣  If you have a GitHub PAT token:"
echo "   pip install git+https://your_token@github.com/noufal85/fmp.git"
echo ""
echo "3️⃣  If you have the FMP package locally:"
echo "   pip install -e /path/to/your/fmp"
echo ""
echo "4️⃣  Configure git credentials (one-time setup):"
echo "   git config --global credential.helper store"
echo "   # Then manually enter credentials when prompted during clone"
echo ""

exit 1