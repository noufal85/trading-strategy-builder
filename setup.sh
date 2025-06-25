#!/bin/bash
# Trading Strategy Builder - Setup Script

echo "🚀 Setting up Trading Strategy Builder..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file from template..."
    cp .env.sample .env
    echo "⚠️  Please edit .env file and add your API keys"
else
    echo "✅ .env file already exists"
fi

# Install core dependencies
echo "📚 Installing core dependencies..."
if pip install -r requirements.txt; then
    echo "✅ Core dependencies installed successfully"
else
    echo "❌ Failed to install core dependencies"
    exit 1
fi

# Install FMP package
echo "📦 Installing FMP package..."
echo "ℹ️  Trying local installation first..."
if pip install -e /home/noufal/automation/fmp 2>/dev/null; then
    echo "✅ FMP package installed from local directory"
else
    echo "⚠️  Local FMP not found. You'll need to install it manually:"
    echo "   Option 1: Using environment variable PAT token (recommended):"
    echo "     export GITHUB_PERSONAL_ACCESS_TOKEN=your_token"
    echo "     pip install git+https://\${GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/noufal85/fmp.git"
    echo "   Option 2: Using inline PAT token:"
    echo "     pip install git+https://your_token@github.com/noufal85/fmp.git"
    echo "   Option 3: From local directory:"
    echo "     pip install -e /path/to/your/fmp"
fi

# Install package in development mode
echo "🔧 Installing strategy_builder package..."
pip install -e .

echo ""
echo "🎉 Setup complete! To get started:"
echo "   source .venv/bin/activate"
echo "   python3 strategy_builder/strategies/trend_following/run_daily_scan.py"
echo ""
echo "📝 Don't forget to edit .env file with your API keys!"