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

# Install dependencies
echo "📚 Installing dependencies..."
if pip install -r requirements.txt; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies. You may need to configure GitHub PAT token:"
    echo "   git config --global credential.helper store"
    echo "   echo 'https://YOUR_USERNAME:YOUR_PAT_TOKEN@github.com' >> ~/.git-credentials"
    exit 1
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