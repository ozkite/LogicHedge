#!/bin/bash

# LogicHedge Environment Setup Script

set -e

echo "🚀 Setting up LogicHedge trading environment..."

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,signals}
mkdir -p logs
mkdir -p configs/exchanges

# Create configuration files
echo "⚙️  Setting up configuration..."
if [ ! -f configs/main_config.yaml ]; then
    cp configs/main_config.yaml.example configs/main_config.yaml
    echo "⚠️  Please update configs/main_config.yaml with your API keys"
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please update .env with your API keys"
fi

# Set permissions
chmod +x scripts/*.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update configs/main_config.yaml with your settings"
echo "2. Update .env with your API keys"
echo "3. Run: source venv/bin/activate"
echo "4. Start bot: python scripts/run_bot.py"
echo ""
echo "For development:"
echo "  Run tests: pytest tests/"
echo "  Format code: black logichedge/"
echo "  Type check: mypy logichedge/"
