#!/bin/bash

# Complete setup script for LogicHedge Orchestrator

echo "🚀 Setting up LogicHedge Orchestrator..."

# 1. Create virtual environment
echo "📦 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip

# Core dependencies
pip install aiohttp web3 ccxt numpy scipy pandas flask pyyaml

# Exchange-specific
pip install dydx3  # dYdX
pip install requests websockets

# 3. Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {configs,logs,data,strategies,integrations,orchestrator,dashboard/templates}

# 4. Create configuration files
echo "⚙️ Creating configuration files..."

# Prime brokerage config
cat > configs/prime_brokerage.yaml << 'EOF'
prime_brokerage_services:
  cedehub:
    enabled: true
    api_key: ${CEDEHUB_API_KEY}
    api_secret: ${CEDEHUB_API_SECRET}
    capital_allocation: 100000
    
  copper:
    enabled: false
    api_key: ${COPPER_API_KEY}
    api_secret: ${COPPER_API_SECRET}
    client_id: ${COPPER_CLIENT_ID}
    capital_allocation: 50000
    
  falconx:
    enabled: false
    api_key: ${FALCONX_API_KEY}
    api_secret: ${FALCONX_API_SECRET}
    capital_allocation: 50000
    
  dydx:
    enabled: true
    private_key: ${DYDX_PRIVATE_KEY}
    stark_key: ${DYDX_STARK_KEY}
    eth_address: ${DYDX_ETH_ADDRESS}
    capital_allocation: 50000
    
  gamma:
    enabled: false
    api_key: ${GAMMA_API_KEY}
    capital_allocation: 20000
    
  ribbon:
    enabled: true
    wallet: ${RIBBON_WALLET}
    capital_allocation: 30000
    
  hashnote:
    enabled: true
    wallet: ${HASHNOTE_WALLET}
    capital_allocation: 50000
    
  paradex:
    enabled: false
    stark_key: ${PARADEX_STARK_KEY}
    eth_key: ${PARADEX_ETH_KEY}
    capital_allocation: 20000

risk_limits:
  max_delta_exposure: 50000
  max_var_95: 20000
  max_daily_loss: 5000
  max_leverage: 3.0
EOF

# Environment template
cat > .env.example << 'EOF'
# CedeHub
CEDEHUB_API_KEY=your_cedehub_api_key
CEDEHUB_API_SECRET=your_cedehub_api_secret

# dYdX
DYDX_PRIVATE_KEY=your_dydx_private_key
DYDX_STARK_KEY=your_dydx_stark_key
DYDX_ETH_ADDRESS=your_eth_address

# Ribbon
RIBBON_WALLET=your_ribbon_wallet

# Hashnote
HASHNOTE_WALLET=your_hashnote_wallet

# Infura
INFURA_KEY=your_infura_key
EOF

echo "⚠️  Please update .env with your actual API keys"
echo "⚠️  Then run: cp .env.example .env"

# 5. Set permissions
chmod +x scripts/*.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Start: python scripts/deploy_orchestrator.py"
echo ""
echo "For dashboard:"
echo "  python dashboard/app.py"
