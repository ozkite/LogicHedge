#!/usr/bin/env python3
"""
Deploy the complete orchestrator system
"""

import asyncio
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)

async def main():
    """Deploy and run the orchestrator"""
    # Load configuration
    config_path = Path("configs/prime_brokerage.yaml")
    
    if not config_path.exists():
        print("❌ Configuration file not found!")
        print("Please create configs/prime_brokerage.yaml")
        return
        
    # Initialize orchestrator
    from logichedge.orchestrator.master_orchestrator import MasterOrchestrator
    
    orchestrator = MasterOrchestrator(str(config_path))
    
    try:
        # Start orchestrator
        await orchestrator.initialize()
        
        print("🎯 Orchestrator deployed successfully!")
        print(f"📊 Services online: {len(orchestrator.services)}")
        print(f"💰 Total capital: ${sum(a.allocated_usd for a in orchestrator.capital_allocations.values()):,.2f}")
        
        # Run forever
        await asyncio.Future()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down orchestrator...")
        await orchestrator.shutdown()
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
