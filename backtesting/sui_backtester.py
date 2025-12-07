"""
Backtesting module for SUI_USDT Futures Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import yaml
import matplotlib.pyplot as plt
from strategies.futures.sui_usdt_futures import SUIUSDTFuturesStrategy

class SUIBacktester:
    """Backtester for SUI futures strategy"""
    
    def __init__(self, config_path: str = "config/sui_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = []
        self.equity_curve = []
        
    def run_backtest(self, 
                    data: pd.DataFrame,
                    initial_capital: float = 10000) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            initial_capital: Starting capital in USDT
        """
        
        capital = initial_capital
        position = None
        trades = []
        
        # Simulate strategy
        for i in range(100, len(data)):
            current_data = data.iloc[:i+1]
            current_price = data['close'].iloc[i]
            
            # Here you would simulate the strategy logic
            # This is a simplified version
            
            # Example entry logic
            if position is None:
                # Check for entry signal
                if self._check_entry_signal(current_data):
                    position = {
                        'entry_price': current_price,
                        'entry_index': i,
                        'size': capital * 0.1 / current_price,  # 10% position
                        'side': 'LONG'
                    }
                    
            # Check for exit
            elif position:
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                
                # Take profit at 5%
                if pnl_pct >= 0.05:
                    capital += capital * 0.1 * pnl_pct  # Update capital
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'duration': i - position['entry_index']
                    })
                    position = None
                
                # Stop loss at 2%
                elif pnl_pct <= -0.02:
                    capital += capital * 0.1 * pnl_pct
                    trades.append({
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'duration': i - position['entry_index']
                    })
                    position = None
            
            # Record equity curve
            self.equity_curve.append(capital)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, initial_capital)
        
        return {
            'trades': trades,
            'metrics': metrics,
            'equity_curve': self.equity_curve,
            'final_capital': capital
        }
    
    def _check_entry_signal(self, data: pd.DataFrame) -> bool:
        """Simplified entry signal logic"""
        # Implement actual strategy logic here
        return False  # Placeholder
    
    def _calculate_metrics(self, trades: List, initial_capital: float) -> Dict:
        """Calculate performance metrics"""
        
        if not trades:
            return {}
        
        # Calculate metrics
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['pnl_pct'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum([t['pnl_pct'] for t in winning_trades]) / 
                          sum([t['pnl_pct'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in trades]
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Max drawdown
        equity = initial_capital
        peak = equity
        max_dd = 0
        
        for trade in trades:
            equity *= (1 + trade['pnl_pct'] * 0.1)  # 10% position size
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_return_pct': (equity - initial_capital) / initial_capital * 100
        }
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[
