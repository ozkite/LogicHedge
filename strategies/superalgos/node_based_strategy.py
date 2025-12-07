"""
Node-Based Strategy System from Superalgos
Visual strategy building with nodes and connections
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

class NodeType(Enum):
    DATA_SOURCE = "data_source"
    INDICATOR = "indicator"
    CONDITION = "condition"
    ACTION = "action"
    LOGIC_GATE = "logic_gate"

@dataclass
class StrategyNode:
    """Strategy node in the visual builder"""
    id: str
    node_type: NodeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    value: Any = None
    output: Any = None
    
@dataclass
class NodeConnection:
    """Connection between nodes"""
    source_id: str
    target_id: str
    parameter_name: str  # Which parameter this connection fills
    
class NodeBasedStrategy:
    """
    Node-based strategy system inspired by Superalgos
    Build strategies visually with nodes and connections
    """
    
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.graph = nx.DiGraph()
        
    def add_node(self, node: StrategyNode):
        """Add a node to the strategy"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, type=node.node_type)
        
    def add_connection(self, connection: NodeConnection):
        """Add connection between nodes"""
        self.connections.append(connection)
        self.graph.add_edge(connection.source_id, connection.target_id)
        
    def build_strategy(self, strategy_name: str):
        """Build a complete strategy from nodes"""
        # Create common strategy templates
        
        # Template 1: RSI + MACD Strategy
        if strategy_name == "rsi_macd_combo":
            self._build_rsi_macd_strategy()
            
        # Template 2: Bollinger Band Breakout
        elif strategy_name == "bb_breakout":
            self._build_bb_breakout_strategy()
            
        # Template 3: Multi-Timeframe Trend
        elif strategy_name == "multi_tf_trend":
            self._build_multi_tf_trend_strategy()
            
    def _build_rsi_macd_strategy(self):
        """Build RSI + MACD combo strategy"""
        # Data Source Node
        data_node = StrategyNode(
            id="price_data",
            node_type=NodeType.DATA_SOURCE,
            parameters={"source": "close"}
        )
        self.add_node(data_node)
        
        # RSI Indicator Node
        rsi_node = StrategyNode(
            id="rsi_indicator",
            node_type=NodeType.INDICATOR,
            parameters={"period": 14}
        )
        self.add_node(rsi_node)
        
        # MACD Indicator Node
        macd_node = StrategyNode(
            id="macd_indicator",
            node_type=NodeType.INDICATOR,
            parameters={"fast": 12, "slow": 26, "signal": 9}
        )
        self.add_node(macd_node)
        
        # RSI Buy Condition
        rsi_buy_cond = StrategyNode(
            id="rsi_buy_condition",
            node_type=NodeType.CONDITION,
            parameters={"operator": "<", "value": 30}
        )
        self.add_node(rsi_buy_cond)
        
        # MACD Buy Condition
        macd_buy_cond = StrategyNode(
            id="macd_buy_condition",
            node_type=NodeType.CONDITION,
            parameters={"operator": "crossover", "direction": "above"}
        )
        self.add_node(macd_buy_cond)
        
        # AND Logic Gate
        and_gate = StrategyNode(
            id="buy_and_gate",
            node_type=NodeType.LOGIC_GATE,
            parameters={"gate_type": "AND"}
        )
        self.add_node(and_gate)
        
        # Buy Action
        buy_action = StrategyNode(
            id="buy_action",
            node_type=NodeType.ACTION,
            parameters={"action_type": "market_buy"}
        )
        self.add_node(buy_action)
        
        # Connect nodes
        self.add_connection(NodeConnection("price_data", "rsi_indicator", "input"))
        self.add_connection(NodeConnection("price_data", "macd_indicator", "input"))
        self.add_connection(NodeConnection("rsi_indicator", "rsi_buy_condition", "value"))
        self.add_connection(NodeConnection("macd_indicator", "macd_buy_condition", "value"))
        self.add_connection(NodeConnection("rsi_buy_condition", "buy_and_gate", "input1"))
        self.add_connection(NodeConnection("macd_buy_condition", "buy_and_gate", "input2"))
        self.add_connection(NodeConnection("buy_and_gate", "buy_action", "trigger"))
        
    async def execute_strategy(self, dataframe: pd.DataFrame) -> Dict:
        """Execute the built strategy on data"""
        # Initialize node values
        for node in self.nodes.values():
            if node.node_type == NodeType.DATA_SOURCE:
                if node.parameters.get("source") == "close":
                    node.value = dataframe['close']
                elif node.parameters.get("source") == "volume":
                    node.value = dataframe['volume']
                    
        # Process nodes in topological order
        try:
            order = list(nx.topological_sort(self.graph))
            
            for node_id in order:
                node = self.nodes[node_id]
                
                if node.node_type == NodeType.INDICATOR:
                    node.output = await self._process_indicator(node)
                elif node.node_type == NodeType.CONDITION:
                    node.output = await self._process_condition(node)
                elif node.node_type == NodeType.LOGIC_GATE:
                    node.output = await self._process_logic_gate(node)
                elif node.node_type == NodeType.ACTION:
                    if node.output:  # If action is triggered
                        return {
                            'signal': node.parameters.get('action_type', '').replace('market_', ''),
                            'confidence': 0.8,
                            'strategy': 'node_based',
                            'node_id': node_id
                        }
                        
        except nx.NetworkXUnfeasible:
            return {'signal': 'hold', 'error': 'Cyclic graph'}
            
        return {'signal': 'hold'}
        
    async def _process_indicator(self, node: StrategyNode) -> Any:
        """Process indicator node"""
        if node.id == "rsi_indicator":
            return talib.RSI(node.value, timeperiod=node.parameters.get('period', 14))
        elif node.id == "macd_indicator":
            return talib.MACD(
                node.value,
                fastperiod=node.parameters.get('fast', 12),
                slowperiod=node.parameters.get('slow', 26),
                signalperiod=node.parameters.get('signal', 9)
            )
        return None
