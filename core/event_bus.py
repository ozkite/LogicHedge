"""
Event-driven architecture for trading system.
Inspired by NoFx and Hummingbot patterns.
"""

import asyncio
from typing import Any, Callable, Dict, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Base event class for all system events"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = None
    source: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class EventBus:
    """Asynchronous event bus for system communication"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to specific event types"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        logger.debug(f"Subscribed {callback.__name__} to {event_type}")
        
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event types"""
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)
            
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        await self._queue.put(event)
        
    async def _process_events(self):
        """Process events from queue"""
        while self._running:
            try:
                event = await self._queue.get()
                await self._dispatch(event)
                self._queue.task_done()
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                
    async def _dispatch(self, event: Event):
        """Dispatch event to subscribers"""
        if event.event_type in self._listeners:
            for callback in self._listeners[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler {callback.__name__}: {e}")
                    
    async def start(self):
        """Start event bus"""
        self._running = True
        asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        
    async def stop(self):
        """Stop event bus"""
        self._running = False
        await self._queue.join()
        logger.info("Event bus stopped")
