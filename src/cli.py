import asyncio
import websockets
import aiohttp
from rich.console import Console
from rich.live import Live
from rich.text import Text
import json
from threading import Thread
import queue

class MemoryChatCLI:
    def __init__(self):
        self.console = Console()
        self.memory_stats = {}
        self.messages = []
        self.input_queue = queue.Queue()
        
    def create_display(self):
        """Create ultra-minimal display"""
        display = Text()
        
        # Get stats
        memory_pct = self.memory_stats.get('memory_percentage', 0)
        status = self.memory_stats.get('limit', 'LOW')
        mode = 'RANT' if self.memory_stats.get('rant_mode', False) else 'NORMAL'
        model = self.memory_stats.get('model_name', 'Gemini')
        
        # Color based on status
        color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "orange1", "CRITICAL": "red"}[status]
        
        # Simple progress bar
        filled = int(memory_pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        
        # One-line status
        display.append(f"[{bar}] {memory_pct:.1f}% ", style=color)
        display.append(f"{status} ", style=color)
        display.append("│ ", style="dim")
        display.append(f"{mode} ", style="magenta" if mode == "RANT" else "white")
        display.append("│ ", style="dim")
        display.append(f"{model}\n\n", style="cyan")
        
        # Last 10 messages
        for msg in self.messages[-10:]:
            if msg.startswith("[You]"):
                display.append(msg[5:] + "\n", style="bright_white")
            elif msg.startswith("[AI]"):
                display.append(msg[4:] + "\n", style="green")
            elif msg.startswith("[SYSTEM]"):
                display.append(msg[9:] + "\n", style="yellow")
            display.append("\n")
        
        display.append("> ", style="cyan bold")
        
        return display
    
    def input_thread(self):
        """Input handler"""
        while True:
            try:
                message = input()
                self.input_queue.put(message)
                if message.lower() == 'exit':
                    break
            except:
                break
    
    async def update_stats(self):
        """Update stats"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get('http://127.0.0.1:8000/status') as resp:
                        self.memory_stats = await resp.json()
                except:
                    pass
                await asyncio.sleep(5)
    
    async def receive_messages(self, websocket):
        """Receive messages"""
        while True:
            try:
                data = json.loads(await websocket.recv())
                msg_type = data.get('type', 'message')
                msg_text = data.get('message', '')
                
                if msg_type == 'system':
                    self.messages.append(f"[SYSTEM] {msg_text}")
                else:
                    self.messages.append(f"[AI] {msg_text}")
            except:
                break
    
    async def send_messages(self, websocket):
        """Send messages"""
        while True:
            if not self.input_queue.empty():
                message = self.input_queue.get()
                if message.lower() == 'exit':
                    break
                await websocket.send(message)
                self.messages.append(f"[You] {message}")
            await asyncio.sleep(0.1)
    
    async def run(self):
        """Main loop"""
        self.console.clear()
        
        async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
            Thread(target=self.input_thread, daemon=True).start()
            tasks = [
                asyncio.create_task(self.update_stats()),
                asyncio.create_task(self.receive_messages(ws)),
                asyncio.create_task(self.send_messages(ws))
            ]
            
            # Display loop
            with Live(self.create_display(), refresh_per_second=2, console=self.console) as live:
                while True:
                    live.update(self.create_display())
                    if not self.input_queue.empty() and self.input_queue.queue[0].lower() == 'exit':
                        break
                    await asyncio.sleep(0.5)
            
            for task in tasks:
                task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(MemoryChatCLI().run())
    except KeyboardInterrupt:
        print("\nBye!")