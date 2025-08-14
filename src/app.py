import asyncio
from datetime import datetime, timedelta
import os
import uvicorn
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import pytz
from google import genai
from dotenv import load_dotenv
from update_mem import History
from _types import EventData
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
client = genai.Client(api_key=key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connection = None 
        self.history: History = History()

        self.rant_mode: bool = False
        self.rant_messages: List[Dict] = []
        self.rant_start_time: Optional[datetime] = None
        self.rant_time: Optional[timedelta] = None
        self.last_rant_duration: float = 0
        self.rant_context: List[Dict] = None
        
        self.rant_sessions: List[Dict] = []
        self.cleanup_task: Optional[asyncio.Task] = None
        self.start_time = datetime.now(pytz.timezone('US/Eastern'))

        self.model_name: str = "gemini-2.0-flash-001"
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        if not hasattr(self.history, 'initialized'):
            self.history.initialize()
            self.history.initialized = True
    
    def add_message_to_history(self, message: str, role: str):
        """Add message to permanent history"""
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(tz=eastern)
        event_obj = EventData(role=role, timestamp=now_time, message=message)
        
        try:
            self.history.add_to_history(event_obj)
        except Exception as e:
            print(f"History update failed: {e}")

    def get_conversation_context(self, num_message: int = 10):
        context = []

        recent_nodes = self.history.history[-num_message:] if len(self.history.history) > num_message else self.history.history

        for node in recent_nodes:
            context.append({
                "role": "user" if node.data.role.lower() == "user" else "model",
                "parts": [{"text": node.data.message}]
            })
        
        return context
    
    def start_rant_mode(self):
        """Rant mode activated, go crazy"""

        self.rant_mode = True
        self.rant_start_time = datetime.now(pytz.timezone('US/Eastern'))
        self.rant_context = self.get_conversation_context(10)

        print(f"Rant mode started at {self.rant_start_time}")

        if self.cleanup_task:
            self.cleanup_task.cancel()
        
    def end_rant_mode(self):
        """Exit rant mode and start cleaning up"""

        if not self.rant_mode:
            return
        
        self.rant_mode = False

        summary = self.generate_rant_summary()

        if summary:
            self.rant_sessions.append(summary)
        
        self.last_rant_duration = (datetime.now(pytz.timezone('US/Eastern')) - self.rant_start_time).total_seconds()
        print(f"Rant mode ended. You babbled for {self.last_rant_duration}")

        self.cleanup_task = asyncio.create_task(self.cleanup_rant_data())


    def generate_rant_summary(self):
        """Generate a brief summary of the rant session"""
        if not self.rant_messages:
            return "Brief rant with no significant content"
        
        try:
            rant_content = "\n".join([
                f"{msg['role']}: {msg['parts'][0]['text']}" 
                for msg in self.rant_messages
            ])
            
            summary_prompt = f"""Summarize this rant conversation in 1-2 sentences. 
            Focus on the main topic or emotion expressed:
            
            {rant_content}
            
            Summary:"""
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": summary_prompt}]}]
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Failed to generate rant summary: {e}")
            return f"Rant session with {len(self.rant_messages)} messages"

    async def process_message(self, user_message: str, websocket: WebSocket):
        """Process incoming message based on mode"""
        
        if user_message.lower() == "/rant":
            self.start_rant_mode()
            await websocket.send_json({
                "type": "system",
                "message": "Rant mode activated! Feel free to vent. Type /done when finished."
            })
            return
        
        elif user_message.lower() == "/done" and self.rant_mode:
            self.end_rant_mode()
            await websocket.send_json({
                "type": "system", 
                "message": "Rant mode ended. Returning to normal conversation."
            })
            return
        
        if self.rant_mode:
            await self.process_rant_message(user_message, websocket)
        else:
            await self.process_normal_message(user_message, websocket)
        
    async def process_normal_message(self, user_message: str, websocket: WebSocket):
        """Process message in normal mode using manual history"""
        
        if len(self.history.history) >= History.MAX_CAPACITY:
            self.history.storage._save_to_disk()
            await websocket.send_json({
                "type": "system",
                "message": f"Memory capacity reached ({History.MAX_CAPACITY} messages). Please export your history or start a new session.",
                "action_required": True,
                "options": ["export_history", "start_new_session"]
            })
            return
        
        # Now safe to add
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(tz=eastern)
        event_obj = EventData(role="user", timestamp=now_time, message=user_message)
        self.history.add_to_history(event_obj)
            
        try:
            relevant_contexts = self.history.get_relevant_context(user_message, max_results=5)
            
            conversation_history = []
            
            for ctx in relevant_contexts:
                if ctx['conversation_thread']:
                    for msg in ctx['conversation_thread']:
                        conversation_history.append({
                            "role": "user" if msg['role'] == "user" else "model",
                            "parts": [{"text": msg['message']}]
                        })
            
            recent_context = self.get_conversation_context(15)
            conversation_history.extend(recent_context)
            
            conversation_history.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })
            
            ai_response = client.models.generate_content(
                model=self.model_name,
                contents=conversation_history
            )

            await websocket.send_json({
                "type": "message",
                "message": ai_response.text
            })
            return
        
        except Exception as e:
            error_msg = f"Rant AI error: {e}"
            await websocket.send_json({
                "type": "error",
                "message": error_msg
            })
    
    async def process_rant_message(self, user_message: str, websocket: WebSocket):
        """Process message in rant mode - no permanent storage"""
        
        try:
            relevant_contexts = self.history.get_relevant_context(user_message, max_results=5)
            
            rant_conversation = []
            
            for ctx in relevant_contexts:
                if ctx['conversation_thread']:
                    for msg in ctx['conversation_thread']:
                        rant_conversation.append({
                            "role": "user" if msg['role'] == "user" else "model",
                            "parts": [{"text": msg['message']}]
                        })
            
            rant_conversation.extend(self.rant_context)
            
            rant_conversation.extend(self.rant_messages)
            
            rant_conversation.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })
            
            self.rant_messages.append({
                "role": "user",
                "parts": [{"text": user_message}]
            })
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=rant_conversation
            )
            
            ai_response = response.text
            
            self.rant_messages.append({
                "role": "model",
                "parts": [{"text": ai_response}]
            })
            
            await websocket.send_json({
                "type": "rant_message",
                "message": ai_response
            })
            return
        except Exception as e:
            error_msg = f"Rant AI error: {e}"
            await websocket.send_json({
                "type": "error",
                "message": error_msg
            })

    async def cleanup_rant_data(self):
        """Clean up rant data after 5 minutes"""
        await asyncio.sleep(300)
        
        self.rant_messages = []
        self.rant_context = []
        print("Rant data cleaned up")
    
    def is_cleaning(self):
        """Check if cleanup task is currently running"""
        return (
            self.cleanup_task is not None and 
            not self.cleanup_task.done()
        )
    
    def get_cleanup_status(self):
        """Get detailed cleanup status"""
        if not self.cleanup_task:
            return {"cleaning": False, "time_remaining": 0}
        
        if self.cleanup_task.done():
            return {"cleaning": False, "time_remaining": 0}
        
        # Calculate time remaining (5 minutes - elapsed time)
        if self.rant_start_time:
            elapsed = (datetime.now(pytz.timezone('US/Eastern')) - self.rant_start_time).total_seconds()
            time_remaining = max(0, 300 - elapsed)
            return {
                "cleaning": True,
                "time_remaining": time_remaining,
                "stage": "waiting"
            }
        
        return {"cleaning": True, "time_remaining": 0}
    
    def get_memory_warning_level(self, percentage):
        if percentage >= 99:
            return "CRITICAL"
        elif percentage >= 95:
            return "HIGH"  
        elif percentage >= 90:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_memory_stats(self):
        """Get memory statistics"""
        current_time = datetime.now(pytz.timezone('US/Eastern'))
        uptime_seconds = (current_time - self.start_time).total_seconds()
        
        # Get connection count from history
        connection_count = sum(
            len(node.children) 
            for node in self.history.context_nodes.context_graph.values() 
            if node is not None
        )
        
        # Calculate memory usage
        total_messages = len(self.history.history)
        max_messages = History.MAX_CAPACITY 
        percentage = (total_messages / max_messages) * 100
        limit_level = self.get_memory_warning_level(percentage)

        return {
            "total_messages": total_messages,
            "user_messages": self.history._user_event_cnt,
            "ai_messages": total_messages - self.history._user_event_cnt,
            "memory_usage": f"{total_messages}/{max_messages}",
            "memory_percentage": percentage,
            "limit": limit_level,
            "root_nodes": len(self.history.root_nodes),
            "connections": connection_count,
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": self._format_uptime(uptime_seconds),
            "model_name": self.model_name
        }

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
      
    def disconnect(self):
        self.active_connection = None
        if self.cleanup_task:
            self.cleanup_task.cancel()
        self.history.storage._save_to_disk()


manager = ConnectionManager()

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    with open("frontend.html", "r") as f:
        return f.read()

@app.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            user_message = await websocket.receive_text()
            
            if user_message:
                await manager.process_message(user_message, websocket)
                    
    except WebSocketDisconnect:
        print("Client disconnected")
        manager.disconnect()
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect()


@app.get("/")
async def root():
    return {"message": "WebSocket server with rant mode is running"}


@app.get("/stats")
async def get_full_stats():
    return manager.get_memory_stats()

if __name__ == "__main__":
    print("Starting AI Memory Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)