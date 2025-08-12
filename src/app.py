import asyncio
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import pytz
from google import genai
from dotenv import load_dotenv
from update_mem import History
from _types import EventData

load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
client = genai.Client(api_key=key)


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

        self.model_name: str = "gemini-2.0-flash-001"
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        if not hasattr(self.history, 'initialized'):
            await self.history.initialize()
            self.history.initialized = True
    
    async def add_message_to_history(self, message: str, role: str):
        """Add message to permanent history"""
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(tz=eastern)
        event_obj = EventData(role=role, timestamp=now_time, message=message)
        
        try:
            await self.history.add_to_history(event_obj, None)
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
    
    async def start_rant_mode(self):
        """Rant mode activated, go crazy"""

        self.rant_mode = True
        self.rant_start_time = datetime.now(pytz.timezone('US/Eastern'))
        self.rant_context = self.get_conversation_context(10)

        print(f"Rant mode started at {self.rant_start_time}")

        if self.cleanup_task:
            self.cleanup_task.cancel()
        
    async def end_rant_mode(self):
        """Exit rant mode and start cleaning up"""

        if not self.rant_mode:
            return
        
        self.rant_mode = False

        summary = await self.generate_rant_summary()

        if summary:
            self.rant_sessions.append(summary)
        
        self.last_rant_duration = (datetime.now(pytz.timezone('US/Eastern')) - self.rant_start_time).total_seconds()
        print(f"Rant mode ended. You babbled for {self.last_rant_duration}")

        self.cleanup_task = asyncio.create_task(self.cleanup_rant_data())


    async def generate_rant_summary(self):
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
            await self.start_rant_mode()
            await websocket.send_json({
                "type": "system",
                "message": "Rant mode activated! Feel free to vent. Type /done when finished."
            })
            return
        
        elif user_message.lower() == "/done" and self.rant_mode:
            await self.end_rant_mode()
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
        
        await self.add_message_to_history(user_message, "user")
    
        try:
            relevant_contexts = await self.history.get_relevant_context(user_message, max_results=5)
            
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
            relevant_contexts = await self.history.get_relevant_context(user_message, max_results=5)
            
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
            time_remaining = max(0, 300 - elapsed)  # 300 seconds = 5 minutes
            return {
                "cleaning": True,
                "time_remaining": time_remaining,
                "stage": "waiting"  # could be "waiting" or "deleting"
            }
        
        return {"cleaning": True, "time_remaining": 0}
      
    def disconnect(self):
        self.active_connection = None
        if self.cleanup_task:
            self.cleanup_task.cancel()


manager = ConnectionManager()


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


@app.get("/status")
async def get_status():
    return {
        "rant_mode": manager.rant_mode,
        "cleaning": manager.get_cleanup_status(),
        "total_messages": len(manager.history.history),
        "rant_messages": len(manager.rant_messages) if manager.rant_mode else 0
    }