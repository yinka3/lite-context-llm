from datetime import datetime
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import pytz
from _types import EventData
from update_mem import History
from google import genai
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("SECRET_KEY")
app = FastAPI()
model = genai.Client(api_key=key)


class ConnectionManager:
    def __init__(self):
        self.active_connection = None 
        self.history = History()      
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        if not hasattr(self.history, 'initialized'):
            await self.history.initialize()
            self.history.initialized = True
    
    async def add_message_to_history(self, message: str, role: str):
        """Simple history addition without locks"""
        eastern = pytz.timezone('US/Eastern')
        now_time = datetime.now(eastern)
        event_obj = EventData(role=role, timestamp=now_time, message=message)
        
        try:
            await self.history.add_to_history(event_obj, None)
        except Exception as e:
            print(f"History update failed: {e}")
            # Continue anyway for demo
    
    def disconnect(self):
        self.active_connection = None


manager = ConnectionManager()

@app.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    chat = model.chats.create(model="gemini-2.5-flash")
    
    try:
        while True:
            user_message = await websocket.receive_text()
            
            if user_message:
                await manager.add_message_to_history(user_message, "user")

                try:
                    response = chat.send_message(message=user_message)
                    ai_response = response.text
                    
                    await manager.add_message_to_history(ai_response, "AI")
                    
                    await websocket.send_text(ai_response)
                    
                except Exception as e:
                    error_msg = f"AI error: {e}"
                    await websocket.send_text(error_msg)
                    
    except WebSocketDisconnect:
        print("Client disconnected")
        manager.disconnect()
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect()