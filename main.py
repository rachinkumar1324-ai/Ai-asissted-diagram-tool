import asyncio
import json
import base64
import os
import requests
import time
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# OpenAI API Configuration 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app = FastAPI(title="AI Collaborative Diagramming Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager 

class ConnectionManager:
    """Manages active WebSocket connections and broadcasting messages."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.drawing_history = [] 

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        await websocket.send_json({"type": "history", "data": self.drawing_history})

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                
                pass
    
    def add_to_history(self, message_data: dict):
        """Adds a message to history, handling cleanup and clear events."""
        msg_type = message_data.get("type")
        
        if msg_type == 'clear':
            self.drawing_history = []
        elif msg_type == 'cleanup':
           
            self.drawing_history = [message_data]
        else:
           
            self.drawing_history.append(message_data)

manager = ConnectionManager()


# WebSocket Endpoint for Real-time Drawing 

@app.websocket("/ws/drawing")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
           
            manager.add_to_history(message_data)
            await manager.broadcast(data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"User disconnected.")
    except Exception as e:
        print(f"An error occurred in websocket connection: {e}")
        manager.disconnect(websocket)


#  AI Cleanup HTTP Endpoint

class ImageRequest(BaseModel):
    image_data_url: str

@app.post("/api/cleanup")
async def ai_cleanup(request: ImageRequest):
    """
    Receives the canvas image data and calls the OpenAI Vision API to clean up the diagram.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OPENAI_API_KEY is missing. AI cleanup cannot proceed."
        )
    
    if not OPENAI_API_KEY.startswith("sk-"):
        raise HTTPException(
            status_code=401, 
            detail="Invalid API Key format. The key must start with 'sk-'. Please use a valid OpenAI API Key."
        )
    # --------------------------------

    try:
        full_data_url = request.image_data_url
        _, base64_data = full_data_url.split(',')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image data format.")
    
    system_prompt = (
        "You are an expert diagram interpreter. Analyze the messy hand-drawn diagram in the image. "
        "Your task is to convert it into a clean, structured JSON format suitable for programmatic rendering. "
        "The coordinates (x, y, width, height, radius, x1, y1, x2, y2) MUST be normalized between 0 and 1000, "
        "corresponding to the virtual canvas size. Infer standard shapes (rectangle, circle, line) and short descriptive text labels. "
        "Use a distinct, bright color (hex code) for each major element. "
        "RESPOND ONLY with the JSON object, do not include any explanatory text or markdown formatting (e.g., ```json)."
    )
    
    json_schema_description = """
[
    {
        "type": "rectangle" | "circle" | "line" | "text",
        "color": "Hex color code for the element (e.g., #1D4ED8).",
        "details": {
            "x": "X position (center for circle/text, top-left for rectangle, 0-1000).",
            "y": "Y position (center for circle/text, top-left for rectangle, 0-1000).",
            "width": "Width for rectangle (0-1000).",
            "height": "Height for rectangle (0-1000).",
            "radius": "Radius for circle (0-1000).",
            "x1": "Start X for line (0-1000).",
            "y1": "Start Y for line (0-1000).",
            "x2": "End X for line (0-1000).",
            "y2": "End Y for line (0-1000).",
            "text": "The text content (only for type 'text')."
        }
    }
]
"""
    
    user_prompt = (
        f"Clean up this diagram into a single JSON array structure that strictly adheres to the following schema. "
        f"Only output the JSON. Schema required:\n{json_schema_description}"
    )

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": full_data_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    max_retries = 5
    delay = 1.0

    for i in range(max_retries):
        try:
            # Use requests.post for the API call
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print("\n[AI CLEANUP ERROR]")
                print("OpenAI API Response Status:", response.status_code)
                print("OpenAI API Response Body:")
                print(response.text)
                print("-" * 50)

            response.raise_for_status()  
            
            result = response.json()
            
            if result.get("choices") and result["choices"][0]["message"]["content"]:
                json_string = result["choices"][0]["message"]["content"].strip()
                
                if json_string.startswith("```json"):
                    json_string = json_string.strip("```json").strip("```").strip()

                clean_diagram_data = json.loads(json_string)
                
                return {
                    "status": "success",
                    "message": "Diagram cleaned successfully by OpenAI.",
                    "data": clean_diagram_data
                }
            else:
                print("OpenAI API call failed to return content:", result)
                raise Exception("AI model returned no content.")
                
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1 and e.response is not None and e.response.status_code in (429, 500, 503):
                time.sleep(delay)
                delay *= 2  
            else:
                print(f"OpenAI API request failed permanently: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI API request failed: {e}")
        except json.JSONDecodeError:
            print(f"OpenAI returned non-JSON data or malformed JSON: {json_string[:200]}... Retrying if necessary.")
            if i < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise HTTPException(status_code=500, detail="AI model returned invalid JSON data.")
        except Exception as e:
            print(f"An unexpected error occurred during AI processing: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI processing: {e}")

    raise HTTPException(status_code=500, detail="AI cleanup failed after multiple retries.")



if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-YOUR_FULL_VALID"):
        print("\n" + "="*90)
        print("!!! WARNING: OPENAI_API_KEY is missing or is still a placeholder. The AI Cleanup endpoint will not work. !!!")
        print("!!! Please set the environment variable or replace the placeholder in the code. !!!")
        print("="*90 + "\n")
        
    print("Starting FastAPI server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
