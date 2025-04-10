from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import logging
import os
from dotenv import load_dotenv
import time
import uuid
import asyncio
from ai_service import ai_service
from agent_system import AgentSystem, AIProvider
from web_search_service import web_search_service

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the agent system
agent_system = AgentSystem(provider=AIProvider.GROQ)

app = FastAPI(title="TLDraw Chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()
        self.disconnecting_clients = set()  # Track clients that are in the process of disconnecting

    async def connect(self, websocket: WebSocket, client_id: str):
        try:
            await websocket.accept()
            async with self.lock:
                self.active_connections[client_id] = websocket
                if client_id in self.disconnecting_clients:
                    self.disconnecting_clients.remove(client_id)
            logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error during WebSocket connection: {e}")
            # No need to re-raise, just log the error

    async def disconnect(self, client_id: str):
        async with self.lock:
            # Add to disconnecting set to prevent repeated error messages
            self.disconnecting_clients.add(client_id)
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, client_id: str):
        async with self.lock:
            # Don't attempt to send messages to disconnecting clients
            if client_id in self.disconnecting_clients:
                logger.debug(f"Skipping send to disconnecting client {client_id}")
                return
                
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to client {client_id}: {e}")
                    # Mark as disconnecting and remove the connection
                    self.disconnecting_clients.add(client_id)
                    await self.disconnect(client_id)

    async def broadcast(self, message: str):
        # Create a copy of the keys to avoid modification during iteration
        async with self.lock:
            clients = list(self.active_connections.keys())
        
        for client_id in clients:
            # Skip clients that are disconnecting
            if client_id in self.disconnecting_clients:
                continue
            await self.send_personal_message(message, client_id)

manager = ConnectionManager()

# Pydantic models
class Message(BaseModel):
    id: str
    text: str
    sender: str
    timestamp: float
    mode: Optional[str] = "chat"

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    mode: Optional[str] = "chat"
    model: Optional[Dict[str, str]] = None

class ChatResponse(BaseModel):
    id: str
    text: str
    sender: str
    timestamp: float
    mode: Optional[str] = "chat"

class AgentRequest(BaseModel):
    message: str
    context: Optional[List[Dict[str, Any]]] = None
    user_id: Optional[str] = None
    model: Optional[Dict[str, str]] = None

class AgentResponse(BaseModel):
    id: str
    original_message: str
    enhanced_prompt: str
    research: str
    visualization: str
    instructions: str
    response: str
    timestamp: float

class ModelInfo(BaseModel):
    provider: str
    name: str
    displayName: str

# Available model configurations
AVAILABLE_MODELS = [
    ModelInfo(provider="groq", name="llama3-8b-8192", displayName="Groq - Llama 3 8B"),
    ModelInfo(provider="groq", name="llama3-70b-8192", displayName="Groq - Llama 3 70B"),
    ModelInfo(provider="gemini", name="gemini-2.0-flash-lite", displayName="Gemini 2.0 Pro"),
    ModelInfo(provider="gemini", name="gemini-2.0-flash", displayName="Gemini 2.0 Flash"),
]

# Simple in-memory message store
message_history: List[Message] = []

# Get message history context for AI
def get_message_context(max_messages: int = 15):
    # Get the most recent N messages for context
    return [msg.model_dump() for msg in message_history[-max_messages:]] if message_history else []

# AI response function using our service
async def get_ai_response(message: str, model_info: Optional[Dict[str, str]] = None) -> str:
    try:
        context = get_message_context()
        
        # Use model info if provided
        if model_info and isinstance(model_info, dict):
            provider = model_info.get("provider")
            model_name = model_info.get("name")
            
            if provider and model_name:
                return await ai_service.get_response(
                    message, 
                    context, 
                    provider=provider, 
                    model=model_name
                )
        
        # Default fallback to Groq
        return await ai_service.get_response(message, context, provider="groq", model="llama3-8b-8192")
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return "I'm having trouble processing your request. Please try again."

# Process message and generate AI response asynchronously
async def process_message(client_id: str, user_message: Message, model_info: Optional[Dict[str, str]] = None):
    try:
        # Add user message to history
        message_history.append(user_message)
        
        # Broadcast user message to all clients
        await manager.broadcast(json.dumps(user_message.model_dump()))
        
        # Send processing status update
        if user_message.mode == "create":
            status_message = {
                "type": "processing_update",
                "status": "Processing your creation request"
            }
            await manager.send_personal_message(json.dumps(status_message), client_id)
        
        # Generate AI response
        ai_response_text = await get_ai_response(user_message.text, model_info)
        ai_message = Message(
            id=str(uuid.uuid4()),
            text=ai_response_text,
            sender="assistant",
            timestamp=time.time(),
            mode=user_message.mode
        )
        
        # Add AI response to history
        message_history.append(ai_message)
        
        # Broadcast AI response
        await manager.broadcast(json.dumps(ai_message.model_dump()))
    except Exception as e:
        logger.error(f"Error processing message from client {client_id}: {e}")
        # Send error message to client
        error_message = Message(
            id=str(uuid.uuid4()),
            text="Sorry, I encountered an error processing your message. Please try again.",
            sender="assistant",
            timestamp=time.time(),
            mode=getattr(user_message, "mode", "chat")
        )
        await manager.send_personal_message(json.dumps(error_message.model_dump()), client_id)

# WebSocket endpoint
@app.websocket("/ws/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    ping_task = None
    
    try:
        # Start ping task to keep connection alive
        ping_task = asyncio.create_task(websocket_ping(websocket))
        
        # Send message history to the new client
        if message_history:
            history_json = json.dumps([msg.model_dump() for msg in message_history])
            await manager.send_personal_message(history_json, client_id)
        
        # Send available models to the client
        models_message = {
            "type": "model_list",
            "models": [model.model_dump() for model in AVAILABLE_MODELS]
        }
        await manager.send_personal_message(json.dumps(models_message), client_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message_data = json.loads(data)
                    
                    # Check if this is a pong response
                    if message_data.get("type") == "pong":
                        logger.debug(f"Received pong from client {client_id}")
                        continue
                        
                    mode = message_data.get("mode", "chat")
                    model_info = message_data.get("model")
                    
                    user_message = Message(
                        id=message_data.get("id", str(uuid.uuid4())),
                        text=message_data.get("text", ""),
                        sender=message_data.get("sender", "user"),
                        timestamp=message_data.get("timestamp", time.time()),
                        mode=mode
                    )
                    
                    # Process message in the background
                    asyncio.create_task(process_message(client_id, user_message, model_info))
                    
                except json.JSONDecodeError:
                    await manager.send_personal_message(f"Invalid JSON: {data}", client_id)
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for client {client_id}")
                await manager.disconnect(client_id)
                break
            except Exception as e:
                logger.error(f"Error receiving message from client {client_id}: {e}")
                if "disconnect" in str(e).lower():
                    logger.info(f"Disconnect detected for client {client_id}")
                    await manager.disconnect(client_id)
                    break
                # Continue the loop for other types of errors
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unhandled WebSocket error for client {client_id}: {e}")
        await manager.disconnect(client_id)
    finally:
        # Cleanup ping task if it exists
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

# REST endpoint for getting message history
@app.get("/messages", response_model=List[Message])
async def get_messages():
    return message_history

# REST endpoint for getting available models
@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    return AVAILABLE_MODELS

# REST endpoint for sending a message
@app.post("/messages", response_model=List[ChatResponse])
async def send_message(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # Create user message
        user_message = Message(
            id=str(uuid.uuid4()),
            text=request.message,
            sender="user",
            timestamp=time.time(),
            mode=request.mode
        )
        
        # Add message to history
        message_history.append(user_message)
        
        # Generate AI response
        ai_response_text = await get_ai_response(request.message, request.model)
        ai_message = Message(
            id=str(uuid.uuid4()),
            text=ai_response_text,
            sender="assistant",
            timestamp=time.time(),
            mode=request.mode
        )
        
        # Add AI response to history
        message_history.append(ai_message)
        
        # If a WebSocket client ID was provided, broadcast these messages
        if request.user_id:
            background_tasks.add_task(
                manager.broadcast, 
                json.dumps(user_message.model_dump())
            )
            background_tasks.add_task(
                manager.broadcast,
                json.dumps(ai_message.model_dump())
            )
        
        return [
            ChatResponse(**user_message.model_dump()),
            ChatResponse(**ai_message.model_dump())
        ]
    except Exception as e:
        logger.error(f"Error in send_message endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing message")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "ok", "timestamp": time.time()}

# Debug endpoint for connection diagnostics
@app.get("/debug/connections")
async def debug_connections():
    """Debug endpoint to check active connections and settings"""
    try:
        # Get CORS settings safely
        cors_origins = []
        for middleware in app.user_middleware:
            if hasattr(middleware, "cls") and middleware.cls == CORSMiddleware:
                if hasattr(middleware, "options"):
                    cors_origins = middleware.options.get("allow_origins", [])
                break
        
        connections_info = {
            "active_connections": len(manager.active_connections),
            "active_connection_ids": list(manager.active_connections.keys()),
            "disconnecting_clients": list(manager.disconnecting_clients),
            "server_settings": {
                "host": os.getenv("HOST", "127.0.0.1"),
                "port": os.getenv("PORT", "8000"),
                "cors_origins": cors_origins
            },
            "message_history_count": len(message_history),
            "timestamp": time.time()
        }
        return connections_info
    except Exception as e:
        logger.error(f"Error in debug_connections endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ping-pong endpoint for basic connectivity testing
@app.get("/ping")
async def ping():
    """Simple ping-pong endpoint for connection testing"""
    return {"ping": "pong", "timestamp": time.time()}

# WebSocket ping handler to keep connections alive
async def websocket_ping(websocket: WebSocket, interval: int = 30):
    """Send ping messages to keep the WebSocket connection alive"""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
            except Exception as e:
                logger.warning(f"Failed to send ping: {e}")
                break
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in websocket_ping: {e}")
    finally:
        logger.debug("Ping task stopped")

# Agent system endpoints
@app.post("/agent/process", response_model=AgentResponse)
async def process_with_agents(request: AgentRequest):
    """Process a message through the agent system"""
    try:
        # Process message through agent system
        result = await agent_system.process_message(request.message)
        
        # Format the response
        response = AgentResponse(
            id=str(uuid.uuid4()),
            original_message=result["original_message"],
            enhanced_prompt=result["enhanced_prompt"],
            research=result["research"],
            visualization=result["visualization"],
            instructions=result["instructions"],
            response=result["response"],
            timestamp=time.time()
        )
        
        return response
    except Exception as e:
        logger.error(f"Error processing message with agent system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent WebSocket endpoint for real-time interaction
@app.websocket("/ws/agent/{client_id}")
async def agent_websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    ping_task = None
    
    try:
        # Start ping task to keep connection alive
        ping_task = asyncio.create_task(websocket_ping(websocket))
        
        # Send message history to the new client
        if message_history:
            history_json = json.dumps([msg.model_dump() for msg in message_history])
            await manager.send_personal_message(history_json, client_id)
        
        # Send available models to the client
        models_message = {
            "type": "model_list",
            "models": [model.model_dump() for model in AVAILABLE_MODELS]
        }
        await manager.send_personal_message(json.dumps(models_message), client_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message_data = json.loads(data)
                    
                    # Check if this is a pong response
                    if message_data.get("type") == "pong":
                        logger.debug(f"Received pong from client {client_id} in agent websocket")
                        continue
                        
                    user_message_text = message_data.get("text", "")
                    mode = message_data.get("mode", "chat")
                    model_info = message_data.get("model")
                    
                    # Create user message object for response
                    user_msg = Message(
                        id=message_data.get("id", str(uuid.uuid4())),
                        text=user_message_text,
                        sender="user",
                        timestamp=time.time(),
                        mode=mode
                    )
                    
                    # Add to message history
                    message_history.append(user_msg)
                    
                    # Broadcast user message
                    await manager.broadcast(json.dumps(user_msg.model_dump()))
                    
                    # Send processing status if in create mode
                    if mode == "create":
                        processing_steps = [
                            "Enhancing your prompt",
                            "Researching your request",
                            "Generating visualization ideas",
                            "Creating drawing instructions"
                        ]
                        
                        for step in processing_steps:
                            status_message = {
                                "type": "processing_update",
                                "status": step
                            }
                            await manager.send_personal_message(json.dumps(status_message), client_id)
                            await asyncio.sleep(1)  # Slight delay between steps
                    
                    # Process with agent system, with error handling
                    try:
                        # Configure agent_system based on model selection if in create mode
                        if mode == "create" and model_info:
                            # We don't need to modify agent system attributes directly anymore
                            # But we do need to log what model is being used
                            provider_name = model_info.get("provider", "groq")
                            model_name = model_info.get("name", "llama3-70b-8192")
                            logger.info(f"Using provider: {provider_name}, model: {model_name} for create mode")
                        
                        # Pass the user message text to the agent system, not the Message object
                        result = await agent_system.process_message(user_message_text)
                        logger.info("Agent processing completed successfully")
                    except Exception as agent_error:
                        logger.error(f"Agent system error: {agent_error}")
                        # Create fallback result with error information
                        result = {
                            "original_message": user_message_text,
                            "enhanced_prompt": f"Error processing prompt: {str(agent_error)}",
                            "research": "Could not complete research due to an error.",
                            "visualization": "Could not generate visualization due to an error.",
                            "instructions": "Could not generate instructions due to an error.",
                            "response": "I'm sorry, but I encountered an error processing your request. The system may be experiencing issues with one or more AI providers. Please try again later."
                        }
                    
                    # Create AI message from agent response
                    ai_message = Message(
                        id=str(uuid.uuid4()),
                        text=result["response"],
                        sender="assistant",
                        timestamp=time.time(),
                        mode=mode
                    )
                    
                    # Add to message history
                    message_history.append(ai_message)
                    
                    # Send detailed agent result to the requesting client
                    detailed_response = {
                        "type": "agent_response",
                        "id": ai_message.id,
                        "original_message": result["original_message"],
                        "enhanced_prompt": result["enhanced_prompt"],
                        "research": result["research"],
                        "visualization": result["visualization"],
                        "instructions": result["instructions"],
                        "response": result["response"],
                        "timestamp": ai_message.timestamp,
                        "mode": mode
                    }
                    
                    # Log any errors in the response
                    for key, value in result.items():
                        if isinstance(value, str) and "Error" in value:
                            logger.warning(f"Error in {key}: {value[:100]}...")
                    
                    await manager.send_personal_message(json.dumps(detailed_response), client_id)
                    
                    # Broadcast simplified AI message to all clients
                    await manager.broadcast(json.dumps(ai_message.model_dump()))
                    
                except json.JSONDecodeError:
                    await manager.send_personal_message(f"Invalid JSON: {data}", client_id)
            except WebSocketDisconnect:
                logger.info(f"Agent WebSocket disconnected for client {client_id}")
                await manager.disconnect(client_id)
                break
            except Exception as e:
                logger.error(f"Error receiving message in agent websocket for client {client_id}: {e}")
                if "disconnect" in str(e).lower():
                    logger.info(f"Disconnect detected for client {client_id} in agent websocket")
                    await manager.disconnect(client_id)
                    break
                # Continue the loop for other errors
                continue
                
    except WebSocketDisconnect:
        logger.info(f"Agent WebSocket disconnected for client {client_id}")
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unhandled error in agent websocket for client {client_id}: {e}")
        await manager.disconnect(client_id)
    finally:
        # Cleanup ping task if it exists
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

# Debug endpoint for API keys
@app.get("/debug/api-keys")
async def debug_api_keys():
    # Function to mask API keys for debugging
    def mask_key(key):
        if not key:
            return "Not set"
        return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "Set but too short"
    
    return {
        "openai": mask_key(os.getenv("OPENAI_API_KEY")),
        "groq": mask_key(os.getenv("GROQ_API_KEY")),
        "gemini": mask_key(os.getenv("GEMINI_API_KEY")),
        "tavily": mask_key(os.getenv("TAVILY_API_KEY"))
    }

class WebSearchRequest(BaseModel):
    query: str
    include_images: bool = True
    search_depth: str = "basic"

@app.post("/web-search")
async def perform_web_search(request: WebSearchRequest):
    """
    Perform a web search using the Tavily API
    """
    try:
        # Check if web search service is available
        if not web_search_service.is_available():
            return JSONResponse(
                status_code=503,
                content={"error": "Web search service is not available - TAVILY_API_KEY not configured"}
            )
            
        # Perform the search
        search_results = await web_search_service.search(
            query=request.query,
            search_depth=request.search_depth,
            include_images=request.include_images
        )
        
        return search_results
    except Exception as e:
        logger.error(f"Error in web search endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Web search failed: {str(e)}"}
        )

class DiagramSearchRequest(BaseModel):
    topic: str

@app.post("/diagram-search")
async def search_diagram_information(request: DiagramSearchRequest):
    """
    Search for diagram-specific information using the Tavily API
    """
    try:
        # Check if web search service is available
        if not web_search_service.is_available():
            return JSONResponse(
                status_code=503,
                content={"error": "Web search service is not available - TAVILY_API_KEY not configured"}
            )
            
        # Perform specialized diagram search
        search_results = await web_search_service.search_for_diagram(request.topic)
        
        return search_results
    except Exception as e:
        logger.error(f"Error in diagram search endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Diagram search failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 