import os
import json
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
import dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        # Initialize OpenAI client with API key
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize alternative API clients if available
        self.use_groq = os.getenv("GROQ_API_KEY") is not None
        self.use_gemini = os.getenv("GEMINI_API_KEY") is not None
        
        if self.use_groq:
            from groq import AsyncGroq
            self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
            logger.info("Groq API initialized")
        
        if self.use_gemini:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini = genai
            logger.info("Gemini API initialized")
        
        # Default model configurations
        self.default_model = "llama3-8b-8192"
        self.default_provider = "groq"
        
        # Define available models
        self.available_models = self._get_available_models()
        
        logger.info("AI Service initialized")
    
    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get a list of available AI models from all providers"""
        models = []
        
        # OpenAI models
        if os.getenv("OPENAI_API_KEY"):
            models.extend([
                {
                    "provider": "openai",
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "description": "OpenAI's most advanced model, optimized for both text and vision tasks"
                },
                {
                    "provider": "openai",
                    "id": "gpt-4o-mini",
                    "name": "GPT-4o Mini",
                    "description": "Smaller, faster, and more affordable version of GPT-4o"
                },
                {
                    "provider": "openai",
                    "id": "gpt-3.5-turbo",
                    "name": "GPT-3.5 Turbo",
                    "description": "Fast, economical model for chat applications"
                }
            ])
        
        # Groq models
        if self.use_groq:
            models.extend([
                {
                    "provider": "groq",
                    "id": "llama3-70b-8192",
                    "name": "Llama-3 70B",
                    "description": "Large Llama 3 model with high performance on Groq's platform"
                },
                {
                    "provider": "groq",
                    "id": "llama3-8b-8192",
                    "name": "Llama-3 8B",
                    "description": "Smaller Llama 3 model with faster inference on Groq's platform"
                },
                {
                    "provider": "groq",
                    "id": "mixtral-8x7b-32768",
                    "name": "Mixtral 8x7B",
                    "description": "Powerful mixture-of-experts model with longer context window"
                }
            ])
        
        # Gemini models
        if self.use_gemini:
            models.extend([
                {
                    "provider": "gemini",
                    "id": "gemini-2.0-pro",
                    "name": "Gemini 2.0 Pro",
                    "description": "Google's advanced multimodal model with improved reasoning capabilities"
                },
                {
                    "provider": "gemini",
                    "id": "gemini-2.0-flash",
                    "name": "Gemini 2.0 Flash",
                    "description": "Faster, more efficient Gemini 2.0 model for quicker responses"
                }
            ])
        
        return models
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return self.available_models
    
    async def get_response(self, 
                          message: str, 
                          context: List[Dict[str, Any]], 
                          provider: Optional[str] = None,
                          model: Optional[str] = None) -> str:
        """
        Get a response from the AI model
        
        Args:
            message: The user's message
            context: Previous messages for context
            provider: Optional provider name (openai, groq, gemini)
            model: Optional model name
            
        Returns:
            The AI response as a string
        """
        try:
            # Select provider and model based on parameters or defaults
            selected_provider = provider or self.default_provider
            selected_model = model or self.default_model
            
            # Log the selected provider and model
            logger.info(f"Using provider: {selected_provider}, model: {selected_model}")
            
            # Format context for the selected provider
            formatted_messages = self._format_context(context, message, selected_provider)
            
            # Call the appropriate provider's API
            if selected_provider == "openai":
                return await self._get_openai_response(formatted_messages, selected_model)
            elif selected_provider == "groq" and self.use_groq:
                return await self._get_groq_response(formatted_messages, selected_model)
            elif selected_provider == "gemini" and self.use_gemini:
                return await self._get_gemini_response(formatted_messages, selected_model)
            else:
                # Fallback to OpenAI if the requested provider is not available
                logger.warning(f"Provider {selected_provider} not available, falling back to OpenAI")
                return await self._get_openai_response(formatted_messages, self.default_model)
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return f"I'm sorry, but I encountered an error: {str(e)}"
    
    def _format_context(self, context: List[Dict[str, Any]], message: str, provider: str) -> Union[List[Dict[str, Any]], str]:
        """Format context based on the provider's API requirements"""
        
        if provider == "gemini":
            # For Gemini, we need to format as a chat history
            gemini_messages = []
            for msg in context:
                role = "user" if msg["sender"] == "user" else "model"
                gemini_messages.append({"role": role, "parts": [msg["text"]]})
            
            # Add the current message
            gemini_messages.append({"role": "user", "parts": [message]})
            return gemini_messages
        
        else:
            # For OpenAI and Groq, format as ChatCompletion messages
            messages = []
            
            # System message to define the assistant's behavior
            messages.append({
                "role": "system",
                "content": "You are a helpful, creative, and friendly AI assistant. Respond concisely and directly to the user's questions."
            })
            
            # Add context messages
            for msg in context:
                role = "user" if msg["sender"] == "user" else "assistant"
                messages.append({
                    "role": role,
                    "content": msg["text"]
                })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            return messages
    
    async def _get_openai_response(self, messages: List[Dict[str, Any]], model: str) -> str:
        """Get response from OpenAI API"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error from OpenAI API: {e}")
            raise
    
    async def _get_groq_response(self, messages: List[Dict[str, Any]], model: str) -> str:
        """Get response from Groq API"""
        try:
            response = await self.groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error from Groq API: {e}")
            raise
    
    async def _get_gemini_response(self, messages: List[Dict[str, Any]], model: str) -> str:
        """Get response from Gemini API"""
        try:
            # Make sure model name is correct
            if not model.startswith("models/"):
                model_name = f"models/{model}"
            else:
                model_name = model
                
            logger.debug(f"Initializing Gemini model: {model_name}")
            
            # Initialize Gemini model
            gemini_model = self.gemini.GenerativeModel(model_name)
            
            # Create chat session
            chat = gemini_model.start_chat(history=messages[:-1])
            
            # Get response
            response = await asyncio.to_thread(
                chat.send_message,
                messages[-1]["parts"][0]
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error from Gemini API: {e}")
            raise

    async def process_messages_with_websocket(self, websocket, message: str, context: List[Dict[str, Any]], model_id: Optional[str] = None) -> str:
        """Process messages and send streaming responses through WebSocket"""
        try:
            # Find the model info if model_id is provided
            model_info = None
            if model_id:
                for model in self.available_models:
                    if model["id"] == model_id:
                        model_info = model
                        break
            
            provider = model_info["provider"] if model_info else self.default_provider
            model = model_id or self.default_model
            
            # Get the AI response
            response = await self.get_response(message, context, provider, model)
            
            # Send the response through WebSocket
            await websocket.send_json({
                "type": "ai_response",
                "content": response,
                "model": model,
                "provider": provider
            })
            
            return response
            
        except Exception as e:
            error_message = f"Error processing message: {str(e)}"
            logger.error(error_message)
            
            # Send error through WebSocket
            await websocket.send_json({
                "type": "error",
                "content": error_message
            })
            
            return error_message

# Initialize the AI service
ai_service = AIService() 