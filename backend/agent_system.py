import os
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import requests
from ai_service import ai_service
from enum import Enum
from groq import Client

# Load environment variables
load_dotenv()

# Configure Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logging.error(f"Failed to configure Gemini: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define AI providers
class AIProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"
    
# Base class for AI providers
class BaseAIProvider:
    def __init__(self, model: str, max_tokens: int = 1000):
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def generate(self, prompt: str) -> str:
        """Generate a response from the AI model"""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the AI model
        Default implementation just yields the final result, but
        subclasses can override this to provide true streaming
        """
        result = await self.generate(prompt)
        yield result

class OpenAIProvider(BaseAIProvider):
    def __init__(self, model: str, max_tokens: int = 1000):
        super().__init__(model, max_tokens)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    async def generate(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response from OpenAI: {e}")
            raise
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response from OpenAI"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"Error streaming response from OpenAI: {e}")
            raise

class GeminiProvider(BaseAIProvider):
    def __init__(self, model: str = "gemini-pro", max_tokens: int = 1000):
        super().__init__(model, max_tokens)
        self.genai = google.generativeai
        self.genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
    async def generate(self, prompt: str) -> str:
        try:
            model = self.genai.GenerativeModel(model_name=self.model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating response from Gemini: {e}")
            raise
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Gemini"""
        try:
            model = self.genai.GenerativeModel(model_name=self.model)
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self.logger.error(f"Error streaming response from Gemini: {e}")
            raise

class GroqProvider(BaseAIProvider):
    def __init__(self, model: str = "llama3-70b-8192", max_tokens: int = 1000):
        super().__init__(model, max_tokens)
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        
    async def generate(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response from Groq: {e}")
            raise
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Groq"""
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"Error streaming response from Groq: {e}")
            raise

class Agent:
    """Base class for all agents in the system"""
    def __init__(self, name: str, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        self.name = name
        self.provider = provider
        self.memory: List[Dict[str, Any]] = []
        self.model = model
        
        # Set default models if none specified
        if not model:
            if provider == AIProvider.OPENAI:
                self.model = "gpt-3.5-turbo"
            elif provider == AIProvider.GEMINI:
                self.model = "gemini-1.5-pro"
            elif provider == AIProvider.GROQ:
                self.model = "llama3-70b-8192"

    def add_to_memory(self, message: str, sender: str = "user"):
        self.memory.append({
            "sender": sender,
            "text": message
        })
    
    def clear_memory(self):
        self.memory = []

    async def generate_text(self, prompt: str, use_memory: bool = True) -> str:
        try:
            context = self.memory if use_memory else []
            
            provider_instance = None
            if self.provider == AIProvider.OPENAI:
                provider_instance = OpenAIProvider(self.model)
            elif self.provider == AIProvider.GEMINI:
                provider_instance = GeminiProvider(self.model)
            elif self.provider == AIProvider.GROQ:
                provider_instance = GroqProvider(self.model)
            else:
                provider_instance = GroqProvider(self.model)  # Default to Groq
            
            response = await provider_instance.generate(prompt)
            
            # Add the response to memory
            if use_memory:
                self.add_to_memory(prompt, "user")
                self.add_to_memory(response, "assistant")
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_text: {e}")
            return f"Error generating text: {str(e)}"

class PromptEnhancerAgent(Agent):
    """Agent responsible for enhancing user prompts"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Prompt Enhancer", provider, model)
    
    async def enhance_prompt(self, prompt: str) -> str:
        system_prompt = """
        You are a prompt enhancement specialist. Your job is to take a user's initial query or prompt and make it more specific, detailed, and effective.
        Consider adding:
        - More specific details and requirements
        - Context that might be helpful
        - Clarifications to prevent misunderstandings
        - Structure to guide the response format
        
        Enhance the prompt but maintain the original intent. Return ONLY the enhanced prompt without explanations or notes.
        """
        
        enhanced_prompt = await self.generate_text(
            f"{system_prompt}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:",
            use_memory=False
        )
        
        return enhanced_prompt

class ResearchAgent(Agent):
    """Agent responsible for searching for information related to the prompt"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Research Agent", provider, model)
    
    async def research(self, topic: str) -> str:
        system_prompt = """
        You are a research specialist. Your task is to provide a comprehensive and factual summary on the given topic.
        Include:
        - Key facts and information
        - Different perspectives if applicable
        - Recent developments if relevant
        - Organized structure with clear sections
        
        Base your research on established facts and reliable information. Be thorough but concise.
        """
        
        research_result = await self.generate_text(
            f"{system_prompt}\n\nResearch topic: {topic}\n\nResearch summary:",
            use_memory=True
        )
        
        return research_result

class VisualizationAgent(Agent):
    """Agent responsible for suggesting visual elements and composition"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Visualization Agent", provider, model)
    
    async def generate_visualization_instruction(self, data_description: str) -> str:
        system_prompt = """
        You are a data visualization specialist. Your task is to suggest the best visualization approach for the described data.
        Provide:
        - Recommended chart or graph type
        - Explanation of why this visualization works best
        - Key elements to include in the visualization
        - Color scheme recommendations if appropriate
        - Tools that could be used to create this visualization
        
        Focus on clarity, effectiveness, and best practices in data visualization.
        """
        
        visualization_instruction = await self.generate_text(
            f"{system_prompt}\n\nData description: {data_description}\n\nVisualization recommendation:",
            use_memory=True
        )
        
        return visualization_instruction

class InstructionAgent(Agent):
    """Agent responsible for providing drawing instructions for TLDraw"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Instruction Agent", provider, model)
    
    async def generate_instructions(self, task: str) -> str:
        system_prompt = """
        You are an instruction specialist. Your task is to break down a complex task into clear, step-by-step instructions.
        Provide:
        - A numbered list of steps in logical order
        - Clear, concise language for each step
        - Warnings or notes for potentially confusing steps
        - Required materials or prerequisites at the beginning
        - Estimated time to complete (if applicable)
        
        Make your instructions easy to follow for someone who is unfamiliar with the task.
        """
        
        instructions = await self.generate_text(
            f"{system_prompt}\n\nTask to break down: {task}\n\nStep-by-step instructions:",
            use_memory=True
        )
        
        return instructions

class CoordinatorAgent(Agent):
    """Agent responsible for coordinating all other agents and providing final response"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Coordinator", provider, model)
        
        # Initialize specialized agents
        self.prompt_enhancer = PromptEnhancerAgent(provider, model)
        self.researcher = ResearchAgent(provider, model)
        self.visualizer = VisualizationAgent(provider, model)
        self.instructor = InstructionAgent(provider, model)
    
    async def process_request(self, user_request: str) -> str:
        # Determine the best agent for the task
        analysis_prompt = f"""
        Analyze this user request and determine which specialized agent would best handle it.
        Options:
        1. Research Agent - For requests needing factual information and summaries
        2. Visualization Agent - For requests about data visualization
        3. Instruction Agent - For requests needing step-by-step guidance
        4. None - If I should handle this directly
        
        User request: "{user_request}"
        
        Respond with ONLY the agent number (1, 2, 3) or "4" for None. No explanation.
        """
        
        agent_choice = await self.generate_text(analysis_prompt, use_memory=False)
        agent_choice = agent_choice.strip()
        
        # Extract just the number from the response
        for char in agent_choice:
            if char.isdigit():
                agent_choice = char
                break
        
        # Enhanced prompt for better results
        enhanced_request = await self.prompt_enhancer.enhance_prompt(user_request)
        
        # Route to appropriate agent
        if agent_choice == "1":
            response = await self.researcher.research(enhanced_request)
        elif agent_choice == "2":
            response = await self.visualizer.generate_visualization_instruction(enhanced_request)
        elif agent_choice == "3":
            response = await self.instructor.generate_instructions(enhanced_request)
        else:
            # Handle directly
            system_prompt = """
            You are a helpful AI assistant. Provide a clear, informative, and friendly response to the user's request.
            """
            response = await self.generate_text(
                f"{system_prompt}\n\nUser request: {enhanced_request}\n\nResponse:",
                use_memory=True
            )
        
        return response

class AgentSystem:
    """Main class for managing the agent system"""
    
    def __init__(self, provider: AIProvider = AIProvider.GROQ, model: Optional[str] = None):
        self.coordinator = CoordinatorAgent(provider, model)
    
    async def process_message(self, message: str) -> str:
        try:
            response = await self.coordinator.process_request(message)
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models from AI service"""
        return ai_service.get_models()
    
    async def process_message_with_websocket(self, websocket, message: str, model_id: Optional[str] = None) -> str:
        """Process a message with WebSocket support for streaming responses"""
        try:
            # Get model provider if model_id is specified
            provider = AIProvider.GROQ  # Default to Groq
            
            if model_id:
                for model in self.get_available_models():
                    if model["id"] == model_id:
                        provider_name = model["provider"].upper()
                        try:
                            provider = AIProvider[provider_name]
                        except KeyError:
                            provider = AIProvider.GROQ
                        break
            
            # Create a new coordinator with the specified model
            coordinator = CoordinatorAgent(provider, model_id)
            
            # Process the request
            response = await coordinator.process_request(message)
            
            # Send the response through the WebSocket
            await websocket.send_json({
                "type": "agent_response",
                "content": response,
                "model": model_id or coordinator.model,
                "provider": provider.value
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

    async def process_request_stream(self, request: Dict[str, Any], websocket=None) -> AsyncGenerator[str, None]:
        """
        Process a request and return a streaming response
        
        Args:
            request: Dictionary containing the request data
            websocket: Optional WebSocket connection for streaming responses
        
        Yields:
            Chunks of the response as they are generated
        """
        try:
            # Extract request data
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 1000)
            model = request.get("model", "llama3-70b-8192")
            agent_type = request.get("agent_type", "default")
            provider_type = request.get("provider", "groq")
            
            # Get appropriate provider
            provider_instance = None
            if provider_type.lower() == "openai":
                provider_instance = OpenAIProvider(model, max_tokens)
            elif provider_type.lower() == "gemini":
                provider_instance = GeminiProvider(model, max_tokens)
            elif provider_type.lower() == "groq":
                provider_instance = GroqProvider(model, max_tokens)
            else:
                provider_instance = GroqProvider(model, max_tokens)
            
            # Process the request based on agent type
            if agent_type == "prompt_enhancer":
                enhanced_prompt = await self.coordinator.prompt_enhancer.enhance_prompt(prompt)
                logger.info(f"Enhanced prompt: {enhanced_prompt}")
                
                # Stream response from provider
                async for chunk in provider_instance.generate_stream(enhanced_prompt):
                    if websocket:
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk
                        })
                    yield chunk
                    
            elif agent_type == "research":
                # Research agent might need special handling for streaming
                research_result = await self.coordinator.researcher.research(prompt)
                if websocket:
                    await websocket.send_json({
                        "type": "response",
                        "content": research_result
                    })
                yield research_result
                    
            else:
                # Default agent processing
                async for chunk in provider_instance.generate_stream(prompt):
                    if websocket:
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk
                        })
                    yield chunk
                    
        except Exception as e:
            error_message = f"Error in agent system: {str(e)}"
            logger.error(error_message)
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "content": error_message
                })
            yield error_message

# Create a singleton instance
agent_system = AgentSystem(provider=AIProvider.GROQ) 