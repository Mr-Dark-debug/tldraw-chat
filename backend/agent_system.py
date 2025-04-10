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
from groq import AsyncGroq as GroqClient
from web_search_service import web_search_service

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
    def __init__(self, model: str = "gemini-1.5-pro", max_tokens: int = 1000):
        super().__init__(model, max_tokens)
        self.genai = google.generativeai
        self.genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
    async def generate(self, prompt: str) -> str:
        try:
            # Make sure model name is correct
            model_name = self.model
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
                
            logger.debug(f"Initializing Gemini model: {model_name}")
            model = self.genai.GenerativeModel(model_name=model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating response from Gemini: {e}")
            raise
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Gemini"""
        try:
            # Make sure model name is correct
            model_name = self.model
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
                
            logger.debug(f"Initializing Gemini model for streaming: {model_name}")
            model = self.genai.GenerativeModel(model_name=model_name)
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
        self.client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))
        
    async def generate(self, prompt: str) -> str:
        try:
            completion = await self.client.chat.completions.create(
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
            completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.max_tokens,
                stream=True
            )
            async for chunk in completion:
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
            
            logger.debug(f"Generating text using provider: {self.provider}, model: {self.model}")
            
            provider_instance = None
            if self.provider == AIProvider.OPENAI:
                logger.debug("Using OpenAIProvider")
                provider_instance = OpenAIProvider(self.model)
            elif self.provider == AIProvider.GEMINI:
                logger.debug("Using GeminiProvider")
                # Check if Gemini API key is available
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    logger.error("GEMINI_API_KEY environment variable is not set!")
                    return "Error: GEMINI_API_KEY is not configured. Please set this environment variable."
                provider_instance = GeminiProvider(self.model)
            elif self.provider == AIProvider.GROQ:
                logger.debug("Using GroqProvider")
                # Check if Groq API key is available
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    logger.error("GROQ_API_KEY environment variable is not set!")
                    return "Error: GROQ_API_KEY is not configured. Please set this environment variable."
                provider_instance = GroqProvider(self.model)
            else:
                logger.debug("Using default GroqProvider")
                provider_instance = GroqProvider(self.model)  # Default to Groq
            
            # Log the actual model being used
            logger.info(f"Using model: {provider_instance.model} with provider: {self.provider.value}")
            
            response = await provider_instance.generate(prompt)
            
            # Add the response to memory
            if use_memory:
                self.add_to_memory(prompt, "user")
                self.add_to_memory(response, "assistant")
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_text: {e}", exc_info=True)  # Add full traceback
            return f"Error generating text: {str(e)}"

class PromptEnhancerAgent(Agent):
    """Agent responsible for enhancing user prompts"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Prompt Enhancer", provider, model)
    
    async def enhance_prompt(self, prompt: str) -> str:
        system_prompt = """
        You are a specialized prompt enhancement agent with expertise in improving user requests. Your task is to:
        
        1. Analyze the initial user prompt carefully to understand the core intent
        2. Expand the prompt with relevant specifics and context the user might have implied but not stated
        3. Add structural elements that will guide the AI's response format 
        4. Clarify any ambiguities in the original prompt
        5. Include appropriate constraints and parameters
        
        For drawing/visualization requests, add specific details about:
        - Visual elements to include
        - Spatial relationships between elements
        - Preferred style, color schemes, or aesthetic direction
        - Level of detail desired
        
        Return ONLY the enhanced prompt without explanations or commentary. Your output will be sent directly to the model.
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
        You are a specialized research agent with extensive knowledge across domains. Your purpose is to provide accurate, comprehensive information relevant to the user's query.

        For each query:
        1. Identify the key concepts, terms, and relationships that need exploration
        2. Provide factual, well-structured information from your knowledge base
        3. Cover multiple perspectives and approaches where applicable
        4. Prioritize depth and accuracy over breadth when the topic is specific
        5. For visualization or drawing requests, include relevant technical knowledge, styles, or conventions

        Your research will be used as context for creating diagrams, visualizations, or instructional content. Focus on providing information that would be helpful for creating visual representations or explanations.

        Format your response with clear sections, using headers where appropriate, and ensuring information flows logically from fundamental to advanced concepts.
        """
        
        # Check if web search service is available and try to use it
        web_search_results = None
        try:
            # Perform web search
            web_search_results = await web_search_service.search_for_diagram(topic)
            logger.info(f"Web search completed for topic: {topic}")
            
            # Make sure we have a valid response dictionary
            if (web_search_results and 
                isinstance(web_search_results, dict) and 
                web_search_results.get("success", False)):
                
                # Extract relevant information
                text_results = web_search_results.get("results", [])
                images = web_search_results.get("images", [])
                
                # Create additional context from web search
                web_context = "### Web Research Results:\n\n"
                
                # Add text results
                if text_results and isinstance(text_results, list):
                    web_context += "#### Text Information:\n"
                    for i, result in enumerate(text_results[:5]):  # Limit to top 5 results
                        if isinstance(result, dict):
                            title = result.get("title", f"Source {i+1}")
                            content = result.get("content", "No content available")
                            url = result.get("url", "")
                            web_context += f"**{title}**\n{content}\nSource: {url}\n\n"
                
                # Add image information
                if images and isinstance(images, list):
                    web_context += "#### Related Images Found:\n"
                    for i, image in enumerate(images[:5]):  # Limit to top 5 images
                        if isinstance(image, dict):
                            url = image.get("url", "")
                            alt_text = image.get("alt_text", f"Image {i+1}")
                            web_context += f"Image {i+1}: {alt_text}\nURL: {url}\n\n"
                
                # Enhance system prompt with web search results
                enhanced_system_prompt = f"{system_prompt}\n\n{web_context}\n\nIncorporate these web search results into your research summary where relevant. Describe any relevant diagrams or visualizations from the images found."
                
                # Generate research with web-enhanced prompt
                research_result = await self.generate_text(
                    f"{enhanced_system_prompt}\n\nResearch topic: {topic}\n\nResearch summary:",
                    use_memory=True
                )
                
                return research_result
        except Exception as e:
            logger.error(f"Error in web search during research: {e}")
            # Fall back to standard research without web search
        
        # Default research without web search if not available or if there was an error
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
        You are a specialized visualization design agent with expertise in creating effective visual representations. Your role is to:

        1. Analyze the user's request and determine the most appropriate visualization approach
        2. Provide specific, actionable guidance on creating the visualization
        3. Consider principles of visual design, information hierarchy, and cognitive processing
        4. Recommend specific visual elements, layout, and composition
        5. Suggest color schemes that enhance understanding and accessibility
        
        For TLDraw specifically, focus on:
        - Simple shapes, connectors, and text elements available in the drawing tool
        - Logical organization of elements with clear spatial relationships
        - Appropriate use of color, size, and positioning to convey hierarchy
        - Practical instructions that can be implemented using basic drawing tools
        
        Your output should be comprehensive enough that someone could follow it to create an effective visualization from scratch, with specific guidance about positioning, sizes, colors, and relationships between elements.
        """
        
        # Check if web search service is available and try to use it
        web_search_results = None
        visualization_prompt = system_prompt
        try:
            # Create a more specific query for diagram examples
            diagram_query = f"{data_description} diagram visualization examples color scheme"
            
            # Perform web search
            web_search_results = await web_search_service.search(diagram_query, search_depth="advanced", include_images=True)
            logger.info(f"Web search completed for visualization: {diagram_query}")
            
            # Make sure we have a valid response dictionary
            if (web_search_results and 
                isinstance(web_search_results, dict) and 
                web_search_results.get("success", False)):
                
                # Extract images
                images = web_search_results.get("images", [])
                text_results = web_search_results.get("results", [])
                
                # Create additional context from web search
                web_context = "### Web Research for Visualization:\n\n"
                
                # Add text results about visualization principles
                if text_results and isinstance(text_results, list):
                    web_context += "#### Visualization References:\n"
                    for i, result in enumerate(text_results[:3]):  # Limit to top 3 results
                        if isinstance(result, dict):
                            title = result.get("title", f"Source {i+1}")
                            content = result.get("content", "No content available")
                            url = result.get("url", "")
                            web_context += f"**{title}**\n{content}\nSource: {url}\n\n"
                
                # Add image descriptions for inspiration
                if images and isinstance(images, list):
                    web_context += "#### Reference Visualizations Found Online:\n"
                    for i, image in enumerate(images[:5]):  # Limit to top 5 images
                        if isinstance(image, dict):
                            url = image.get("url", "")
                            alt_text = image.get("alt_text", f"Diagram {i+1}")
                            web_context += f"Reference Diagram {i+1}: {alt_text}\nURL: {url}\n\n"
                
                visualization_prompt = f"{system_prompt}\n\n{web_context}\n\nUse these reference visualizations as inspiration for your recommendations. Adapt color schemes, layouts, and design elements from these examples where appropriate for the TLDraw environment. Be specific about colors (include hex codes), proportions, and spatial arrangements."
        except Exception as e:
            logger.error(f"Error in web search during visualization: {e}")
            # Fall back to standard visualization without web search
            
        # Generate visualization instructions
        visualization_instruction = await self.generate_text(
            f"{visualization_prompt}\n\nVisualization request: {data_description}\n\nVisualization recommendation:",
            use_memory=True
        )
        
        return visualization_instruction

class InstructionAgent(Agent):
    """Agent responsible for providing drawing instructions for TLDraw"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI, model: Optional[str] = None):
        super().__init__("Instruction Agent", provider, model)
    
    async def generate_instructions(self, task: str) -> str:
        system_prompt = """
        You are a specialized instruction agent for TLDraw, a simple drawing tool. Your purpose is to provide clear, step-by-step instructions for creating diagrams, illustrations, or visual representations.

        For each drawing task:
        1. Break down the creation process into clear, sequential steps
        2. Start with the basic structure or layout and progress to details
        3. Provide specific guidance on placement, size, and relationships between elements
        4. Include color recommendations, styling suggestions, and text placement where appropriate
        5. Ensure instructions are easy to follow for users with basic drawing skills
        
        Format your instructions as a numbered list with descriptive headings for major sections. Include a brief introduction explaining the overall approach, and conclude with any finishing touches or refinements.
        
        Remember that TLDraw has simple tools for:
        - Creating basic shapes (rectangles, circles, triangles)
        - Drawing lines and arrows
        - Adding text
        - Using color
        - Grouping elements
        
        Optimize your instructions for these basic capabilities rather than advanced graphic design features.
        """
        
        # Check if web search service is available and try to use it
        web_search_results = None
        instruction_prompt = system_prompt
        try:
            # Create a more specific query for drawing tutorials
            drawing_query = f"how to draw {task} diagram step by step tutorial"
            
            # Perform web search
            web_search_results = await web_search_service.search(drawing_query, search_depth="advanced", include_images=True)
            logger.info(f"Web search completed for drawing instructions: {drawing_query}")
            
            # Make sure we have a valid response dictionary
            if (web_search_results and 
                isinstance(web_search_results, dict) and 
                web_search_results.get("success", False)):
                
                # Extract results
                text_results = web_search_results.get("results", [])
                images = web_search_results.get("images", [])
                
                # Create additional context from web search
                web_context = "### Web Research for Drawing Instructions:\n\n"
                
                # Add text results about step-by-step tutorials
                if text_results and isinstance(text_results, list):
                    web_context += "#### Drawing Tutorials Found:\n"
                    for i, result in enumerate(text_results[:3]):  # Limit to top 3 results
                        if isinstance(result, dict):
                            title = result.get("title", f"Tutorial {i+1}")
                            content = result.get("content", "No content available")
                            url = result.get("url", "")
                            web_context += f"**{title}**\n{content}\nSource: {url}\n\n"
                
                # Add image descriptions for reference
                if images and isinstance(images, list):
                    web_context += "#### Reference Images for Drawing Process:\n"
                    for i, image in enumerate(images[:3]):  # Limit to top 3 images
                        if isinstance(image, dict):
                            url = image.get("url", "")
                            alt_text = image.get("alt_text", f"Drawing Example {i+1}")
                            web_context += f"Drawing Example {i+1}: {alt_text}\nURL: {url}\n\n"
                
                instruction_prompt = f"{system_prompt}\n\n{web_context}\n\nIncorporate helpful techniques from these tutorials into your drawing instructions. Be specific about colors (include hex codes where possible), proportions, and step-by-step guidance tailored for TLDraw's capabilities."
        except Exception as e:
            logger.error(f"Error in web search during instruction generation: {e}")
            # Fall back to standard instructions without web search
            
        # Generate instructions
        instructions = await self.generate_text(
            f"{instruction_prompt}\n\nDrawing task: {task}\n\nStep-by-step instructions:",
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
        You are a coordinator agent responsible for routing user requests to the most appropriate specialized agent.
        
        Analyze this user request carefully and determine which specialized agent would best handle it:
        
        1. Research Agent - For requests requiring factual information, background knowledge, or domain expertise
        2. Visualization Agent - For requests about visual design, diagram layout, or data visualization principles
        3. Instruction Agent - For requests needing step-by-step guidance on creating a drawing or diagram
        4. None - If you should handle this directly as a general assistant
        
        Consider:
        - The primary intent of the user's request
        - What specialized knowledge would best serve this request
        - Whether the user is asking for information, visualization guidance, or specific instructions
        
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
        
        # Route to appropriate agent based on the analysis
        if agent_choice == "1":
            response = await self.researcher.research(enhanced_request)
        elif agent_choice == "2":
            response = await self.visualizer.generate_visualization_instruction(enhanced_request)
        elif agent_choice == "3":
            response = await self.instructor.generate_instructions(enhanced_request)
        else:
            # Handle directly with a comprehensive general assistant prompt
            system_prompt = """
            You are a helpful AI assistant skilled in providing clear, concise, and accurate responses.
            
            When responding to the user:
            1. Provide direct, actionable information
            2. Use a friendly, conversational tone
            3. When appropriate, organize information with bullet points or numbered lists
            4. Include relevant context without being overly verbose
            5. If the request is ambiguous, address the most likely interpretation first
            
            Focus on delivering maximum value and clarity in your response.
            """
            
            response = await self.generate_text(
                f"{system_prompt}\n\nUser request: {enhanced_request}\n\nResponse:",
                use_memory=True
            )
        
        return response

class AgentSystem:
    """Main class for managing the agent system"""
    
    def __init__(self, provider: AIProvider = AIProvider.GROQ, model: Optional[str] = None):
        # Initialize all agents with appropriate providers and models
        self.coordinator = CoordinatorAgent(provider, model)
        self.prompt_enhancer = PromptEnhancerAgent(provider, model)
        self.researcher = ResearchAgent(provider, model)
        self.visualizer = VisualizationAgent(provider, model)
        self.instructor = InstructionAgent(provider, model)
        
        # Set default models based on their best capabilities
        if provider == AIProvider.GROQ:
            # Groq models have different specialties
            self.prompt_enhancer.model = "llama3-8b-8192"  # Smaller model for simple prompt enhancement
            self.researcher.model = "llama3-70b-8192"      # Larger model for research
            self.visualizer.model = "llama3-70b-8192"      # Larger model for visualization
            self.instructor.model = "llama3-8b-8192"       # Smaller model for instructions
            self.coordinator.model = "llama3-70b-8192"     # Larger model for coordination
        elif provider == AIProvider.GEMINI:
            # Gemini models - using updated model names
            self.prompt_enhancer.model = "gemini-1.5-flash"  # Fast model for simple enhancement
            self.researcher.model = "gemini-1.5-pro"         # Pro model for research
            self.visualizer.model = "gemini-1.5-pro"         # Pro model for visualization
            self.instructor.model = "gemini-1.5-flash"       # Fast model for instructions
            self.coordinator.model = "gemini-1.5-pro"        # Pro model for coordination
    
    async def process_message(self, message: str) -> Dict[str, str]:
        """
        Process a message through the agent system and return the full context
        
        Returns:
            Dictionary with all processing stages and results
        """
        try:
            logger.info(f"Starting agent processing for message: {message[:50]}...")
            
            # Step 1: Enhance the prompt
            logger.info("Step 1: Enhancing prompt...")
            try:
                enhanced_prompt = await self.prompt_enhancer.enhance_prompt(message)
                logger.info("Prompt enhancement completed")
            except Exception as e:
                logger.error(f"Error in prompt enhancement: {e}")
                enhanced_prompt = f"Error processing prompt: {str(e)}"
            
            # Step 2: Research related information
            logger.info("Step 2: Researching information...")
            try:
                research_info = await self.researcher.research(message)
                logger.info("Research completed")
            except Exception as e:
                logger.error(f"Error in research: {e}")
                research_info = f"Could not complete research due to an error: {str(e)}"
            
            # Step 3: Generate visualization suggestions
            logger.info("Step 3: Generating visualization suggestions...")
            try:
                visualization = await self.visualizer.generate_visualization_instruction(message)
                logger.info("Visualization generation completed")
            except Exception as e:
                logger.error(f"Error in visualization: {e}")
                visualization = f"Could not generate visualization due to an error: {str(e)}"
            
            # Step 4: Generate instructions for TLDraw
            logger.info("Step 4: Generating drawing instructions...")
            try:
                instructions = await self.instructor.generate_instructions(message)
                logger.info("Instruction generation completed")
            except Exception as e:
                logger.error(f"Error in instruction generation: {e}")
                instructions = f"Could not generate instructions due to an error: {str(e)}"
            
            # Step 5: Generate final response from coordinator
            logger.info("Step 5: Generating final response...")
            try:
                response = await self.coordinator.process_request(message)
                logger.info("Final response generation completed")
            except Exception as e:
                logger.error(f"Error in final response generation: {e}")
                response = f"I'm sorry, but I encountered an error processing your request. The system may be experiencing issues with one or more AI providers. Please try again later. Error: {str(e)}"
            
            logger.info("Agent processing completed successfully")
            
            # Return all stages of processing
            return {
                "original_message": message,
                "enhanced_prompt": enhanced_prompt,
                "research": research_info,
                "visualization": visualization,
                "instructions": instructions,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"Error processing message: {str(e)}"
            return {
                "original_message": message,
                "enhanced_prompt": f"Error processing prompt: {str(e)}",
                "research": "Could not complete research due to an error.",
                "visualization": "Could not generate visualization due to an error.",
                "instructions": "Could not generate instructions due to an error.",
                "response": "I'm sorry, but I encountered an error processing your request. The system may be experiencing issues with one or more AI providers. Please try again later."
            }
        
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
                # Access the prompt_enhancer directly, not through coordinator
                enhanced_prompt = await self.prompt_enhancer.enhance_prompt(prompt)
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
                # Research agent - access directly
                research_result = await self.researcher.research(prompt)
                if websocket:
                    await websocket.send_json({
                        "type": "response",
                        "content": research_result
                    })
                yield research_result
                
            elif agent_type == "visualization":
                # Visualization agent - access directly
                vis_result = await self.visualizer.generate_visualization_instruction(prompt)
                if websocket:
                    await websocket.send_json({
                        "type": "response",
                        "content": vis_result
                    })
                yield vis_result
                
            elif agent_type == "instruction":
                # Instruction agent - access directly
                inst_result = await self.instructor.generate_instructions(prompt)
                if websocket:
                    await websocket.send_json({
                        "type": "response",
                        "content": inst_result
                    })
                yield inst_result
                    
            else:
                # Default agent processing - use the full process_message flow
                results = await self.process_message(prompt)
                response = results["response"]
                
                if websocket:
                    await websocket.send_json({
                        "type": "full_response",
                        "content": results
                    })
                yield response
                    
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