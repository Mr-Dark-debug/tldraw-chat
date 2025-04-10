import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tavily import AsyncTavilyClient

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class WebSearchService:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not found in environment variables!")
        else:
            self.tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
            logger.info("Tavily API initialized")
            
    def is_available(self) -> bool:
        """Check if web search service is available"""
        return self.tavily_api_key is not None
    
    async def search(self, query: str, search_depth: str = "basic", include_images: bool = True) -> Dict[str, Any]:
        """
        Perform a web search using Tavily
        
        Args:
            query: The search query
            search_depth: "basic" or "advanced" - determines the depth of the search
            include_images: Whether to include images in the search results
            
        Returns:
            Dictionary containing search results and status
        """
        try:
            if not self.is_available():
                logger.error("Tavily API is not available - missing API key")
                return {
                    "success": False,
                    "error": "Web search service is not available",
                    "results": [],
                    "images": []
                }
            
            # Ensure query is a string and not too long
            if not isinstance(query, str):
                query = str(query)
                
            # Tavily has a limit on query length (400 chars)
            if len(query) > 400:
                query = query[:397] + "..."
                logger.warning(f"Query truncated to 400 characters: {query}")
                
            # Perform the search with the async client
            response = await self.tavily_client.search(
                query=query, 
                search_depth=search_depth,
                include_images=include_images
            )
            
            # Ensure response is a dictionary
            if not isinstance(response, dict):
                logger.error(f"Unexpected response type from Tavily: {type(response)}")
                return {
                    "success": False,
                    "error": f"Unexpected response type: {type(response)}",
                    "results": [],
                    "images": []
                }
                
            # Extract images if they exist
            images = []
            if include_images and "images" in response:
                images = response.get("images", [])
                
            # Extract text results
            results = response.get("results", [])
            
            return {
                "success": True,
                "results": results,
                "images": images
            }
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "images": []
            }
            
    async def search_for_diagram(self, topic: str) -> Dict[str, Any]:
        """
        Specialized search for diagram-related information
        
        Args:
            topic: The diagram topic to search for
            
        Returns:
            Dictionary containing search results focused on diagram information
        """
        try:
            # Ensure topic is a string
            if not isinstance(topic, str):
                topic = str(topic)
                
            # Create a more specific query for diagrams
            diagram_query = f"{topic} diagram visualization flowchart example"
            
            # Perform the search with advanced depth for better results
            result = await self.search(diagram_query, search_depth="advanced", include_images=True)
            
            # Add some analysis
            if result["success"]:
                return {
                    "success": True,
                    "topic": topic,
                    "results": result["results"],
                    "images": result["images"],
                    "query_used": diagram_query
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error in diagram search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "images": []
            }

# Create a singleton instance
web_search_service = WebSearchService() 