import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class DiagramGenerator:
    """
    Class responsible for generating diagrams based on text prompts.
    Provides methods for validating prompts and generating diagram data.
    """
    
    def __init__(self):
        """Initialize the DiagramGenerator with default settings."""
        logger.info("Initializing DiagramGenerator")
        
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate if the prompt is suitable for diagram generation.
        
        Args:
            prompt (str): The prompt to validate
            
        Returns:
            bool: True if the prompt is valid, False otherwise
        """
        if not prompt or len(prompt.strip()) == 0:
            logger.warning("Empty prompt received")
            return False
            
        # Add more validation rules as needed
        
        logger.info(f"Prompt validated: {prompt[:50]}...")
        return True
        
    def generate_diagram(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a diagram based on the given text prompt.
        
        Args:
            prompt (str): Text description of the diagram to generate
            
        Returns:
            Dict[str, Any]: A dictionary containing the diagram data in TLDraw format
        """
        logger.info(f"Generating diagram for prompt: {prompt}")
        
        try:
            # This is a simplified example that creates a basic TLDraw diagram
            # In a real implementation, this might involve calling an AI model
            # or processing the prompt to generate diagram elements
            
            # Generate an if-else flow chart diagram since that's what the user wants
            # The structure follows TLDraw's expected format
            diagram_data = {
                "document": {
                    "id": "doc",
                    "name": "Generated Diagram",
                    "version": 15.5,
                    "pages": {
                        "page1": {
                            "id": "page1",
                            "name": "Page 1",
                            "shapes": {
                                # Decision diamond
                                "shape:shape1": {
                                    "id": "shape:shape1",
                                    "type": "geo",
                                    "x": 200,
                                    "y": 100,
                                    "width": 150,
                                    "height": 80,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "geo": "diamond",
                                        "w": 150,
                                        "h": 80,
                                        "color": "black",
                                        "fill": "none",
                                        "text": "Condition?",
                                        "font": "draw",
                                        "align": "middle",
                                        "autoSize": True
                                    }
                                },
                                # True path (YES) rectangle
                                "shape:shape2": {
                                    "id": "shape:shape2",
                                    "type": "geo",
                                    "x": 350,
                                    "y": 250,
                                    "width": 120,
                                    "height": 60,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "geo": "rectangle",
                                        "w": 120,
                                        "h": 60,
                                        "color": "black",
                                        "fill": "none",
                                        "text": "TRUE action",
                                        "font": "draw",
                                        "align": "middle",
                                        "autoSize": True
                                    }
                                },
                                # False path (NO) rectangle
                                "shape:shape3": {
                                    "id": "shape:shape3",
                                    "type": "geo",
                                    "x": 80,
                                    "y": 250,
                                    "width": 120,
                                    "height": 60,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "geo": "rectangle",
                                        "w": 120,
                                        "h": 60,
                                        "color": "black",
                                        "fill": "none",
                                        "text": "FALSE action",
                                        "font": "draw",
                                        "align": "middle",
                                        "autoSize": True
                                    }
                                },
                                # YES Label
                                "shape:shape4": {
                                    "id": "shape:shape4",
                                    "type": "text",
                                    "x": 300,
                                    "y": 180,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "text": "YES",
                                        "w": 50,
                                        "h": 30,
                                        "font": "draw",
                                        "align": "middle",
                                        "autoSize": True
                                    }
                                },
                                # NO Label
                                "shape:shape5": {
                                    "id": "shape:shape5",
                                    "type": "text",
                                    "x": 150,
                                    "y": 180,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "text": "NO",
                                        "w": 50,
                                        "h": 30,
                                        "font": "draw",
                                        "align": "middle",
                                        "autoSize": True
                                    }
                                },
                                # Title
                                "shape:shape6": {
                                    "id": "shape:shape6",
                                    "type": "text",
                                    "x": 150,
                                    "y": 30,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "text": "If-Else Flow Diagram",
                                        "w": 250,
                                        "h": 30,
                                        "font": "draw",
                                        "align": "middle",
                                        "autoSize": True,
                                        "size": "xl"
                                    }
                                }
                            },
                            "connectors": {
                                # Arrow from decision to TRUE action
                                "shape:arrow1": {
                                    "id": "shape:arrow1",
                                    "type": "arrow",
                                    "x": 275,
                                    "y": 140,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "start": {
                                            "type": "binding",
                                            "boundShapeId": "shape:shape1"
                                        },
                                        "end": {
                                            "type": "binding",
                                            "boundShapeId": "shape:shape2"
                                        },
                                        "color": "black"
                                    }
                                },
                                # Arrow from decision to FALSE action
                                "shape:arrow2": {
                                    "id": "shape:arrow2",
                                    "type": "arrow",
                                    "x": 150,
                                    "y": 140,
                                    "rotation": 0,
                                    "isLocked": False,
                                    "opacity": 1,
                                    "props": {
                                        "start": {
                                            "type": "binding",
                                            "boundShapeId": "shape:shape1"
                                        },
                                        "end": {
                                            "type": "binding",
                                            "boundShapeId": "shape:shape3"
                                        },
                                        "color": "black"
                                    }
                                }
                            }
                        }
                    },
                    "currentPageId": "page1"
                }
            }
            
            return diagram_data
            
        except Exception as e:
            logger.error(f"Error in diagram generation: {str(e)}")
            raise Exception(f"Failed to generate diagram: {str(e)}")
    
    def enhance_diagram(self, diagram_data: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """
        Enhance an existing diagram based on user feedback.
        
        Args:
            diagram_data (Dict[str, Any]): The existing diagram data
            feedback (str): User feedback for improving the diagram
            
        Returns:
            Dict[str, Any]: The enhanced diagram data
        """
        logger.info(f"Enhancing diagram with feedback: {feedback}")
        
        # In a real implementation, this would analyze the feedback
        # and make improvements to the existing diagram
        
        # For now, just add a note shape with the feedback
        try:
            current_page_id = diagram_data["document"]["currentPageId"]
            shapes = diagram_data["document"]["pages"][current_page_id]["shapes"]
            
            # Create a new shape ID with proper "shape:" prefix
            new_shape_id = f"shape:shape{len(shapes) + 1}"
            
            # Add a new shape with the feedback
            shapes[new_shape_id] = {
                "id": new_shape_id,
                "type": "note",
                "x": 100,
                "y": 250,
                "width": 200,
                "height": 100,
                "text": f"Feedback: {feedback[:50]}" + ("..." if len(feedback) > 50 else "")
            }
            
            return diagram_data
            
        except Exception as e:
            logger.error(f"Error in diagram enhancement: {str(e)}")
            raise Exception(f"Failed to enhance diagram: {str(e)}") 