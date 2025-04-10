import json
import logging
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

def create_response(
    success: bool, 
    data: Optional[Dict[str, Any]] = None, 
    message: str = "", 
    status_code: int = 200
) -> Dict[str, Any]:
    """
    Create a standardized API response.
    
    Args:
        success (bool): Whether the operation was successful
        data (Dict[str, Any], optional): Response data
        message (str, optional): Human-readable message
        status_code (int, optional): HTTP status code
        
    Returns:
        Dict[str, Any]: Standardized response dictionary
    """
    response = {
        "success": success,
        "message": message,
        "status_code": status_code
    }
    
    if data is not None:
        response["data"] = data
        
    return response

def validate_request_data(data: Dict[str, Any], required_fields: list) -> Union[str, None]:
    """
    Validate that the request data contains all required fields.
    
    Args:
        data (Dict[str, Any]): Request data to validate
        required_fields (list): List of required field names
        
    Returns:
        Union[str, None]: Error message if validation fails, None if successful
    """
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
        if data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            return f"Field cannot be empty: {field}"
    
    return None

def serialize_for_json(obj: Any) -> Any:
    """
    Convert complex objects to JSON-serializable types.
    
    Args:
        obj (Any): Object to serialize
        
    Returns:
        Any: JSON-serializable representation
    """
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj) 