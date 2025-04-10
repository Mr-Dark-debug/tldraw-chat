import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get configuration from environment variables with defaults
host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8000"))
log_level = os.getenv("LOG_LEVEL", "info").lower()

def main():
    try:
        # Check if we're running from the correct directory
        if not os.path.exists(os.path.join(os.getcwd(), "main.py")):
            # Add the current directory to the path so Python can find the main module
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            if not os.path.exists("main.py"):
                logger.error("Cannot find main.py in current directory.")
                logger.error("Please run this script from the backend directory.")
                logger.error("Use: cd backend && python run.py")
                sys.exit(1)
                
        logger.info(f"Starting TLDraw Chat API on http://{host}:{port}")
        logger.info(f"Documentation available at http://{host}:{port}/docs")
        
        if host == "0.0.0.0":
            logger.warning("Server is running on 0.0.0.0, which might cause 'ERR_ADDRESS_INVALID' in some browsers.")
            logger.warning("Consider using 127.0.0.1 instead by setting HOST=127.0.0.1 in your .env file.")
            
        uvicorn.run("main:app", host=host, port=port, log_level=log_level, reload=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 