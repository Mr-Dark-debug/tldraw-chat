import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables with defaults
host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8000"))
log_level = os.getenv("LOG_LEVEL", "info").lower()

if __name__ == "__main__":
    print(f"Starting TLDraw Chat API on http://{host}:{port}")
    print(f"Documentation available at http://{host}:{port}/docs")
    uvicorn.run("main:app", host=host, port=port, log_level=log_level, reload=True) 