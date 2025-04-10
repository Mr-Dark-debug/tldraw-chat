# TLDraw Chat

A powerful interactive diagramming and chat application that combines AI-powered chat with TLDraw's drawing capabilities.

## Introduction

TLDraw Chat is a full-stack application that enables users to communicate with an AI assistant while creating and manipulating diagrams in real-time. The application integrates TLDraw's powerful diagramming tools with a robust AI backend, allowing users to generate diagrams through natural language prompts, manipulate them interactively, and collaborate on visual content.

![TLDraw Chat Interface](./public/tldraw-chat-screenshot.png)
*TLDraw Chat interface showing a diagram generation example with the AI assistant*

## Features

- **AI-Powered Chat Interface**: Communicate with an intelligent assistant that can understand and respond to natural language queries
- **Diagram Generation**: Generate diagrams directly from text prompts through the AI integration
- **Interactive Drawing**: Use TLDraw's full suite of drawing and diagramming tools
- **Error Handling**: Robust error boundaries to prevent diagram errors from crashing the application
- **Real-time Updates**: Changes to diagrams are reflected in real-time

## Tech Stack

### Frontend
- **Next.js**: React framework for building the web application
- **TypeScript**: Strongly typed programming language that builds on JavaScript
- **TLDraw**: Interactive drawing and diagramming library
- **@tldraw/ai**: AI integration for the TLDraw library

### Backend
- **FastAPI**: Modern, fast web framework for building APIs with Python
- **Flask**: WSGI web application framework as a secondary API layer
- **Python 3.9+**: Core backend language
- **AI Services**:
  - Groq API
  - Gemini API
  - Support for other LLM integrations

### Communication
- **WebSockets**: Real-time bidirectional communication between clients and server
- **RESTful APIs**: Structured HTTP endpoints for data exchange

## Dependencies

### Frontend Dependencies
- React
- Next.js
- TypeScript
- TLDraw
- @tldraw/tldraw
- @tldraw/ai

### Backend Dependencies
- Python 3.9+
- FastAPI
- Flask
- Uvicorn
- Groq Client
- Gemini API Client
- Tavily API (for web search)

## Project Structure

```
tldraw-chat/
├── backend/                  # Python backend code
│   ├── ai/                   # AI service modules
│   │   ├── diagram_generator.py  # Generates diagrams from prompts
│   │   └── content_validator.py  # Validates content for safety
│   ├── routes/               # API endpoints
│   │   └── diagram_routes.py # Diagram-related API routes
│   ├── utils/                # Utility functions
│   ├── venv/                 # Python virtual environment
│   ├── main.py               # Main FastAPI application
│   ├── run.py                # Server startup script
│   └── requirements.txt      # Python dependencies
│
├── components/               # React components
│   ├── hooks/                # Custom React hooks
│   │   └── useTldrawAiDiagram.ts  # Hook for TLDraw AI integration
│   ├── ui/                   # UI components
│   ├── utils/                # Frontend utilities
│   ├── ChatInterface.tsx     # Chat interface component
│   └── TldrawWrapper.tsx     # Wrapper for TLDraw component
│
├── pages/                    # Next.js pages
│   ├── api/                  # API routes
│   │   └── generate-diagram.ts  # Diagram generation endpoint
│   └── index.tsx             # Main application page
│
├── public/                   # Static assets
├── styles/                   # CSS styles
└── README.md                 # This file
```

## Setup Instructions

### Prerequisites
- Node.js 16+ and npm/yarn
- Python 3.9+
- Git

### Frontend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tldraw-chat.git
   cd tldraw-chat
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Create a `.env.local` file in the root directory with the following variables:
   ```
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
   ```

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Running the Application

1. Start the backend server:
   ```bash
   # From the backend directory with virtual environment activated
   python run.py
   ```

2. In a separate terminal, start the frontend development server:
   ```bash
   # From the project root
   npm run dev
   # or
   yarn dev
   ```

3. Open your browser and navigate to http://localhost:3000

## Usage Guide

### Chat Interface
- Type messages in the chat input to communicate with the AI assistant
- The AI can answer questions, provide information, and generate diagrams

### Diagram Generation
- To generate a diagram, type a prompt like: "Create a flowchart for user authentication"
- The diagram will be generated and displayed in the TLDraw canvas
- You can then edit the diagram using TLDraw's tools

### Editing Diagrams
- Use TLDraw's tools to modify shapes, add connections, and customize your diagram
- All changes are saved automatically

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [TLDraw](https://tldraw.com) for their excellent diagramming library
- [Next.js](https://nextjs.org) for the React framework
- [FastAPI](https://fastapi.tiangolo.com) for the backend framework
