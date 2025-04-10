'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@radix-ui/react-scroll-area';
import { Send, ChevronDown, RefreshCw } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { 
  formatTime, 
  scrollToBottom, 
  isStatusUpdate, 
  CopyButton, 
  MarkdownRenderer 
} from '@/components/utils/chat-utils';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  mode?: 'chat' | 'create';
}

interface ServerMessage {
  id: string;
  text: string;
  sender: string;
  timestamp: number;
}

interface ModelInfo {
  provider: string;
  name: string;
  displayName: string;
}

// Interface for agent processing details
interface AgentDetails {
  id: string;
  original_message?: string;
  enhanced_prompt?: string;
  research?: string;
  visualization?: string;
  instructions?: string;
  response?: string;
}

// Extended WebSocket interface to include responseTimeout
interface ExtendedWebSocket extends WebSocket {
  responseTimeout?: NodeJS.Timeout;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [useAgentSystem, setUseAgentSystem] = useState(true);
  const [agentDetails, setAgentDetails] = useState<AgentDetails | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentMode, setCurrentMode] = useState<'chat' | 'create'>('chat');
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelInfo>({
    provider: 'groq',
    name: 'llama3-8b-8192',
    displayName: 'Groq - Llama 3 8B'
  });
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([
    { provider: 'groq', name: 'llama3-8b-8192', displayName: 'Groq - Llama 3 8B' },
    { provider: 'groq', name: 'llama3-70b-8192', displayName: 'Groq - Llama 3 70B' },
    { provider: 'gemini', name: 'gemini-1.5-pro', displayName: 'Gemini 1.5 Pro' },
  ]);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const ws = useRef<ExtendedWebSocket | null>(null);
  const clientId = useRef<string>('');

  // Mark component as mounted for client-side rendering only
  useEffect(() => {
    setIsMounted(true);
    
    // Client-side only code - initialize clientId ref
    if (typeof window !== 'undefined') {
      clientId.current = uuidv4();
      
      // Load saved messages from localStorage
      try {
        const savedMessages = localStorage.getItem('tldraw-chat-messages');
        if (savedMessages) {
          const parsedMessages = JSON.parse(savedMessages);
          // Convert timestamps back to Date objects
          const formattedMessages = parsedMessages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          }));
          setMessages(formattedMessages);
        }
      } catch (e) {
        console.error("Error loading messages from localStorage:", e);
      }
      
      // Fetch available models from the backend
      fetch('http://127.0.0.1:8000/models')
        .then(response => response.json())
        .then(data => {
          if (data && Array.isArray(data)) {
            setAvailableModels(data);
          }
        })
        .catch(error => {
          console.error('Error fetching available models:', error);
        });
    }
    
    return () => {
      // Clean up WebSocket on component unmount
      if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
        try {
          ws.current.close();
        } catch (err) {
          console.error("Error closing WebSocket connection:", err);
        }
      }
    };
  }, []);

  // Function to establish WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (!isMounted || typeof WebSocket === 'undefined') return; // Avoid WebSocket connection during SSR
    if (ws.current?.readyState === WebSocket.OPEN) return;
    
    setIsConnecting(true);
    setIsConnected(false);
    
    // Use the correct WebSocket URL based on protocol (ws or wss)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || '127.0.0.1';
    const port = '8000'; // Default backend port
    const endpoint = currentMode === 'create' ? 'agent' : 'ws';
    const wsUrl = `${protocol}//${host}:${port}/ws/${endpoint}/${clientId.current}`;
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    
    try {
      if (ws.current) {
        // Close any existing connection first
        if (ws.current.readyState !== WebSocket.CLOSED) {
          try {
            ws.current.close();
          } catch (e) {
            console.error("Error closing existing WebSocket:", e);
          }
        }
        ws.current = null;
      }
      
      const socket = new WebSocket(wsUrl);
      ws.current = socket;
      
      socket.onopen = () => {
        console.log('WebSocket connection established');
        setIsConnected(true);
        setIsConnecting(false);
      };
      
      socket.onmessage = (event) => {
        handleSocketMessage(event);
      };
      
      socket.onclose = (event) => {
        console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
        setIsConnected(false);
        setIsConnecting(false);
        
        // Auto-reconnect after a delay, but only if the socket wasn't closed deliberately
        if (!event.wasClean) {
          console.log('Connection closed unexpectedly, reconnecting in 5 seconds...');
          setTimeout(() => {
            connectWebSocket();
          }, 5000);
        }
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        setIsConnecting(false);
      };
      
    } catch (e) {
      console.error("Error creating WebSocket:", e);
      setIsConnected(false);
      setIsConnecting(false);
    }
  }, [isMounted, currentMode]);

  // Handle incoming WebSocket messages
  const handleSocketMessage = (event: MessageEvent) => {
    try {
      // Clear any response timeout when we receive a message
      if (ws.current?.responseTimeout) {
        clearTimeout(ws.current.responseTimeout);
        ws.current.responseTimeout = undefined;
      }
      
      // Parse the message data
      const data = JSON.parse(event.data);
      
      // Handle ping messages from server to keep connection alive
      if (data.type === 'ping') {
        // Send pong response back to the server
        try {
          if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({
              type: 'pong',
              timestamp: Date.now() / 1000
            }));
          }
        } catch (e) {
          console.error('Error sending pong:', e);
        }
        return; // Don't process pings further
      }
      
      console.log('Received message:', data);
      
      if (data.type === 'agent_response') {
        // This is a detailed agent response
        setIsProcessing(false);
        
        // Create a message ID for the response
        const messageId = data.id || uuidv4();
        
        // Process agent response and create diagram if in create mode
        if (currentMode === 'create') {
          // Pass the entire data object to generateDiagramFromInstructions
          generateDiagramFromInstructions(data);
        }
        
        // Update UI with the response
        const aiMessageData: Message = {
          id: messageId,
          text: data.response,
          sender: 'assistant',
          timestamp: new Date(data.timestamp * 1000 || Date.now()),
          mode: currentMode
        };
        
        // Add message to our state
        const updatedMessages = [...messages, aiMessageData];
        setMessages(updatedMessages);
        
        // Create agent details object with all required fields
        const details: AgentDetails = {
          id: messageId,
          original_message: data.original_message || '',
          enhanced_prompt: data.enhanced_prompt || '',
          research: data.research || '',
          visualization: data.visualization || '',
          instructions: data.instructions || '',
          response: data.response || ''
        };
        
        // Update agent details for UI
        setAgentDetails(details);
        console.log('Updated agent details:', details);
        
        // Store messages in localStorage for persistence
        saveMessagesToLocalStorage(updatedMessages);
      } else if (data.type === 'model_list') {
        // Process model list response
        setAvailableModels(data.models || []);
      } else if (data.type === 'processing_update') {
        // Process status update from backend
        const statusMessage = {
          id: uuidv4(),
          text: `${data.status}...`,
          sender: 'assistant' as const,
          timestamp: new Date(),
          isStatusUpdate: true
        };
        
        setMessages((prevMessages) => {
          // Remove any previous status updates
          const filteredMessages = prevMessages.filter((msg: any) => !msg.isStatusUpdate);
          return [...filteredMessages, statusMessage];
        });
      } else if (data.type === 'error') {
        // Handle error message
        console.error('Server error:', data.content);
        const errorMessage = {
          id: uuidv4(),
          text: data.content || "An error occurred while processing your request.",
          sender: 'assistant' as const,
          timestamp: new Date(),
          mode: currentMode,
        };
        
        setMessages(prevMessages => [...prevMessages, errorMessage]);
        setIsProcessing(false);
      } else {
        // This is a single message
        setIsProcessing(false);
        let messageText = data.text;
        
        // Clean up any error messages in regular messages
        if (messageText && messageText.includes('Error generating text with')) {
          messageText = "I encountered an issue with one of my services, but I'll still try to help.";
        }
        
        const newMessage = {
          ...data,
          text: messageText,
          timestamp: new Date(data.timestamp * 1000),
          mode: data.mode || currentMode,
        };
        
        setMessages((prevMessages) => {
          // Check if the message already exists
          const exists = prevMessages.some((msg) => msg.id === newMessage.id);
          if (exists) return prevMessages;
          
          // Filter out status update messages
          const filteredMessages = prevMessages.filter((msg: any) => !msg.isStatusUpdate);
          const updatedMessages = [...filteredMessages, newMessage];
          
          // Store in localStorage for persistence
          saveMessagesToLocalStorage(updatedMessages);
          
          return updatedMessages;
        });
      }
    } catch (e) {
      console.error('Error handling WebSocket message:', e);
    }
  };

  // Generate a diagram from AI instructions
  const generateDiagramFromInstructions = (response: any) => {
    if (!isMounted || typeof window === 'undefined') return false;
    
    try {
      console.log('Generating diagram from response:', response);
      
      if (!response || typeof response !== 'object') {
        console.error('Invalid response format for diagram generation');
        return false;
      }

      // Try to extract diagram instructions from the agent response
      let instructions = '';
      let visualization = '';
      
      // Check if there's a dedicated visualization section
      if (response.visualization && typeof response.visualization === 'string') {
        visualization = response.visualization;
        console.log('Found visualization data:', visualization.substring(0, 100) + '...');
      } else if (response.text && typeof response.text === 'string') {
        // Try to extract from regular text response
        const visualizationMatch = response.text.match(/visualization:(.*?)(?=(##|\n\n|$))/i);
        if (visualizationMatch && visualizationMatch[1]) {
          visualization = visualizationMatch[1].trim();
          console.log('Extracted visualization from text:', visualization.substring(0, 100) + '...');
        }
      }
      
      // Extract instructions from content or text
      if (response.instructions && typeof response.instructions === 'string') {
        instructions = response.instructions;
        console.log('Found instructions data:', instructions.substring(0, 100) + '...');
      } else if (response.content && typeof response.content === 'string') {
        instructions = response.content;
        console.log('Using content as instructions:', instructions.substring(0, 100) + '...');
      } else if (response.text && typeof response.text === 'string') {
        instructions = response.text;
        console.log('Using text as instructions:', instructions.substring(0, 100) + '...');
        
        // Try to extract specific diagram section if it exists
        const diagramMatch = response.text.match(/diagram:(.*?)(?=(##|\n\n|$))/i);
        if (diagramMatch && diagramMatch[1]) {
          instructions = diagramMatch[1].trim();
          console.log('Extracted diagram instructions from text:', instructions.substring(0, 100) + '...');
        }
      }
      
      if (!instructions && !visualization) {
        console.warn('No instructions or visualization found in agent response');
        // Create at least minimal instructions
        instructions = "Create a simple diagram with a central node labeled 'Main Concept'";
        console.log('Using fallback minimal instructions');
      }
      
      // Ensure we have a non-empty instructions object
      const instructionsObj = {
        instructions: instructions || "Create a simple diagram with shapes and connections",
        visualization: visualization || ""
      };
      
      // Send message to TldrawWrapper to create the diagram
      const drawMessage = {
        type: 'draw_diagram',
        instructions: instructionsObj
      };
      
      console.log('Sending diagram instruction message:', drawMessage);
      
      // First try direct event dispatch for better compatibility
      try {
        const event = new CustomEvent('tlDrawDiagram', { detail: drawMessage });
        window.dispatchEvent(event);
        console.log('Dispatched tlDrawDiagram custom event');
      } catch (e) {
        console.error('Error dispatching custom event:', e);
      }
      
      // Also use window.postMessage as fallback
      window.postMessage(drawMessage, window.location.origin);
      console.log('Sent postMessage with diagram instructions');
      
      return true; // Indicate successful dispatch
    } catch (error) {
      console.error('Error generating diagram from instructions:', error);
      return false;
    }
  };

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom(messagesEndRef);
  }, [messages]);

  // Connect to WebSocket on component mount or when mode changes
  useEffect(() => {
    if (!isMounted) return;
    
    // Close existing connection if switching endpoints
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      try {
        ws.current.close();
      } catch (e) {
        console.error("Error closing WebSocket connection:", e);
      }
    }
    
    // Add a small delay before reconnecting to avoid rapid connection attempts
    const reconnectTimer = setTimeout(() => {
      connectWebSocket();
    }, 500);
    
    return () => {
      clearTimeout(reconnectTimer);
    };
  }, [connectWebSocket, isMounted, currentMode]);

  // Function to save messages to localStorage
  const saveMessagesToLocalStorage = (messagesToSave: Message[]) => {
    if (!isMounted || typeof window === 'undefined') return;
    
    try {
      localStorage.setItem('tldraw-chat-messages', JSON.stringify(messagesToSave));
    } catch (e) {
      console.error("Error storing messages in localStorage:", e);
    }
  };

  // Handle key down event for input
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Check if Enter key was pressed without Shift key
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) {
        handleSend();
      }
    }
  };

  // Parse and handle command
  const parseCommand = (text: string) => {
    if (text.startsWith('/chat')) {
      switchMode('chat');
      return text.replace('/chat', '').trim();
    }
    
    if (text.startsWith('/create')) {
      switchMode('create');
      return text.replace('/create', '').trim();
    }
    
    return text;
  };

  // Function to send message via WebSocket
  const sendMessage = (text: string) => {
    if (!isMounted || !text.trim() || !ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    
    // Parse any command in the text
    const parsedText = parseCommand(text);
    
    const messageId = uuidv4();
    const message = {
      id: messageId,
      text: parsedText,
      sender: 'user',
      timestamp: Date.now() / 1000, // Unix timestamp
      mode: currentMode,
      model: selectedModel, // Send the selected model with the message
    };
    
    try {
      ws.current.send(JSON.stringify(message));
      setInput('');
      setIsProcessing(true);
      
      // Add the user message to the UI immediately
      const userMessage = {
        id: messageId,
        text: parsedText,
        sender: 'user' as const,
        timestamp: new Date(),
        mode: currentMode,
      };
      
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages, userMessage];
        // Store in localStorage for persistence
        saveMessagesToLocalStorage(updatedMessages);
        return updatedMessages;
      });
      
      // Set a timeout to handle case where server doesn't respond
      const responseTimeout = setTimeout(() => {
        if (isProcessing) {
          setIsProcessing(false);
          
          // Add an error message if we haven't received a response
          const timeoutMessage = {
            id: uuidv4(),
            text: "I'm sorry, the server took too long to respond. Please try again or refresh the page if the problem persists.",
            sender: 'assistant' as const,
            timestamp: new Date(),
            mode: currentMode,
          };
          
          setMessages((prevMessages) => {
            const updatedMessages = [...prevMessages, timeoutMessage];
            saveMessagesToLocalStorage(updatedMessages);
            return updatedMessages;
          });
          
          // Try to reconnect the WebSocket
          if (ws.current?.readyState !== WebSocket.OPEN) {
            connectWebSocket();
          }
        }
      }, 30000);
      
      // Store timeout ID to clear it if we get a response
      ws.current.responseTimeout = responseTimeout;
      
      // Focus the input after sending
      if (inputRef.current) {
        inputRef.current.focus();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setIsProcessing(false);
      
      // Add an error message if sending fails
      const errorMessage = {
        id: uuidv4(),
        text: "There was an error sending your message. Please try again.",
        sender: 'assistant' as const,
        timestamp: new Date(),
        mode: currentMode,
      };
      
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages, errorMessage];
        saveMessagesToLocalStorage(updatedMessages);
        return updatedMessages;
      });
    }
  };

  const handleSend = () => {
    sendMessage(input);
  };

  // Handle model selection
  const selectModel = (model: ModelInfo) => {
    setSelectedModel(model);
    setShowModelSelector(false);
  };

  // Handle mode switching
  const switchMode = (mode: 'chat' | 'create') => {
    if (mode === currentMode) return;
    
    setCurrentMode(mode);
    
    // Reset the WebSocket connection when switching modes
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.close();
    }
    
    // Auto-select appropriate model for the mode
    if (mode === 'create') {
      setUseAgentSystem(true);
      // Select Llama-3 70B for creative tasks
      const creativeModel = availableModels.find(m => m.name === 'llama3-70b-8192');
      if (creativeModel) {
        setSelectedModel(creativeModel);
      }
    } else {
      // For chat mode, use default model
      const chatModel = availableModels.find(m => m.name === 'llama3-8b-8192');
      if (chatModel) {
        setSelectedModel(chatModel);
      }
    }
  };

  // Render UI elements
  const renderCommandPalette = () => {
    return (
      <div className="flex items-center gap-2 mb-2">
        <div className="flex items-center space-x-2 rounded-full bg-gray-100 p-1">
          <button
            onClick={() => switchMode('chat')}
            className={`px-3 py-1 text-sm rounded-full ${currentMode === 'chat' 
              ? 'bg-blue-500 text-white' 
              : 'bg-transparent text-gray-700 hover:bg-gray-200'}`}
          >
            Chat
          </button>
          <button
            onClick={() => switchMode('create')}
            className={`px-3 py-1 text-sm rounded-full ${currentMode === 'create' 
              ? 'bg-blue-500 text-white' 
              : 'bg-transparent text-gray-700 hover:bg-gray-200'}`}
          >
            Create
          </button>
        </div>

        {currentMode === 'chat' && (
          <div className="relative inline-block">
            <button 
              className="flex items-center gap-1 px-3 py-1 rounded-full bg-amber-200 text-amber-800 border border-amber-300 hover:bg-amber-300 transition-colors font-medium"
              onClick={() => setShowModelSelector(!showModelSelector)}
              title="Select AI model"
            >
              <span>{selectedModel.displayName}</span>
              <ChevronDown size={14} />
            </button>
            
            {showModelSelector && (
              <div className="absolute bottom-full mb-2 left-0 w-60 bg-white rounded-md shadow-lg z-10 border border-gray-200 max-h-72 overflow-y-auto">
                <div className="sticky top-0 bg-gray-50 p-2 border-b text-xs font-medium text-gray-700">
                  Select AI Model
                </div>
                <ul className="py-1">
                  {availableModels.map((model) => (
                    <li 
                      key={`${model.provider}-${model.name}`}
                      className={`px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer ${
                        selectedModel.name === model.name ? 'bg-blue-50 text-blue-600 font-medium' : 'text-gray-700'
                      }`}
                      onClick={() => selectModel(model)}
                    >
                      <div className="flex items-center justify-between">
                        <span>{model.displayName}</span>
                        {selectedModel.name === model.name && (
                          <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // Fallback to REST API if WebSocket isn't connected
  const sendMessageREST = async (text: string) => {
    if (!text.trim()) return;
    
    try {
      // Parse any command in the text
      const parsedText = parseCommand(text);
      
      // Show user message immediately
      const tempId = uuidv4();
      const userMessage = {
        id: tempId,
        text: parsedText,
        sender: 'user' as const,
        timestamp: new Date(),
        mode: currentMode,
      };
      
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setInput('');
      setIsProcessing(true);
      
      // Focus the input after sending
      if (inputRef.current) {
        inputRef.current.focus();
      }
      
      // Send message to backend
      const response = await fetch('http://127.0.0.1:8000/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: parsedText,
          user_id: clientId.current,
          mode: currentMode,
          model: selectedModel,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setIsProcessing(false);
      
      // Replace our temporary message with the server's version and add the AI response
      setMessages((prevMessages) => {
        // Remove the temporary message
        const filteredMessages = prevMessages.filter((msg) => msg.id !== tempId);
        
        // Add both messages from the server
        const serverMessages = data.map((msg: ServerMessage) => ({
          ...msg,
          timestamp: new Date(msg.timestamp * 1000),
          mode: currentMode,
        }));
        
        return [...filteredMessages, ...serverMessages];
      });
      
    } catch (error) {
      console.error('Error sending message via REST API:', error);
      setIsProcessing(false);
      
      // Add a fallback AI response if the request failed
      const aiMessage = {
        id: uuidv4(),
        text: "Sorry, I couldn't reach the server. Please try again later.",
        sender: 'assistant' as const,
        timestamp: new Date(),
        mode: currentMode,
      };
      
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    }
  };

  // Return loading state if not mounted
  if (!isMounted) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="p-4 text-gray-500">Loading chat interface...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white border-l shadow-sm">
      <div className="p-4 border-b bg-gradient-to-r from-blue-50 to-indigo-50 sticky top-0 z-10">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-800">
            {currentMode === 'chat' ? 'TLDraw Chat' : 'TLDraw Create'}
          </h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">
              {isConnected ? 'Connected' : isConnecting ? 'Connecting...' : 'Offline'}
            </span>
            <span 
              className={`inline-block w-3 h-3 rounded-full ${
                isConnected ? 'bg-green-500' : isConnecting ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              title={isConnected ? 'Online' : isConnecting ? 'Connecting...' : 'Offline'}
            ></span>
          </div>
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        <div ref={messagesContainerRef} className="h-[calc(100vh-12rem)] overflow-y-auto overflow-x-hidden px-4 py-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-6">
              <div className="w-16 h-16 rounded-full bg-blue-50 flex items-center justify-center mb-3">
                <Send size={24} className="text-blue-400" />
              </div>
              <p className="text-gray-500 mb-1">No messages yet</p>
              <p className="text-sm text-gray-400">Send a message to start the conversation</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={message.id || index}
                  className={`group flex ${
                    message.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`relative max-w-[85%] rounded-2xl px-4 py-3 ${
                      message.sender === 'user'
                        ? 'bg-blue-500 text-white rounded-tr-none'
                        : isStatusUpdate(message)
                          ? 'bg-gray-100 text-gray-600 italic rounded-tl-none border border-gray-200 flex items-center'
                          : 'bg-gray-100 text-gray-800 rounded-tl-none'
                    }`}
                  >
                    {/* Copy button */}
                    <div className={`absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity ${
                      message.sender === 'user' ? 'text-white' : 'text-gray-500'
                    }`}>
                      <CopyButton text={message.text} />
                    </div>
                    
                    <div className="flex flex-col">
                      {isStatusUpdate(message) ? (
                        <div className="flex items-center space-x-2">
                          <RefreshCw size={14} className="animate-spin" />
                          <span className="break-words">{message.text}</span>
                        </div>
                      ) : (
                        <>
                          {message.sender === 'user' ? (
                            <span className="break-words whitespace-pre-wrap pr-6">{message.text}</span>
                          ) : (
                            <div className="pr-6">
                              <MarkdownRenderer content={message.text} />
                            </div>
                          )}
                          
                          {/* Display agent details if this is the last assistant message and we have details */}
                          {message.mode === 'create' && 
                           message.sender === 'assistant' && 
                           agentDetails && 
                           agentDetails.id === message.id && (
                            <div className="mt-3 pt-2 border-t border-gray-200 text-xs text-gray-600">
                              <details>
                                <summary className="cursor-pointer font-medium text-blue-600 hover:text-blue-700">
                                  View Creation Details
                                </summary>
                                <div className="mt-2 space-y-3 bg-white p-3 rounded-md border border-gray-200 shadow-sm">
                                  <div>
                                    <h4 className="font-semibold mb-1 text-gray-700">Enhanced Prompt</h4>
                                    <p className="text-gray-600">{agentDetails.enhanced_prompt}</p>
                                  </div>
                                  <div>
                                    <h4 className="font-semibold mb-1 text-gray-700">Research</h4>
                                    <p className="text-gray-600">{agentDetails.research}</p>
                                  </div>
                                  <div>
                                    <h4 className="font-semibold mb-1 text-gray-700">Visualization</h4>
                                    <p className="text-gray-600">{agentDetails.visualization}</p>
                                  </div>
                                  <div>
                                    <h4 className="font-semibold mb-1 text-gray-700">Drawing Instructions</h4>
                                    <p className="text-gray-600">{agentDetails.instructions}</p>
                                  </div>
                                  <div className="mt-2">
                                    <button 
                                      className="text-blue-600 hover:text-blue-800 text-sm font-medium bg-blue-50 hover:bg-blue-100 px-3 py-1 rounded transition-colors"
                                      onClick={() => {
                                        // Attempt to redraw diagram from the saved instructions
                                        if (agentDetails) {
                                          generateDiagramFromInstructions({
                                            visualization: agentDetails.visualization,
                                            instructions: agentDetails.instructions
                                          });
                                        }
                                      }}
                                    >
                                      Redraw Diagram
                                    </button>
                                  </div>
                                </div>
                              </details>
                            </div>
                          )}
                        </>
                      )}
                      
                      {!isStatusUpdate(message) && (
                        <span 
                          className={`text-xs mt-1 ${
                            message.sender === 'user' ? 'text-blue-100' : 'text-gray-500'
                          }`}
                        >
                          {formatTime(message.timestamp)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>
      
      <div className="p-3 border-t bg-white">
        {isMounted && renderCommandPalette()}
        <div className="flex items-center gap-2 bg-gray-50 p-1 rounded-full shadow-sm">
          {isMounted && (
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 px-4 py-2 bg-transparent border-none focus:outline-none text-gray-800 placeholder:text-gray-400"
              placeholder={
                currentMode === 'chat' 
                  ? "Ask me anything..." 
                  : "Describe what you'd like to create..."
              }
              disabled={isProcessing}
            />
          )}
          {isMounted && (
            <Button 
              onClick={isConnected ? handleSend : () => sendMessageREST(input)}
              type="button"
              className="rounded-full h-9 w-9 p-0 bg-blue-500 hover:bg-blue-600"
              disabled={!input.trim() || isProcessing}
            >
              {isProcessing ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
} 