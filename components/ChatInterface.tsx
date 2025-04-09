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

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [useAgentSystem, setUseAgentSystem] = useState(true);
  const [agentDetails, setAgentDetails] = useState<any>(null);
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
  const ws = useRef<WebSocket | null>(null);
  const clientId = useRef<string>(uuidv4());

  // Mark component as mounted for client-side rendering only
  useEffect(() => {
    setIsMounted(true);
    
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
  }, []);

  // Function to establish WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (!isMounted) return; // Avoid WebSocket connection during SSR
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
          ws.current.close();
        }
        ws.current = null;
      }
      
      const socket = new WebSocket(wsUrl);
      
      socket.onopen = () => {
        console.log('WebSocket connection established');
        setIsConnected(true);
        setIsConnecting(false);
      };
      
      socket.onmessage = (event) => {
        try {
          // Parse the message data
          const data = JSON.parse(event.data);
          
          if (Array.isArray(data)) {
            // This is the message history
            const formattedMessages = data.map((msg) => ({
              ...msg,
              timestamp: new Date(msg.timestamp * 1000), // Convert timestamp to Date
            }));
            setMessages(formattedMessages);
          } else if (data.type === 'agent_response') {
            // This is a detailed agent response
            setIsProcessing(false);
            
            // Clean up any error messages for better presentation
            const cleanErrorMessages = (text: string) => {
              if (text && text.includes('Error generating text with')) {
                return "Sorry, I encountered an API error. I'll still try to help you with what I know.";
              }
              return text || '';
            };
            
            // Clean up the content for display
            const enhancedPrompt = cleanErrorMessages(data.enhanced_prompt);
            const research = cleanErrorMessages(data.research);
            const visualization = cleanErrorMessages(data.visualization);
            const instructions = cleanErrorMessages(data.instructions);
            const response = cleanErrorMessages(data.response);
            
            // Create a new AI message if one doesn't exist for this response
            const messageExists = messages.some(msg => msg.id === data.id);
            
            if (!messageExists) {
              const newMessage = {
                id: data.id,
                text: response,
                sender: 'assistant' as const,
                timestamp: new Date(data.timestamp * 1000),
                mode: currentMode,
              };
              
              setMessages(prevMessages => [...prevMessages, newMessage]);
            }
            
            setAgentDetails({
              id: data.id,
              originalMessage: data.original_message,
              enhancedPrompt: enhancedPrompt,
              research: research,
              visualization: visualization,
              instructions: instructions,
              response: response,
              timestamp: new Date(data.timestamp * 1000)
            });

            // Only show error info in console for debugging
            if ((data.enhanced_prompt && data.enhanced_prompt.includes('Error')) || 
                (data.research && data.research.includes('Error')) ||
                (data.visualization && data.visualization.includes('Error')) ||
                (data.instructions && data.instructions.includes('Error'))) {
              console.warn("Agent encountered errors:", {
                enhancedPrompt: data.enhanced_prompt,
                research: data.research,
                visualization: data.visualization,
                instructions: data.instructions
              });
            }
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
              return [...filteredMessages, newMessage];
            });
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      socket.onclose = (event) => {
        console.log(`WebSocket closed with code: ${event.code}, reason: ${event.reason}`);
        setIsConnected(false);
        setIsConnecting(false);
        
        // Only attempt to reconnect if the connection was not closed intentionally
        // and if the component is still mounted
        if (event.code !== 1000 && isMounted) {
          console.log('Attempting to reconnect in 3 seconds...');
          setTimeout(() => {
            if (isMounted) connectWebSocket();
          }, 3000);
        }
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        setIsConnecting(false);
      };
      
      ws.current = socket;
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      setIsConnected(false);
      setIsConnecting(false);
      
      // Try to reconnect after a delay
      setTimeout(() => {
        if (isMounted) connectWebSocket();
      }, 5000);
    }
  }, [isMounted, currentMode, clientId, messages]);

  // Connect to WebSocket on component mount or when mode changes
  useEffect(() => {
    if (isMounted) {
      // Close existing connection if switching endpoints
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
      connectWebSocket();
    }
    
    // Cleanup function to close WebSocket connection
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connectWebSocket, isMounted, currentMode]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom(messagesEndRef);
  }, [messages]);

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
    if (!text.trim() || !ws.current || ws.current.readyState !== WebSocket.OPEN) return;
    
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
      
      // Focus the input after sending
      if (inputRef.current) {
        inputRef.current.focus();
      }
    } catch (error) {
      console.error('Error sending message:', error);
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
                                    <p className="text-gray-600">{agentDetails.enhancedPrompt}</p>
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
        {renderCommandPalette()}
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
          <Button 
            onClick={isMounted ? (isConnected ? handleSend : () => sendMessageREST(input)) : undefined}
            type="button"
            className="rounded-full h-9 w-9 p-0 bg-blue-500 hover:bg-blue-600"
            disabled={!isMounted || !input.trim() || isProcessing}
          >
            {isProcessing ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
} 