import React, { useEffect, useRef, useState } from 'react';
import { Tldraw, Editor, TLShape, createShapeId, useEditor } from '@tldraw/tldraw';
import '@tldraw/tldraw/tldraw.css';
// @ts-ignore - Ignore type checking for this import
import { useTldrawAi } from '@tldraw/ai';
import { TLDRAW_AI_OPTIONS } from './hooks/useTldrawAiDiagram';

// Define a global interface to access the editor externally
declare global {
  interface Window {
    tldrawEditor: Editor | null;
  }
}

// Define interfaces for flow chart steps and connections for backward compatibility
interface FlowChartStep {
  id: string;
  text: string;
  type: string;
  connections: FlowChartConnection[];
}

interface FlowChartConnection {
  to: string;
  label?: string;
}

export default function TldrawWrapper() {
  // Add a state to track if component is mounted (client-side only)
  const [isMounted, setIsMounted] = useState(false);

  // Set component as mounted only on client-side
  useEffect(() => {
    setIsMounted(true);
    return () => {
      // Cleanup global reference when component unmounts
      if (typeof window !== 'undefined') {
        window.tldrawEditor = null;
      }
    };
  }, []);

  // If not mounted yet, return a loading placeholder
  if (!isMounted) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="p-4 text-gray-500">Loading drawing canvas...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 bg-gradient-to-r from-white to-blue-50 border-b flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Drawing Board</h1>
          <p className="text-xs text-gray-500">Create and collaborate</p>
        </div>
      </div>
      <div className="flex-1 relative">
        <Tldraw
          onMount={(editor) => {
            console.log('TLDraw editor mounted');
            // Make editor available globally for debugging and external access
            if (typeof window !== 'undefined') {
              window.tldrawEditor = editor;
            }
          }}
          components={{
            ErrorFallback: () => <div className="p-4 text-red-500">Error loading tldraw editor. Please refresh the page.</div>
          }} 
        >
          <TldrawAIHandler />
        </Tldraw>
      </div>
    </div>
  );
}

/**
 * This component must be a child of Tldraw to use the useEditor hook
 */
function TldrawAIHandler() {
  const editor = useEditor();
  const ai = useTldrawAi(TLDRAW_AI_OPTIONS);
  
  // Utility function to create shapes with proper TLDraw validations
  const createDiagramElement = (type: string, x: number, y: number, props: any, id?: string) => {
    try {
      const shapeOptions: any = {
        type,
        x,
        y
      };
      
      // Add id if provided
      if (id) {
        shapeOptions.id = id;
      }
      
      // Handle text shapes - remove h property if present
      if (type === 'text' && props.h !== undefined) {
        const { h, ...restProps } = props;
        // Default to autoSize true if not specified
        if (restProps.autoSize === undefined) {
          restProps.autoSize = true;
        }
        shapeOptions.props = restProps;
        return editor.createShape(shapeOptions);
      }
      
      // Handle geo shapes - ensure fill property is valid
      if (type === 'geo' && props.fill !== undefined && 
          !['none', 'semi', 'solid', 'pattern', 'fill'].includes(props.fill)) {
        // Store the original fill value as a stroke color
        const strokeColor = props.fill;
        const { fill, ...restProps } = props;
        shapeOptions.props = {
          ...restProps,
          fill: 'solid', // Default to solid fill
          stroke: strokeColor // Use the original color as stroke
        };
        return editor.createShape(shapeOptions);
      }
      
      // Default case - pass props as is
      shapeOptions.props = props;
      return editor.createShape(shapeOptions);
    } catch (error) {
      console.error(`Error creating ${type} shape:`, error);
      console.log('Attempted with props:', props);
      // Return null or a dummy ID to indicate failure
      return null;
    }
  };

  // Function to generate diagram using tldraw/ai
  const generateDiagramWithAI = async (prompt: string) => {
    if (!editor || !ai) {
      console.error('Editor or AI not initialized');
      return;
    }

    console.log('Generating diagram with AI:', prompt);
    
    try {
      // Clear any existing content
      const currentShapes = editor.getCurrentPageShapeIds();
      
      if (currentShapes.size > 0) {
        editor.deleteShapes([...currentShapes]);
      }
      
      // Use the AI to generate the diagram
      await ai.prompt(prompt);
      
      // Zoom to fit the created diagram
      if (editor && typeof editor.zoomToFit === 'function') {
        editor.zoomToFit();
      }
      
      console.log('Diagram created successfully with AI');
    } catch (error) {
      console.error('Error generating diagram with AI:', error);
      
      // Display error message on canvas
      createDiagramElement('text', 200, 200, {
        w: 320,
        text: `Error generating diagram: ${error instanceof Error ? error.message : 'Unknown error'}`,
        color: 'red',
        size: 'l',
        font: 'draw',
        align: 'middle',
        autoSize: true,
      });
    }
  };

  // Handle messages from chat
  useEffect(() => {
    // Define event handler inside useEffect to avoid stale closures
    const handleMessage = (event: MessageEvent | Event) => {
      try {
        console.log('TldrawWrapper received message event:', event);
        
        // Check if data is already a JavaScript object or needs to be parsed
        let data;
        
        if (event instanceof CustomEvent && event.detail) {
          // Handle CustomEvent with detail property
          console.log('Processing CustomEvent with detail:', event.detail);
          data = event.detail;
        } else if (event instanceof MessageEvent) {
          if (typeof event.data === 'string') {
            // Handle string data that needs parsing
            console.log('Parsing string data from MessageEvent');
            try {
              data = JSON.parse(event.data);
            } catch (parseError) {
              console.error('Error parsing message data:', parseError);
              console.log('Raw message data:', event.data);
              return;
            }
          } else {
            // Assume data is already an object
            console.log('Using object data from MessageEvent');
            data = event.data;
          }
        } else {
          console.warn('Unrecognized event type:', event);
          return;
        }
        
        console.log('TldrawWrapper processed message data:', data);
        
        // Check if it's a message to draw a diagram
        if (data && data.type === 'draw_diagram') {
          console.log('Drawing diagram with instructions:', data.instructions);
          
          if (!data.instructions) {
            console.error('Missing instructions in draw_diagram message');
            
            // Create a simple error message on the canvas
            createDiagramElement('text', 200, 200, {
              w: 320,
              text: 'Error: Missing diagram instructions',
              color: 'red',
              size: 'l',
              font: 'draw',
              align: 'middle',
              autoSize: true,
            });
            return;
          }
          
          // Extract instruction text for the AI
          const prompt = typeof data.instructions === 'object' && data.instructions.instructions 
            ? String(data.instructions.instructions) 
            : typeof data.instructions === 'string' 
              ? data.instructions 
              : '';
          
          if (!prompt) {
            console.error('No valid prompt found in instructions');
            return;
          }
          
          // Use @tldraw/ai to generate the diagram
          generateDiagramWithAI(prompt);
        }
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : 'Unknown error';
        console.error('Error processing message in TldrawWrapper:', e);
        
        // Display error on the canvas
        createDiagramElement('text', 200, 200, {
          w: 320,
          text: `Error: ${errorMessage}`,
          color: 'red',
          size: 'l',
          font: 'draw',
          align: 'middle',
          autoSize: true,
        });
      }
    };

    // Listen for messages from the chat component
    console.log('Setting up message listeners in TldrawWrapper');
    
    // Clean approach: ensure we're on client-side before adding event listeners
    if (typeof window !== 'undefined') {
      window.addEventListener('message', handleMessage);
      window.addEventListener('tlDrawDiagram', handleMessage as EventListener);
      
      return () => {
        window.removeEventListener('message', handleMessage);
        window.removeEventListener('tlDrawDiagram', handleMessage as EventListener);
      };
    }
    
    return undefined;
  }, [editor, ai]); // Add dependencies for editor and ai

  // Utility function to test diagram generation
  const testDiagramGeneration = async () => {
    try {
      console.log('Testing diagram generation');
      await ai.prompt("Create a simple if-else flow diagram for checking user permissions");
      console.log('Diagram generation completed');
    } catch (error) {
      console.error('Error in test diagram generation:', error);
    }
  };
  
  // Add a test button to the UI
  useEffect(() => {
    // Add button only on client side and only once
    if (typeof document !== 'undefined') {
      // Check if the button already exists
      const existingButton = document.getElementById('test-diagram-button');
      if (!existingButton) {
        const button = document.createElement('button');
        button.id = 'test-diagram-button';
        button.innerText = 'Generate Test Diagram';
        button.className = 'fixed bottom-5 right-5 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded z-50';
        button.addEventListener('click', testDiagramGeneration);
        document.body.appendChild(button);
      }
    }
    
    return () => {
      // Clean up
      if (typeof document !== 'undefined') {
        const button = document.getElementById('test-diagram-button');
        if (button) {
          button.removeEventListener('click', testDiagramGeneration);
          button.remove();
        }
      }
    };
  }, [editor, ai]);

  return null; // This component just handles events, doesn't render anything
}