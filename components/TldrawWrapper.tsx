import React, { useEffect, useRef } from 'react';
import { Tldraw, Editor, TLShape, createShapeId } from '@tldraw/tldraw';
import '@tldraw/tldraw/tldraw.css';

// Define a global interface to access the editor externally
declare global {
  interface Window {
    tldrawEditor: Editor | null;
  }
}

export default function TldrawWrapper() {
  const editorRef = useRef<Editor | null>(null);

  // Create shapes based on instructions from the AI
  const createShapesFromInstructions = (instructions: any) => {
    if (!editorRef.current) {
      console.error('TLDraw editor not initialized yet');
      return;
    }
    
    try {
      console.log('Parsing instructions to create diagram:', instructions);
      
      // Clear the canvas first
      editorRef.current.deleteAll();
      
      // Extract instruction text and visualization hints
      const instructionsText = typeof instructions === 'object' && instructions.instructions 
        ? instructions.instructions 
        : typeof instructions === 'string' 
          ? instructions 
          : '';
      
      const visualization = typeof instructions === 'object' && instructions.visualization 
        ? instructions.visualization 
        : '';
      
      console.log('Using instructions text:', instructionsText);
      console.log('Using visualization hints:', visualization);
      
      // Parse the instructions to identify shapes, colors, and positions
      const parseInstructions = () => {
        // Look for common diagram elements in the instructions
        const isFlowchart = /flowchart|flow chart|flow diagram/i.test(instructionsText);
        const isProcessDiagram = /process diagram|process flow/i.test(instructionsText);
        const isOddEvenDiagram = /odd[- ]even|even[- ]odd|parity/i.test(instructionsText);
        
        // Extract color information from visualization text
        const colorMatch = visualization.match(/#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}|rgba?\([^)]+\)|hsla?\([^)]+\)|[a-zA-Z]+/g);
        let colors = colorMatch ? [...new Set(colorMatch)] : [];
        
        // Filter out non-color words that might have matched
        colors = colors.filter(color => 
          /^#[0-9A-Fa-f]{6}$|^#[0-9A-Fa-f]{3}$|^rgba?\([^)]+\)$|^hsla?\([^)]+\)$/.test(color) || 
          ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white',
           'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'maroon', 'navy', 'olive', 'aqua',
           'gray', 'grey', 'silver', 'gold', 'turquoise'].includes(color.toLowerCase())
        );
        
        // If no colors found, use default pleasant palette
        if (colors.length < 3) {
          colors = ['#C7E4F4', '#FFD166', '#06D6A0', '#EF476F', '#118AB2'];
        }
        
        return {
          type: isFlowchart ? 'flowchart' : isProcessDiagram ? 'process' : isOddEvenDiagram ? 'odd-even' : 'generic',
          colors
        };
      };
      
      const parsedInstructions = parseInstructions();
      
      // Create diagram based on type
      if (parsedInstructions.type === 'odd-even') {
        createOddEvenFlowchart(parsedInstructions.colors);
      } else if (parsedInstructions.type === 'flowchart' || parsedInstructions.type === 'process') {
        createGenericFlowchart(parsedInstructions.colors, instructionsText);
      } else {
        createGenericDiagram(parsedInstructions.colors, instructionsText);
      }
      
      // Zoom to fit the created diagram
      editorRef.current.zoomToFit();
    } catch (error) {
      console.error('Error creating shapes:', error);
      
      // Create a simple error message on the canvas
      if (editorRef.current) {
        editorRef.current.createShape({
          type: 'text',
          x: 200,
          y: 200,
          props: {
            text: 'Error creating diagram. Please try again.',
            font: 'draw',
            size: 'xl',
            color: 'red',
            textAlign: 'middle',
          },
        });
      }
    }
  };
  
  // Create a standard odd-even number detection flowchart
  const createOddEvenFlowchart = (colors: string[]) => {
    if (!editorRef.current) return;
    
    // 1. Create Input Box
    const inputId = createShapeId('input');
    editorRef.current.createShape({
      id: inputId,
      type: 'geo',
      x: 200,
      y: 100,
      props: {
        geo: 'rectangle',
        w: 200,
        h: 80,
        text: 'Input Number',
        fill: colors[0] || '#C7E4F4',
        color: 'black',
      },
    });

    // 2. Create Decision Diamond
    const decisionId = createShapeId('decision');
    editorRef.current.createShape({
      id: decisionId,
      type: 'geo',
      x: 200,
      y: 250,
      props: {
        geo: 'diamond',
        w: 200,
        h: 120,
        text: 'Number % 2 == 0?',
        fill: colors[1] || '#FFD166',
        color: 'black',
      },
    });

    // 3. Create Even Result Box
    const evenId = createShapeId('even');
    editorRef.current.createShape({
      id: evenId,
      type: 'geo',
      x: 350,
      y: 400,
      props: {
        geo: 'rectangle',
        w: 150,
        h: 60,
        text: 'Even Number',
        fill: colors[2] || '#06D6A0',
        color: 'black',
      },
    });

    // 4. Create Odd Result Box
    const oddId = createShapeId('odd');
    editorRef.current.createShape({
      id: oddId,
      type: 'geo',
      x: 50,
      y: 400,
      props: {
        geo: 'rectangle',
        w: 150,
        h: 60,
        text: 'Odd Number',
        fill: colors[3] || '#EF476F',
        color: 'black',
      },
    });

    // 5. Create Arrow from Input to Decision
    editorRef.current.createShape({
      type: 'arrow',
      props: {
        start: {
          type: 'binding',
          boundShapeId: inputId,
          normalizedAnchor: { x: 0.5, y: 1 },
        },
        end: {
          type: 'binding',
          boundShapeId: decisionId,
          normalizedAnchor: { x: 0.5, y: 0 },
        },
      },
    });

    // 6. Create Arrow from Decision to Even
    editorRef.current.createShape({
      type: 'arrow',
      props: {
        start: {
          type: 'binding',
          boundShapeId: decisionId,
          normalizedAnchor: { x: 0.8, y: 0.7 },
        },
        end: {
          type: 'binding',
          boundShapeId: evenId,
          normalizedAnchor: { x: 0.5, y: 0 },
        },
        text: 'Yes',
      },
    });

    // 7. Create Arrow from Decision to Odd
    editorRef.current.createShape({
      type: 'arrow',
      props: {
        start: {
          type: 'binding',
          boundShapeId: decisionId,
          normalizedAnchor: { x: 0.2, y: 0.7 },
        },
        end: {
          type: 'binding',
          boundShapeId: oddId,
          normalizedAnchor: { x: 0.5, y: 0 },
        },
        text: 'No',
      },
    });
  };
  
  // Create a generic flowchart based on AI instructions
  const createGenericFlowchart = (colors: string[], instructionsText: string) => {
    if (!editorRef.current) return;
    
    // Parse the text to identify potential flowchart elements
    const steps = extractFlowchartSteps(instructionsText);
    
    // Create the shapes based on the extracted steps
    const shapeIds: Record<string, string> = {};
    const startY = 100;
    const startX = 300;
    const stepHeight = 80;
    const stepSpacing = 100;
    
    // Create nodes first
    steps.forEach((step, index) => {
      const id = createShapeId(`step-${index}`);
      shapeIds[step.id] = id;
      
      // Determine shape type
      let shapeType = 'rectangle';
      if (step.type === 'decision') {
        shapeType = 'diamond';
      } else if (step.type === 'start' || step.type === 'end') {
        shapeType = 'ellipse';
      }
      
      // Create shape
      editorRef.current!.createShape({
        id,
        type: 'geo',
        x: startX,
        y: startY + (index * (stepHeight + stepSpacing)),
        props: {
          geo: shapeType,
          w: shapeType === 'diamond' ? 220 : 200,
          h: shapeType === 'diamond' ? 120 : stepHeight,
          text: step.text,
          fill: colors[index % colors.length] || '#C7E4F4',
          color: 'black',
        },
      });
    });
    
    // Then create connections
    steps.forEach((step, index) => {
      if (step.connections.length > 0) {
        step.connections.forEach(connection => {
          if (shapeIds[connection.to]) {
            editorRef.current!.createShape({
              type: 'arrow',
              props: {
                start: {
                  type: 'binding',
                  boundShapeId: shapeIds[step.id],
                  normalizedAnchor: { x: 0.5, y: 1 },
                },
                end: {
                  type: 'binding',
                  boundShapeId: shapeIds[connection.to],
                  normalizedAnchor: { x: 0.5, y: 0 },
                },
                text: connection.label || '',
              },
            });
          }
        });
      }
    });
  };
  
  // Create a more freeform diagram based on general AI instructions
  const createGenericDiagram = (colors: string[], instructionsText: string) => {
    if (!editorRef.current) return;
    
    // Create a title for the diagram
    editorRef.current.createShape({
      type: 'text',
      x: 200,
      y: 50,
      props: {
        text: 'Diagram Generated by AI',
        font: 'draw',
        size: 'xl',
        color: 'black',
        textAlign: 'middle',
      },
    });
    
    // Create a central node
    const centralId = createShapeId('central');
    editorRef.current.createShape({
      id: centralId,
      type: 'geo',
      x: 300,
      y: 200,
      props: {
        geo: 'rectangle',
        w: 180,
        h: 80,
        text: 'Main Concept',
        fill: colors[0] || '#C7E4F4',
        color: 'black',
      },
    });
    
    // Create surrounding elements based on keywords in the instructions
    // Extract keywords from instructions
    const keywords = extractKeywords(instructionsText);
    
    // Create surrounding shapes for the keywords
    keywords.forEach((keyword, index) => {
      if (index < 5) { // Limit to 5 surrounding shapes
        const angle = (2 * Math.PI * index) / keywords.length;
        const distance = 200; // Distance from center
        
        const x = 300 + Math.cos(angle) * distance;
        const y = 200 + Math.sin(angle) * distance;
        
        const nodeId = createShapeId(`node-${index}`);
        
        // Create the shape
        editorRef.current!.createShape({
          id: nodeId,
          type: 'geo',
          x,
          y,
          props: {
            geo: index % 2 === 0 ? 'rectangle' : 'ellipse',
            w: 140,
            h: 70,
            text: keyword,
            fill: colors[(index + 1) % colors.length] || '#FFD166',
            color: 'black',
          },
        });
        
        // Connect to central node
        editorRef.current!.createShape({
          type: 'arrow',
          props: {
            start: {
              type: 'binding',
              boundShapeId: centralId,
              normalizedAnchor: { 
                x: 0.5 + 0.4 * Math.cos(angle), 
                y: 0.5 + 0.4 * Math.sin(angle) 
              },
            },
            end: {
              type: 'binding',
              boundShapeId: nodeId,
              normalizedAnchor: { 
                x: 0.5 - 0.4 * Math.cos(angle), 
                y: 0.5 - 0.4 * Math.sin(angle) 
              },
            },
          },
        });
      }
    });
  };
  
  // Helper function to extract flowchart steps from instructions text
  const extractFlowchartSteps = (instructionsText: string) => {
    // Simple regex-based parser for flowchart steps
    const stepRegex = /(?:Step|^\d+[\.\)]+)[^\n]+(.*?)(?=(?:Step|\d+[\.\)]+|$))/gmi;
    const decisionKeywords = ['if', 'check', 'condition', 'decide', 'determine', 'compare'];
    
    const matches = instructionsText.match(stepRegex) || [];
    
    // Extract steps from the text
    const steps = matches.map((step, index) => {
      // Clean up the step text
      const cleanStep = step.replace(/^(?:Step|^\d+[\.\)]+)[^\n]+/mi, '').trim();
      
      // Determine step type
      let type = 'process';
      if (index === 0) {
        type = 'start';
      } else if (index === matches.length - 1) {
        type = 'end';
      } else if (decisionKeywords.some(keyword => cleanStep.toLowerCase().includes(keyword))) {
        type = 'decision';
      }
      
      // Create a step object
      return {
        id: `step-${index}`,
        text: cleanStep.length > 50 ? cleanStep.substring(0, 47) + '...' : cleanStep,
        type,
        connections: index < matches.length - 1 ? [{ to: `step-${index + 1}` }] : []
      };
    });
    
    // If no steps were extracted, create a basic set
    if (steps.length === 0) {
      return [
        { id: 'start', text: 'Start', type: 'start', connections: [{ to: 'process' }] },
        { id: 'process', text: 'Process', type: 'process', connections: [{ to: 'end' }] },
        { id: 'end', text: 'End', type: 'end', connections: [] }
      ];
    }
    
    return steps;
  };
  
  // Helper function to extract keywords from text
  const extractKeywords = (text: string) => {
    // Split text into words
    const words = text.split(/\s+/);
    
    // Filter out common words and keep only meaningful nouns and verbs
    const commonWords = ['the', 'and', 'is', 'of', 'to', 'in', 'a', 'for', 'with', 'as', 'on', 'at', 'by', 'an', 'or', 'but'];
    
    const keywords = words
      .filter(word => word.length > 3) // Only words longer than 3 chars
      .filter(word => !commonWords.includes(word.toLowerCase())) // Remove common words
      .map(word => word.replace(/[.,;:!?]$/, '')); // Remove trailing punctuation
    
    // Get unique keywords
    const uniqueKeywords = [...new Set(keywords)];
    
    // Return top keywords
    return uniqueKeywords.slice(0, 7); // Limit to top 7 keywords
  };

  // Handle messages from chat
  useEffect(() => {
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
            data = JSON.parse(event.data);
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
          createShapesFromInstructions(data.instructions);
        }
      } catch (e) {
        console.error('Error processing message in TldrawWrapper:', e);
      }
    };

    // Listen for messages from the chat component
    console.log('Setting up message listeners in TldrawWrapper');
    window.addEventListener('message', handleMessage);
    
    // Also listen for custom events from ChatInterface (for backward compatibility)
    window.addEventListener('tlDrawDiagram', handleMessage as EventListener);
    
    return () => {
      window.removeEventListener('message', handleMessage);
      window.removeEventListener('tlDrawDiagram', handleMessage as EventListener);
    };
  }, []);

  const handleMount = (editor: Editor) => {
    console.log('TLDraw editor mounted');
    // Store editor reference
    editorRef.current = editor;
    // Also make it available globally for debugging and external access
    window.tldrawEditor = editor;
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 bg-gradient-to-r from-white to-blue-50 border-b flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Drawing Board</h1>
          <p className="text-xs text-gray-500">Create and collaborate</p>
        </div>
      </div>
      <div className="flex-1">
        <Tldraw onMount={handleMount} />
      </div>
    </div>
  );
} 