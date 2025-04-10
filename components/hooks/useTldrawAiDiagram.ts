import { useCallback, useState } from 'react';
import { Editor, TLShapeId } from '@tldraw/tldraw';

// Define a type for changes
interface ShapeChange {
  id: string;
  type: string;
  [key: string]: any;
}

// Define custom shape format
interface ValidShape {
  id: string;
  type: string;
  x: number;
  y: number;
  rotation: number;
  isLocked: boolean;
  opacity: number;
  props: Record<string, any>;
  parentId?: string; // Optional parentId property
}

// Define interface for generate parameters to match @tldraw/ai types
interface GenerateParams {
  editor: Editor;
  prompt: string | Record<string, any>;
  signal?: AbortSignal;
}

/**
 * TLDraw AI options for configuration
 */
export const TLDRAW_AI_OPTIONS = {
  transforms: [],
  
  // Generate function for non-streaming diagram generation
  generate: async ({ editor, prompt, signal }: GenerateParams): Promise<any[]> => {
    const promptText = typeof prompt === 'string' ? prompt : JSON.stringify(prompt);
    
    if (!promptText.trim()) {
      throw new Error('Please provide a prompt');
    }

    // Prepare the request data
    const requestData = {
      prompt: promptText,
      locale: 'en',
    };

    // Make a request to the backend API
    const response = await fetch('/api/generate-diagram', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
      signal,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Request failed with status ${response.status}`);
    }

    const data = await response.json();
    
    // Ensure data.changes is an array of valid shape objects
    if (!data.changes || !Array.isArray(data.changes)) {
      throw new Error('Invalid response format: changes is not an array');
    }

    console.log('Backend response:', JSON.stringify(data, null, 2));

    // Validate each change object to ensure it has required properties
    const validChanges = data.changes
      .filter((change: ShapeChange) => {
        return (
          change &&
          typeof change === 'object' &&
          'id' in change &&
          'type' in change &&
          typeof change.id === 'string' &&
          typeof change.type === 'string'
        );
      })
      .map((change: ShapeChange): ValidShape => {
        // Ensure each change has the necessary structure expected by TLDraw
        const validChange: ValidShape = {
          id: change.id,
          type: change.type,
          x: change.x || 0,
          y: change.y || 0,
          rotation: change.rotation || 0,
          isLocked: change.isLocked || false,
          opacity: change.opacity || 1,
          props: change.props || {}
        };
        
        // If the shape has a parentId property, include it
        if (change.parentId) {
          validChange.parentId = change.parentId;
        }
        
        return validChange;
      });

    if (validChanges.length === 0) {
      throw new Error('No valid shape changes found in the response');
    }

    // Transform into the format expected by TLDraw AI's applyChanges function
    return validChanges.map((shape: ValidShape) => ({
      type: "createShape",
      shape
    }));
  },
  
  // Stream function for streaming diagram generation
  stream: async function* ({ editor, prompt, signal }: GenerateParams) {
    const promptText = typeof prompt === 'string' ? prompt : JSON.stringify(prompt);
    
    if (!promptText.trim()) {
      throw new Error('Please provide a prompt');
    }

    try {
      // Prepare the request data
      const requestData = {
        prompt: promptText,
        locale: 'en',
      };

      // Make a request to the backend API
      const response = await fetch('/api/generate-diagram', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      
      // Ensure data.changes is an array of valid shape objects
      if (!data.changes || !Array.isArray(data.changes)) {
        throw new Error('Invalid response format: changes is not an array');
      }

      console.log('Backend response for streaming:', JSON.stringify(data, null, 2));

      // Validate each change object to ensure it has required properties
      const validChanges = data.changes
        .filter((change: ShapeChange) => {
          return (
            change &&
            typeof change === 'object' &&
            'id' in change &&
            'type' in change &&
            typeof change.id === 'string' &&
            typeof change.type === 'string'
          );
        })
        .map((change: ShapeChange): ValidShape => {
          // Ensure each change has the necessary structure expected by TLDraw
          const validChange: ValidShape = {
            id: change.id,
            type: change.type,
            x: change.x || 0,
            y: change.y || 0,
            rotation: change.rotation || 0,
            isLocked: change.isLocked || false,
            opacity: change.opacity || 1,
            props: change.props || {}
          };
          
          // If the shape has a parentId property, include it
          if (change.parentId) {
            validChange.parentId = change.parentId;
          }
          
          return validChange;
        });

      if (validChanges.length === 0) {
        throw new Error('No valid shape changes found in the response');
      }

      // Yield each change individually to simulate streaming
      for (const shape of validChanges) {
        // For each change, we need to yield a proper change action with "createShape" type
        // This is what tldraw/ai expects according to their documentation
        yield {
          type: "createShape",
          shape
        };
        
        // Small delay to simulate streaming
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    } catch (error) {
      console.error('Error in stream function:', error);
      throw error;
    }
  }
};

/**
 * Custom hook for using TLDraw AI diagram generation
 */
export const useTldrawAiDiagram = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Generate a diagram based on the prompt
   */
  const generate = useCallback(
    async ({ editor, prompt, signal }: { editor: Editor; prompt: string; signal?: AbortSignal }) => {
      if (!prompt.trim()) {
        setError('Please provide a prompt');
        return;
      }

      try {
        setIsLoading(true);
        setError(null);

        // Prepare the request data
        const requestData = {
          prompt,
          locale: 'en',
        };

        // Make a request to the backend API
        const response = await fetch('/api/generate-diagram', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestData),
          signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || `Request failed with status ${response.status}`);
        }

        const data = await response.json();
        
        // Ensure data.changes is an array of valid shape objects
        if (!data.changes || !Array.isArray(data.changes)) {
          throw new Error('Invalid response format: changes is not an array');
        }

        // Validate each change object to ensure it has required properties
        const validChanges = data.changes.filter((change: ShapeChange) => {
          return (
            change &&
            typeof change === 'object' &&
            'id' in change &&
            'type' in change &&
            typeof change.id === 'string' &&
            typeof change.type === 'string'
          );
        });

        if (validChanges.length === 0) {
          throw new Error('No valid shape changes found in the response');
        }

        // Apply the changes to the editor
        editor.updateShapes(validChanges);

        // Select all shapes that were created
        const shapeIds: TLShapeId[] = validChanges.map((change: ShapeChange) => change.id as TLShapeId);
        if (shapeIds.length > 0) {
          editor.select(...shapeIds);
        }

        return {
          success: true,
        };
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        console.error('Error generating diagram:', err);
        setError(errorMessage);
        return {
          success: false,
          error: errorMessage,
        };
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    generate,
    isLoading,
    error,
  };
}; 