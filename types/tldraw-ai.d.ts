/**
 * Type declarations for @tldraw/ai package
 */

declare module '@tldraw/ai' {
  import { Editor } from '@tldraw/tldraw';

  export interface GenerateParams {
    editor: Editor;
    prompt: string | Record<string, any>;
    signal?: AbortSignal;
  }

  export interface TldrawAiOptions {
    /**
     * Array of transform functions to apply to the editor instance
     */
    transforms: any[];
    
    /**
     * Function to generate diagram changes based on a prompt
     */
    generate?: (params: GenerateParams) => Promise<any[]>;
    
    /**
     * Function to stream diagram changes based on a prompt
     */
    stream?: (params: GenerateParams) => AsyncGenerator<any, void, unknown>;
  }

  /**
   * Custom hook for using tldraw AI functionality
   */
  export function useTldrawAi(options: TldrawAiOptions): {
    /**
     * Function to send a prompt to the AI and get diagram changes
     */
    prompt: (prompt: string | { message: any[]; stream?: boolean }) => Promise<void>;
    
    /**
     * Function to cancel any ongoing AI operations
     */
    cancel: () => void;
    
    /**
     * Function to repeat the last AI operation
     */
    repeat: () => Promise<void>;
    
    /**
     * Whether the AI is currently processing a prompt
     */
    isLoading: boolean;
    
    /**
     * Error message if an error occurred during processing
     */
    error: Error | null;
  };
} 