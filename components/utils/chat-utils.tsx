'use client';

import React from 'react';
import { Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

/**
 * Format timestamp as a readable time string
 */
export const formatTime = (date: Date): string => {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/**
 * Auto scroll to bottom of a container
 */
export const scrollToBottom = (ref: React.RefObject<HTMLDivElement | null>) => {
  if (ref.current) {
    ref.current.scrollIntoView({ behavior: 'smooth' });
  }
};

/**
 * Check if a message is a status update
 */
export const isStatusUpdate = (message: any): boolean => {
  return message.isStatusUpdate === true;
};

/**
 * Copy text to clipboard
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy text:', error);
    return false;
  }
};

/**
 * Copy button component with animation
 */
export const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="p-1 rounded-md text-gray-500 hover:bg-gray-100 transition-colors"
      aria-label="Copy to clipboard"
    >
      {copied ? (
        <Check size={16} className="text-green-500" />
      ) : (
        <Copy size={16} />
      )}
    </button>
  );
};

/**
 * Markdown renderer component
 */
export const MarkdownRenderer = ({ content }: { content: string }) => {
  return (
    <div className="prose prose-sm max-w-none break-words">
      <ReactMarkdown
        components={{
          // Customize styling for markdown components
          a: ({ node, ...props }) => (
            <a 
              {...props} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-blue-600 hover:underline"
            />
          ),
          code: ({ node, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '');
            return match ? (
              <div className="relative">
                <pre className="rounded bg-gray-800 p-4 text-sm overflow-x-auto">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
                <CopyButton text={String(children).replace(/\n$/, '')} />
              </div>
            ) : (
              <code className="bg-gray-100 rounded px-1 py-0.5" {...props}>
                {children}
              </code>
            );
          },
          pre: ({ node, ...props }) => (
            <pre className="bg-gray-800 p-4 rounded-md overflow-x-auto relative" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}; 