'use client';

import { Suspense, lazy } from 'react';
import dynamic from 'next/dynamic';
import { ChatInterface } from '@/components/ChatInterface';
import ErrorBoundary from '@/components/ErrorBoundary';
import ClientOnly from '@/components/ClientOnly';

// Import TLDraw dynamically with SSR disabled
const TldrawWithNoSSR = dynamic(
  () => import('@/components/TldrawWrapper'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="p-4 text-gray-500">Loading drawing canvas...</div>
      </div>
    ) 
  }
);

export default function Home() {
  return (
    <main className="flex h-screen bg-gray-50" suppressHydrationWarning>
      <div className="flex-1 border-r shadow-sm overflow-hidden">
        <ErrorBoundary fallback={
          <div className="flex items-center justify-center h-full">
            <div className="p-4 bg-red-50 border border-red-200 rounded-md max-w-md">
              <h3 className="text-lg font-medium text-red-800 mb-2">Drawing area failed to load</h3>
              <p className="text-sm text-red-600 mb-4">
                There was a problem loading the drawing canvas. Please refresh the page to try again.
              </p>
              <button
                onClick={() => window.location.reload()}
                className="px-3 py-1 bg-red-100 text-red-800 text-sm rounded-md hover:bg-red-200 transition-colors"
              >
                Refresh page
              </button>
            </div>
          </div>
        }>
          <Suspense fallback={
            <div className="flex items-center justify-center h-full bg-gray-50">
              <div className="p-4 text-gray-500">Loading drawing canvas...</div>
            </div>
          }>
            <TldrawWithNoSSR />
          </Suspense>
        </ErrorBoundary>
      </div>
      <div className="w-96 overflow-hidden">
        <ErrorBoundary fallback={
          <div className="p-4 bg-red-50 h-full">
            <h3 className="text-lg font-medium text-red-800 mb-2">Chat failed to load</h3>
            <p className="text-sm text-red-600 mb-4">
              There was a problem loading the chat interface. Please refresh the page to try again.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-3 py-1 bg-red-100 text-red-800 text-sm rounded-md hover:bg-red-200 transition-colors"
            >
              Refresh page
            </button>
          </div>
        }>
          <ClientOnly>
            <ChatInterface />
          </ClientOnly>
        </ErrorBoundary>
      </div>
    </main>
  );
}
