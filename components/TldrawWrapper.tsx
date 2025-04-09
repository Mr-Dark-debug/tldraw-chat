import React from 'react';
import { Tldraw } from '@tldraw/tldraw';
import '@tldraw/tldraw/tldraw.css';

export default function TldrawWrapper() {
  return (
    <div className="flex flex-col h-full">
      <div className="p-3 bg-gradient-to-r from-white to-blue-50 border-b flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">Drawing Board</h1>
          <p className="text-xs text-gray-500">Create and collaborate</p>
        </div>
      </div>
      <div className="flex-1">
        <Tldraw />
      </div>
    </div>
  );
} 