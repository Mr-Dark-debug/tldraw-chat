'use client';

import dynamic from 'next/dynamic';
import { ChatInterface } from '@/components/ChatInterface';

// Import TLDraw dynamically with SSR disabled
const TldrawWithNoSSR = dynamic(
  () => import('@/components/TldrawWrapper'),
  { ssr: false }
);

export default function Home() {
  return (
    <main className="flex h-screen bg-gray-50">
      <div className="flex-1 border-r shadow-sm overflow-hidden">
        <TldrawWithNoSSR />
      </div>
      <div className="w-96 overflow-hidden">
        <ChatInterface />
      </div>
    </main>
  );
}
