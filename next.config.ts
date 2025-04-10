import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  transpilePackages: ['@tldraw/tldraw', '@tldraw/ai'],
  experimental: {
    // Enable other experimental features but not serverActions which was invalid
    esmExternals: true,
  },
  // Configure compiler options
  compiler: {
    // Remove all console.* calls in production
    removeConsole: process.env.NODE_ENV === "production" ? {
      exclude: ['error', 'warn'],
    } : false,
  },
};

export default nextConfig;
