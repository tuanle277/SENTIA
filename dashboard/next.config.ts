import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone', // Enable standalone mode for Docker
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
