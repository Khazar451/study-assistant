import type { NextConfig } from "next";

// STATIC_EXPORT=1 builds plain HTML/JS into frontend/out, which FastAPI serves itself
// (single-process hosting, e.g. a Hugging Face Gradio-SDK Space). Otherwise Next proxies /api to FastAPI.
const staticExport = process.env.STATIC_EXPORT === "1";

const nextConfig: NextConfig = staticExport
  ? { output: "export", images: { unoptimized: true } }
  : {
      experimental: {
        proxyTimeout: 300_000,
        proxyClientMaxBodySize: "200mb",
      },
      async rewrites() {
        const apiUrl = process.env.API_URL || "http://127.0.0.1:8000";
        return [
          {
            source: "/api/:path*",
            destination: `${apiUrl}/api/:path*`,
          },
        ];
      },
    };

export default nextConfig;
