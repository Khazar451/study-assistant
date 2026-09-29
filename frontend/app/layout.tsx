import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Study Assistant · Grounded Academic RAG Tutor",
  description:
    "Zero-hallucination, high-precision academic question answering and active recall exam tutor powered by NVIDIA NIM and Nebius AI Studio.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>{children}</body>
    </html>
  );
}
