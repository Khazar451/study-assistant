"use client";

import React, { useState, useEffect, useRef } from "react";

// --- Clean SVG Icons (Zero Emojis) ---

function IconLogo() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
      <path d="M6 6h10" />
      <path d="M6 10h10" />
    </svg>
  );
}

function IconChat() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />
    </svg>
  );
}

function IconMaterials() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  );
}

function IconFlashcards() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="14" x="3" y="5" rx="2" />
      <path d="M7 15h4" />
      <path d="M7 11h10" />
    </svg>
  );
}

function IconSun() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}

function IconMoon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
  );
}

function IconUpload() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}

function IconFile() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
    </svg>
  );
}

function IconTrash() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  );
}

function IconArrowRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="m12 5 7 7-7 7" />
    </svg>
  );
}

function IconUser() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function IconTutor() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
    </svg>
  );
}

function IconBook() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z" />
    </svg>
  );
}

function IconFlip() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m3 16 4 4 4-4" />
      <path d="M7 20V4" />
      <path d="m21 8-4-4-4 4" />
      <path d="M17 4v16" />
    </svg>
  );
}

// --- Types ---

interface Message {
  id: string;
  sender: "user" | "tutor";
  text: string;
  sources?: string[];
  chunks?: any[];
  durationMs?: number;
  model?: string;
  timestamp: string;
}

interface DocItem {
  source: string;
  file_path: string;
  chunk_count: number;
  page_count: number;
}

interface Flashcard {
  id: string;
  question: string;
  answer: string;
  category: string;
  difficulty: string;
  source: string;
}

interface SystemStatus {
  status: string;
  provider: string;
  nvidia_configured: boolean;
  nebius_configured: boolean;
  indexed_chunks: number;
  persist_directory: string;
  generation_model: string;
}

export default function StudyAssistantApp() {
  // Navigation tabs (RAG pipeline tab removed per requirement)
  const [activeTab, setActiveTab] = useState<"chat" | "materials" | "flashcards">("chat");

  // Theme state: dark / light
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Status state
  const [status, setStatus] = useState<SystemStatus>({
    status: "connecting",
    provider: "NVIDIA NIM",
    nvidia_configured: true,
    nebius_configured: true,
    indexed_chunks: 0,
    persist_directory: "./chroma_db",
    generation_model: "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
  });

  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "tutor",
      text: "Welcome to Study Assistant. Ask any question regarding your course materials to receive grounded explanations with verified page citations.\n\nTip: You can upload your syllabus, notes, or sample Kepler materials in the Course Materials tab.",
      sources: [],
      timestamp: "Just now",
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [augmentMode, setAugmentMode] = useState<"expand" | "rewrite" | "hyde">("expand");
  const [selectedCitation, setSelectedCitation] = useState<any | null>(null);

  // Retrieval configuration
  const topK = 15;
  const topN = 5;

  // Materials state
  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Flashcards state
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [flippedCards, setFlippedCards] = useState<{ [id: string]: boolean }>({});
  const [cardScores, setCardScores] = useState<{ [id: string]: "review" | "mastered" }>({});
  const [isGeneratingCards, setIsGeneratingCards] = useState(false);
  const [flashcardTopic, setFlashcardTopic] = useState("");

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Initialize theme from localStorage or system preference
  useEffect(() => {
    const savedTheme = localStorage.getItem("study-assistant-theme") as "dark" | "light" | null;
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.setAttribute("data-theme", savedTheme);
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
    }
    fetchStatus();
    fetchDocuments();
  }, []);

  // Theme toggle handler
  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("study-assistant-theme", nextTheme);
  };

  useEffect(() => {
    if (activeTab === "chat") {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, activeTab]);

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch {
      // Backend offline or starting
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch("/api/documents");
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch {
      // Ignore
    }
  };

  // Submit academic question via SSE streaming
  const handleAsk = async (queryText?: string) => {
    const q = queryText || inputQuery;
    if (!q.trim() || isGenerating) return;

    setInputQuery("");
    const userMsgId = `user-${Date.now()}`;
    const tutorMsgId = `tutor-${Date.now()}`;

    const userMsg: Message = {
      id: userMsgId,
      sender: "user",
      text: q.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    const initialTutorMsg: Message = {
      id: tutorMsgId,
      sender: "tutor",
      text: "",
      sources: [],
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg, initialTutorMsg]);
    setIsGenerating(true);

    const startTime = Date.now();

    try {
      const url = `/api/ask/stream?query=${encodeURIComponent(q)}&top_k=${topK}&top_n=${topN}&augment_mode=${augmentMode}`;
      const response = await fetch(url);

      if (!response.ok || !response.body) {
        throw new Error("Streaming connection failed.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamedAnswer = "";
      let finalSources: string[] = [];
      let finalChunks: any[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value);
        const lines = chunkText.split("\n\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.replace("data: ", ""));
              if (event.type === "token") {
                streamedAnswer += event.token;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === tutorMsgId ? { ...msg, text: streamedAnswer } : msg
                  )
                );
              } else if (event.type === "chunks") {
                finalChunks = event.chunks || [];
              } else if (event.type === "done") {
                finalSources = event.sources || [];
              }
            } catch {
              // Ignore partial frames
            }
          }
        }
      }

      const durationMs = Date.now() - startTime;
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === tutorMsgId
            ? {
                ...msg,
                text: streamedAnswer || "No response received.",
                sources: finalSources,
                chunks: finalChunks,
                durationMs,
              }
            : msg
        )
      );
    } catch {
      // Fallback: Non-streaming POST
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: q,
            top_k: topK,
            top_n: topN,
            augment_mode: augmentMode,
          }),
        });
        const data = await res.json();
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tutorMsgId
              ? {
                  ...msg,
                  text: data.answer || "Could not retrieve answer.",
                  sources: data.sources || [],
                  chunks: data.chunks || [],
                  durationMs: data.duration_ms,
                  model: data.model,
                }
              : msg
          )
        );
      } catch (postErr: any) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tutorMsgId
              ? {
                  ...msg,
                  text: `Connection Error: Unable to communicate with the Study Assistant engine on port 8000. (${postErr.message})`,
                }
              : msg
          )
        );
      }
    } finally {
      setIsGenerating(false);
      fetchStatus();
    }
  };

  // Upload file
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/ingest/file", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadMessage({
          type: "success",
          text: `Successfully ingested '${file.name}' (${data.result?.chunks_created || "new"} chunks indexed)`,
        });
        fetchDocuments();
        fetchStatus();
      } else {
        setUploadMessage({
          type: "error",
          text: `Ingestion error: ${data.detail || "Upload failed"}`,
        });
      }
    } catch (err: any) {
      setUploadMessage({
        type: "error",
        text: `Ingestion failed: ${err.message}`,
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Ingest sample Kepler astronomy document
  const handleIngestSample = async () => {
    setIsUploading(true);
    setUploadMessage(null);
    try {
      const res = await fetch("/api/ingest/sample", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        setUploadMessage({
          type: "success",
          text: "Kepler's Planetary Laws sample material indexed successfully.",
        });
        fetchDocuments();
        fetchStatus();
      } else {
        setUploadMessage({
          type: "error",
          text: `Error: ${data.detail || "Could not load sample"}`,
        });
      }
    } catch (err: any) {
      setUploadMessage({
        type: "error",
        text: `Failed to ingest sample: ${err.message}`,
      });
    } finally {
      setIsUploading(false);
    }
  };

  // Clear database
  const handleClearDatabase = async () => {
    if (!confirm("Are you sure you want to clear all indexed documents from ChromaDB?")) return;
    try {
      const res = await fetch("/api/clear", { method: "POST" });
      if (res.ok) {
        fetchDocuments();
        fetchStatus();
      }
    } catch (err: any) {
      alert(`Clear failed: ${err.message}`);
    }
  };

  // Generate flashcards
  const handleGenerateFlashcards = async () => {
    setIsGeneratingCards(true);
    try {
      const res = await fetch("/api/flashcards", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: flashcardTopic.trim() || undefined,
          count: 4,
        }),
      });
      const data = await res.json();
      if (res.ok && data.flashcards) {
        setFlashcards(data.flashcards);
        setFlippedCards({});
      } else {
        alert(data.detail || "Unable to generate flashcards. Please ingest documents first.");
      }
    } catch (err: any) {
      alert(`Failed to generate flashcards: ${err.message}`);
    } finally {
      setIsGeneratingCards(false);
    }
  };

  const toggleCardFlip = (id: string) => {
    setFlippedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const scoreCard = (id: string, score: "review" | "mastered", e: React.MouseEvent) => {
    e.stopPropagation();
    setCardScores((prev) => ({ ...prev, [id]: score }));
  };

  return (
    <div className="app-shell">
      {/* Sidebar Navigation */}
      <aside className="app-sidebar">
        <div className="sidebar-header">
          <div className="brand-badge">
            <IconLogo />
          </div>
          <div className="brand-info">
            <h1>Study Assistant</h1>
            <span>Academic RAG Tutor</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
            id="nav-chat"
          >
            <span className="nav-icon">
              <IconChat />
            </span>
            <span className="nav-label">Academic Tutor</span>
          </button>

          <button
            className={`nav-item ${activeTab === "materials" ? "active" : ""}`}
            onClick={() => setActiveTab("materials")}
            id="nav-materials"
          >
            <span className="nav-icon">
              <IconMaterials />
            </span>
            <span className="nav-label">Course Materials</span>
          </button>

          <button
            className={`nav-item ${activeTab === "flashcards" ? "active" : ""}`}
            onClick={() => setActiveTab("flashcards")}
            id="nav-flashcards"
          >
            <span className="nav-icon">
              <IconFlashcards />
            </span>
            <span className="nav-label">Study Flashcards</span>
          </button>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="app-main">
        {/* Top Header */}
        <header className="top-header">
          <div className="header-left">
            <h2 className="header-title">
              {activeTab === "chat" && "Academic Tutor"}
              {activeTab === "materials" && "Course Materials"}
              {activeTab === "flashcards" && "Study Flashcards"}
            </h2>
            <span className="header-subtitle">
              {activeTab === "chat" && "Grounded Page-Level Citations"}
              {activeTab === "materials" && "PDF, Markdown and Text Indexer"}
              {activeTab === "flashcards" && "Exam Preparation and Review Cards"}
            </span>
          </div>

          <div className="header-actions">
            {/* Dark / Light Mode Switcher */}
            <button
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title={`Switch to ${theme === "dark" ? "Light" : "Dark"} mode`}
              id="theme-toggle"
            >
              {theme === "dark" ? <IconSun /> : <IconMoon />}
              <span>{theme === "dark" ? "Light Mode" : "Dark Mode"}</span>
            </button>

            <div className="pill-badge">
              <span>NVIDIA NIM / Nebius</span>
            </div>
          </div>
        </header>

        {/* --- VIEW 1: TUTOR CHAT --- */}
        {activeTab === "chat" && (
          <div className="view-container">
            <div className="chat-wrapper">
              <div className="chat-history">
                {messages.map((msg) => (
                  <div key={msg.id} className="message-card">
                    <div className={`avatar-icon ${msg.sender === "user" ? "user-avatar" : "tutor-avatar"}`}>
                      {msg.sender === "user" ? <IconUser /> : <IconTutor />}
                    </div>
                    <div className={`message-bubble ${msg.sender === "user" ? "user-bubble" : ""}`}>
                      <div className="message-meta">
                        <span className="author-name">{msg.sender === "user" ? "Student" : "Study Tutor"}</span>
                        <span>{msg.timestamp}</span>
                      </div>

                      <div className="message-content">{msg.text}</div>

                      {/* Cited Sources List */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="citations-footer">
                          <span className="citations-label">Sources:</span>
                          {msg.sources.map((src, i) => (
                            <button
                              key={i}
                              className="citation-pill"
                              onClick={() => {
                                const chunkMatch = msg.chunks?.find((c) =>
                                  src.includes(c.source || c.metadata?.source || "")
                                );
                                setSelectedCitation(chunkMatch || { source: src, preview: "Cited in generated response." });
                              }}
                            >
                              <IconBook />
                              <span>{src}</span>
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Latency & Model Footer */}
                      {msg.durationMs !== undefined && (
                        <div style={{ marginTop: "10px", fontSize: "11.5px", color: "var(--text-muted)" }}>
                          Generated in {msg.durationMs}ms via {msg.model || status.provider}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="message-card">
                    <div className="avatar-icon tutor-avatar">
                      <IconTutor />
                    </div>
                    <div className="message-bubble">
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-secondary)" }}>
                        <span className="dot-pulse" />
                        <span>Searching materials and generating grounded answer...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Controls */}
              <div className="chat-controls-bar">
                <div className="input-options-row">
                  <div className="strategy-group">
                    <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Augmentation:</span>
                    <button
                      className={`strategy-btn ${augmentMode === "expand" ? "active" : ""}`}
                      onClick={() => setAugmentMode("expand")}
                      title="Multi-Query: Generates variations to capture diverse technical terminology"
                    >
                      Multi-Query
                    </button>
                    <button
                      className={`strategy-btn ${augmentMode === "rewrite" ? "active" : ""}`}
                      onClick={() => setAugmentMode("rewrite")}
                      title="Academic Rewrite: Converts informal queries into formal academic terminology"
                    >
                      Rewrite
                    </button>
                    <button
                      className={`strategy-btn ${augmentMode === "hyde" ? "active" : ""}`}
                      onClick={() => setAugmentMode("hyde")}
                      title="Deep Search: Passage-level semantic matching for complex questions"
                    >
                      Deep Search
                    </button>
                  </div>

                  <div style={{ fontSize: "11.5px", color: "var(--text-muted)" }}>
                    Press Enter to send
                  </div>
                </div>

                <div className="input-field-row">
                  <textarea
                    className="chat-textarea"
                    placeholder="Ask a question grounded in your course materials..."
                    value={inputQuery}
                    onChange={(e) => setInputQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleAsk();
                      }
                    }}
                    rows={2}
                    disabled={isGenerating}
                    id="chat-input"
                  />
                  <button
                    className="send-btn"
                    onClick={() => handleAsk()}
                    disabled={isGenerating || !inputQuery.trim()}
                    id="send-button"
                  >
                    <span>Ask</span>
                    <IconArrowRight />
                  </button>
                </div>

                {/* Prompt Starters */}
                <div className="prompt-starters">
                  <span style={{ fontSize: "11.5px", color: "var(--text-muted)", alignSelf: "center" }}>Quick Starters:</span>
                  <button
                    className="starter-pill"
                    onClick={() => handleAsk("What is Kepler's First Law and how does eccentricity affect it?")}
                  >
                    Kepler's First Law
                  </button>
                  <button
                    className="starter-pill"
                    onClick={() => handleAsk("Explain the relationship between orbital period and semi-major axis in Kepler's Third Law")}
                  >
                    Third Law Formula
                  </button>
                  <button
                    className="starter-pill"
                    onClick={() => handleAsk("What is the difference between perihelion and aphelion speed?")}
                  >
                    Perihelion vs Aphelion
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* --- VIEW 2: COURSE MATERIALS --- */}
        {activeTab === "materials" && (
          <div className="view-container">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
              <div>
                <h3 style={{ fontSize: "18px", fontWeight: "700" }}>Course Materials</h3>
                <p style={{ color: "var(--text-muted)", fontSize: "13.5px" }}>
                  Upload academic documents (.pdf, .txt, .md). Every chunk is indexed in ChromaDB with page tracking.
                </p>
              </div>
              <div style={{ display: "flex", gap: "8px" }}>
                <button className="btn-secondary" onClick={handleIngestSample} disabled={isUploading} id="load-sample-btn">
                  Load Kepler Sample
                </button>
                <button className="btn-danger" onClick={handleClearDatabase}>
                  <IconTrash />
                  <span>Clear Index</span>
                </button>
              </div>
            </div>

            {uploadMessage && (
              <div
                style={{
                  padding: "12px 16px",
                  borderRadius: "8px",
                  marginBottom: "16px",
                  background: uploadMessage.type === "success" ? "var(--accent-success-subtle)" : "var(--accent-danger-subtle)",
                  border: `1px solid ${uploadMessage.type === "success" ? "var(--accent-success-border)" : "var(--accent-danger-border)"}`,
                  color: uploadMessage.type === "success" ? "var(--accent-success)" : "var(--accent-danger)",
                  fontSize: "13px",
                  fontWeight: 500,
                }}
              >
                {uploadMessage.text}
              </div>
            )}

            {/* Drag & Drop Upload Zone */}
            <div
              className="dropzone-card"
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
              <div className="dropzone-icon">
                <IconUpload />
              </div>
              <div className="dropzone-title">
                {isUploading ? "Processing document chunks..." : "Click or drag course files here"}
              </div>
              <div className="dropzone-sub">
                Supported: PDF lecture slides, TXT transcripts, and Markdown notes
              </div>
            </div>

            {/* Document Table */}
            <div className="doc-table-card">
              <div className="table-header">
                <span className="table-title">Indexed Documents ({documents.length})</span>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                  Verified course library
                </span>
              </div>

              {documents.length === 0 ? (
                <div style={{ padding: "36px", textAlign: "center", color: "var(--text-muted)", fontSize: "13.5px" }}>
                  No materials indexed yet. Upload a file or click &quot;Load Kepler Sample&quot; to begin.
                </div>
              ) : (
                documents.map((doc, idx) => (
                  <div key={idx} className="doc-item">
                    <div className="doc-name">
                      <IconFile />
                      <div>
                        <div>{doc.source}</div>
                        <div style={{ fontSize: "11.5px", color: "var(--text-muted)" }}>{doc.file_path}</div>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                      <span className="doc-badge-pill">{doc.page_count} page(s)</span>
                      <span className="doc-badge-pill" style={{ color: "var(--accent-success)" }}>
                        Indexed
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* --- VIEW 3: STUDY FLASHCARDS --- */}
        {activeTab === "flashcards" && (
          <div className="view-container">
            <div className="flashcards-header">
              <div>
                <h3 style={{ fontSize: "18px", fontWeight: "700" }}>Study Flashcards</h3>
                <p style={{ color: "var(--text-muted)", fontSize: "13.5px" }}>
                  Generate exam-ready question-and-answer study cards derived strictly from course materials.
                </p>
              </div>

              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <input
                  type="text"
                  placeholder="Focus topic (optional)..."
                  value={flashcardTopic}
                  onChange={(e) => setFlashcardTopic(e.target.value)}
                  style={{
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "6px",
                    padding: "7px 10px",
                    color: "var(--text-primary)",
                    fontSize: "12.5px",
                    width: "200px",
                  }}
                />
                <button
                  className="btn-primary"
                  onClick={handleGenerateFlashcards}
                  disabled={isGeneratingCards}
                  id="generate-cards-btn"
                >
                  {isGeneratingCards ? "Generating..." : "Generate Cards"}
                </button>
              </div>
            </div>

            {flashcards.length === 0 ? (
              <div
                style={{
                  marginTop: "32px",
                  padding: "48px 20px",
                  textAlign: "center",
                  background: "var(--bg-card)",
                  borderRadius: "12px",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div style={{ marginBottom: "12px", display: "flex", justifyContent: "center", color: "var(--text-muted)" }}>
                  <IconFlashcards />
                </div>
                <h4 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "6px" }}>No Flashcards Generated Yet</h4>
                <p style={{ color: "var(--text-muted)", fontSize: "13px", maxWidth: "420px", margin: "0 auto" }}>
                  Click &quot;Generate Cards&quot; to transform your indexed study materials into study review cards.
                </p>
              </div>
            ) : (
              <div className="cards-grid" style={{ marginTop: "20px" }}>
                {flashcards.map((card) => {
                  const isFlipped = flippedCards[card.id];
                  const score = cardScores[card.id];
                  return (
                    <div
                      key={card.id}
                      className={`flip-card ${isFlipped ? "flipped" : ""}`}
                      onClick={() => toggleCardFlip(card.id)}
                    >
                      <div className="flip-card-inner">
                        {/* Front: Question */}
                        <div className="flip-card-front">
                          <div className="card-top">
                            <span className="card-category">{card.category}</span>
                            <span className="card-difficulty">{card.difficulty}</span>
                          </div>
                          <div className="card-question">{card.question}</div>
                          <div className="card-hint">
                            <IconFlip />
                            <span>Click card to reveal answer</span>
                            {score && (
                              <span style={{ marginLeft: "auto", fontWeight: "600", color: score === "mastered" ? "var(--accent-success)" : "var(--accent-warning)" }}>
                                {score === "mastered" ? "Mastered" : "Needs Review"}
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Back: Grounded Answer */}
                        <div className="flip-card-back">
                          <div className="card-top">
                            <span className="card-category" style={{ color: "var(--accent-primary)" }}>
                              Explanation & Formula
                            </span>
                            <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>{card.source}</span>
                          </div>
                          <div className="card-answer">{card.answer}</div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <button
                              className="btn-secondary"
                              style={{ padding: "4px 8px", fontSize: "11px" }}
                              onClick={(e) => scoreCard(card.id, "review", e)}
                            >
                              Needs Review
                            </button>
                            <button
                              className="btn-primary"
                              style={{ padding: "4px 10px", fontSize: "11px" }}
                              onClick={(e) => scoreCard(card.id, "mastered", e)}
                            >
                              Mastered
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Citation Inspector Modal */}
      {selectedCitation && (
        <div className="modal-overlay" onClick={() => setSelectedCitation(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <IconBook />
                <h4 style={{ fontSize: "15px", fontWeight: "700" }}>
                  Verified Source Excerpt
                </h4>
              </div>
              <button
                style={{ background: "none", border: "none", color: "var(--text-muted)", fontSize: "16px", cursor: "pointer" }}
                onClick={() => setSelectedCitation(null)}
              >
                ✕
              </button>
            </div>

            <div style={{ marginBottom: "12px", fontSize: "12.5px", color: "var(--accent-primary)" }}>
              Document: {selectedCitation.source || selectedCitation.metadata?.source || "Unknown"}
              {selectedCitation.page && ` · Page ${selectedCitation.page}`}
            </div>

            <div
              style={{
                background: "var(--bg-primary)",
                padding: "14px",
                borderRadius: "8px",
                border: "1px solid var(--border-subtle)",
                fontSize: "13px",
                lineHeight: "1.6",
                color: "var(--text-secondary)",
                maxHeight: "240px",
                overflowY: "auto",
              }}
            >
              {selectedCitation.preview || selectedCitation.text || "No preview excerpt available."}
            </div>

            {selectedCitation.score !== undefined && (
              <div style={{ marginTop: "10px", fontSize: "11.5px", color: "var(--text-muted)" }}>
                Relevance Confidence: {(selectedCitation.score * 100).toFixed(1)}%
              </div>
            )}

            <div style={{ marginTop: "18px", display: "flex", justifyContent: "flex-end" }}>
              <button className="btn-secondary" onClick={() => setSelectedCitation(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
