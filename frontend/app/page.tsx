"use client";

import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import katex from "katex";
import "katex/dist/katex.min.css";

function cleanHeadingText(children: React.ReactNode): React.ReactNode {
  if (typeof children === "string") {
    return children.replace(/^#+\s*/, "");
  }
  if (Array.isArray(children)) {
    return children.map((child, idx) => {
      if (idx === 0 && typeof child === "string") {
        return child.replace(/^#+\s*/, "");
      }
      return child;
    });
  }
  return children;
}

function ensureMathSpacing(s: string): string {
  if (!s || !s.includes("$")) return s;
  const parts = s.split(/(\$\$[\s\S]*?\$\$|\$(?!\$)[^\$\n]+?(?<!\$)\$)/g);
  const out: string[] = [];
  for (let i = 0; i < parts.length; i++) {
    let part = parts[i];
    if (part.startsWith("$")) {
      if (!part.startsWith("$$")) {
        // Prepend space if previous text ended with an alphanumeric character
        if (out.length > 0 && out[out.length - 1] && /[a-zA-Z0-9]$/.test(out[out.length - 1])) {
          out[out.length - 1] += " ";
        }
      }
      out.push(part);
    } else {
      // Append space if previous block was inline math and this text starts with an alphanumeric character
      if (i > 0 && parts[i - 1].startsWith("$") && !parts[i - 1].startsWith("$$")) {
        if (part && /^[a-zA-Z0-9]/.test(part)) {
          part = " " + part;
        }
      }
      out.push(part);
    }
  }
  return out.join("");
}

function prepareTutorMarkdown(text: string): string {
  if (!text) return "";
  let cleaned = text.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\uFE0F]/gu, "");

  // Convert raw Unicode math symbols into standard LaTeX
  const unicodeMathMap: { [k: string]: string } = {
    "∫": "\\int ",
    "∬": "\\iint ",
    "∭": "\\iiint ",
    "∮": "\\oint ",
    "∑": "\\sum ",
    "∏": "\\prod ",
    "√": "\\sqrt",
    "∂": "\\partial ",
    "∞": "\\infty ",
    "≈": "\\approx ",
    "≠": "\\ne ",
    "≤": "\\le ",
    "≥": "\\ge ",
    "→": "\\to ",
    "±": "\\pm ",
    "·": "\\cdot ",
    "×": "\\times ",
    "÷": "\\div ",
  };
  for (const [uni, lat] of Object.entries(unicodeMathMap)) {
    cleaned = cleaned.replaceAll(uni, lat);
  }

  // 1. Unescape escaped hashes: '\#\#' -> '##'
  cleaned = cleaned.replace(/\\+(#{1,6})/g, "$1");

  // 2. Normalize non-breaking spaces or zero-width spaces
  cleaned = cleaned.replace(/[\u00A0\u2000-\u200B]/g, " ");

  // 3. Collapse duplicate hashes: '## ## Title' -> '## Title'
  cleaned = cleaned.replace(/(^|\n)(#{1,6})\s+(#{1,6})\s+/g, "$1$2 ");

  // 4. Ensure space after hashes: '##Title' -> '## Title'
  cleaned = cleaned.replace(/(^|\n)(#{1,6})([^\s#\n])/g, "$1$2 $3");

  // 5. Ensure headings glued to previous sentences get their own line with a blank line before
  cleaned = cleaned.replace(/([^\n])\s+(#{1,6}\s+[^\n]+)/g, "$1\n\n$2");

  // 6. Pre-clean duplicate/empty display math markers: '$$ \n\n $$' -> '$$'
  cleaned = cleaned.replace(/\$\$(?:\s*\$\$)+/g, "$$");
  cleaned = cleaned.replace(/\$\$\s*\$\$/g, "");

  // Convert \[ ... \] to display $$ ... $$ and \( ... \) to inline $ ... $
  cleaned = cleaned.replace(/\\\[([\s\S]*?)\\\]/g, "\n$$\n$1\n$$\n");
  cleaned = cleaned.replace(/\\\(([\s\S]*?)\\\)/g, (_, inner) => `$${inner.trim()}$`);

  // Sanitize dangling sizing tags with no delimiters (e.g. \bigl with no bracket)
  cleaned = cleaned.replace(/\\(left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)\s*$/gm, "");
  cleaned = cleaned.replace(/\\(left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)(?=\s*($|[^()[\]{}|.\\/]))/g, "");

  // 7. Extract citations from math mode ($$ ... $$ and $ ... $)
  // Matches \quad\text{[3]}, \text{[3]}, \tag{[3]}, [3], etc. with optional trailing punctuation
  const mathCiteRe = /(?:\\[q]?quad|\\hspace\{[^}]*\}|\s*)*(?:\\text|\\tag)?\s*\{?\s*(\[(?:Source|Web Source):[^\]]+?\]|\[\d{1,2}\])\s*\}?\s*([.,;:!?])?/gi;

  cleaned = cleaned.replace(/\$\$([\s\S]*?)\$\$/g, (match, body) => {
    const cites: string[] = [];
    let punct = "";
    const cleanBody = body.replace(mathCiteRe, (_: string, cite: string, p?: string) => {
      if (cite) cites.push(cite.trim());
      if (p) punct = p;
      return "";
    }).trim();
    if (cites.length === 0) return match;
    const citeStr = cites.join(" ") + punct;
    return `\n$$\n${cleanBody}\n$$\n${citeStr}\n`;
  });

  cleaned = cleaned.replace(/(?<!\$)\$(?!\$)([\s\S]*?)(?<!\$)\$(?!\$)/g, (match, body) => {
    const cites: string[] = [];
    let punct = "";
    const cleanBody = body.replace(mathCiteRe, (_: string, cite: string, p?: string) => {
      if (cite) cites.push(cite.trim());
      if (p) punct = p;
      return "";
    }).trim();
    if (cites.length === 0) return match;
    const citeStr = cites.join(" ") + punct;
    return `$${cleanBody}$ ${citeStr}`;
  });

  // 8. Balance odd $$ count so unclosed $$ doesn't swallow markdown & citations into KaTeX
  const displayDollarCount = (cleaned.match(/\$\$/g) || []).length;
  if (displayDollarCount % 2 !== 0) {
    cleaned += "\n$$";
  }

  // 9. Turn setext headings (title \n === / ---) into ATX headings
  const lines = cleaned.split("\n");
  const output: string[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    const title = lines[i].trim();
    const next = (lines[i + 1] || "").trim();
    if (title && !/^[#\-*|>]/.test(title) && /^=+$/.test(next)) {
      output.push(`\n## ${title}\n`);
      i += 1;
      continue;
    }
    if (title && !/^[#\-*|>]/.test(title) && next.length >= 3 && /^-+$/.test(next)) {
      output.push(`\n### ${title}\n`);
      i += 1;
      continue;
    }
    output.push(lines[i]);
  }
  return ensureMathSpacing(output.join("\n"));
}


function linkCitations(text: string): string {
  if (!text) return "";
  // Split by $$ ... $$, \[ ... \], \( ... \), and $ ... $ so citations are ONLY linked OUTSIDE of math formulas.
  // This guarantees #ref anchors never contaminate KaTeX math mode.
  return text
    .split(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$(?:\\.|[^$\n])+\$)/g)
    .map((chunk) => {
      if (chunk.startsWith("$") || chunk.startsWith("\\(") || chunk.startsWith("\\[")) {
        return chunk;
      }
      return chunk.replace(/\[(\d{1,2})\]/g, "[$1](#ref-$1)");
    })
    .join("");
}


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

interface BoundaryCheck {
  input: string;
  lhs: string | number;
  rhs: string | number;
  match: boolean;
}

interface VerificationPayload {
  has_math: boolean;
  status: "verified" | "discrepancy" | "inconclusive";
  method?: string;
  summary: string;
  verified_formula?: string;
  boundary_checks?: BoundaryCheck[];
  wolfram_url?: string;
  python_code?: string;
}

interface Message {
  id: string;
  sender: "user" | "tutor";
  text: string;
  sources?: string[];
  chunks?: any[];
  durationMs?: number;
  model?: string;
  provider?: string;
  grounding?: "course" | "web" | "general" | "notes" | "both" | "";
  references?: ReferenceItem[];
  verification?: VerificationPayload | null;
  timestamp: string;
  interrupted?: boolean;
}

interface ReferenceItem {
  n: number;
  label: string;
  source: string;
  page?: number | string;
  source_type?: string;
  file_path?: string;
  url?: string;
  preview?: string;
  image_url?: string;
}

interface DocItem {
  source: string;
  file_path: string;
  chunk_count: number;
  page_count: number;
  has_visuals?: boolean;
  visual_count?: number;
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
  tavily_configured?: boolean;
  tavily_available?: boolean;
  indexed_chunks: number;
  persist_directory: string;
  generation_model: string;
  augment_model?: string;
  rerank_model?: string;
  embedding_model?: string;
}

function MathFormula({
  formula,
  inline = true,
  className = "",
}: {
  formula: string;
  inline?: boolean;
  className?: string;
}) {
  if (!formula) return null;
  let clean = formula.trim();
  if (clean.startsWith("$$") && clean.endsWith("$$")) {
    clean = clean.slice(2, -2).trim();
  } else if (clean.startsWith("$") && clean.endsWith("$")) {
    clean = clean.slice(1, -1).trim();
  } else if (clean.startsWith("\\[") && clean.endsWith("\\]")) {
    clean = clean.slice(2, -2).trim();
  } else if (clean.startsWith("\\(") && clean.endsWith("\\)")) {
    clean = clean.slice(2, -2).trim();
  }

  try {
    const html = katex.renderToString(clean, {
      displayMode: !inline,
      throwOnError: false,
      strict: "ignore",
    });
    return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
  } catch {
    return <span className={className}>{formula}</span>;
  }
}

function VerificationCard({ verification }: { verification?: VerificationPayload | null }) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showCode, setShowCode] = useState(false);

  // Strictly guard: only render the card when symbolic check is verified!
  // Never show noisy, broken, or unverified "inconclusive" cards.
  if (!verification || !verification.has_math || verification.status !== "verified") return null;

  const handleCopy = () => {
    if (verification.python_code) {
      navigator.clipboard.writeText(verification.python_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="verification-card status-verified">
      <div className="verification-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="verification-badge-group">
          <span className="verification-badge badge-verified">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Symbolically Verified (CAS)
          </span>
          {verification.verified_formula && (
            <span className="verification-formula-preview" title="Verified Symbolic Identity">
              <MathFormula formula={verification.verified_formula} inline={true} />
            </span>
          )}
        </div>
        <button
          type="button"
          className="verification-toggle-btn"
          aria-expanded={isOpen}
          onClick={(e) => {
            e.stopPropagation();
            setIsOpen(!isOpen);
          }}
        >
          {isOpen ? "Hide Details" : "Inspect Proof"}
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ transform: isOpen ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
      </div>

      {isOpen && (
        <div className="verification-body">
          <div className="verification-summary-text">
            <strong>Method:</strong> {verification.summary}
          </div>

          {verification.boundary_checks && verification.boundary_checks.length > 0 && (
            <div className="verification-boundary-section">
              <span className="verification-section-title">Boundary Input Sanity Checks</span>
              <div className="verification-boundary-grid">
                {verification.boundary_checks.map((check, idx) => (
                  <div key={idx} className={`boundary-chip ${check.match ? "match" : "diverge"}`}>
                    <span className="boundary-input">{check.input}</span>
                    <span className="boundary-eval">
                      <MathFormula formula={`${check.lhs} = ${check.rhs}`} inline={true} />
                    </span>
                    <span className="boundary-status">{check.match ? "✓ Match" : "✗ Divergence"}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="verification-actions">
            {verification.wolfram_url && (
              <a
                href={verification.wolfram_url}
                target="_blank"
                rel="noopener noreferrer"
                className="verification-action-btn wolfram-btn"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                  <polyline points="15 3 21 3 21 9" />
                  <line x1="10" y1="14" x2="21" y2="3" />
                </svg>
                Verify on Wolfram Alpha
              </a>
            )}

            {verification.python_code && (
              <>
                <button
                  type="button"
                  className="verification-action-btn python-copy-btn"
                  onClick={handleCopy}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  {copied ? "Copied Python Script!" : "Copy SymPy Script"}
                </button>

                <button
                  type="button"
                  className="verification-action-btn code-toggle-btn"
                  onClick={() => setShowCode(!showCode)}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  {showCode ? "Hide Code" : "View Code"}
                </button>
              </>
            )}
          </div>

          {showCode && verification.python_code && (
            <pre className="verification-code-box">
              <code>{verification.python_code}</code>
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function StudyAssistantApp() {
  // Navigation tabs (RAG pipeline tab removed per requirement)
  const [activeTab, setActiveTab] = useState<"chat" | "materials" | "flashcards">("chat");

  // Theme state: dark / light
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Status state
  const [status, setStatus] = useState<SystemStatus>({
    status: "connecting",
    provider: "",
    nvidia_configured: false,
    nebius_configured: false,
    tavily_configured: false,
    tavily_available: false,
    indexed_chunks: 0,
    persist_directory: "./chroma_db",
    generation_model: "",
  });

  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "tutor",
      text: "Ask a question about your notes, or turn on Tavily Web for academic pages. Upload a lecture PDF in Course Materials when you want page citations from your own syllabus.",
      sources: [],
      timestamp: "Just now",
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [augmentMode, setAugmentMode] = useState<"expand" | "rewrite" | "hyde">("expand");
  const [sourceMode, setSourceMode] = useState<"notes" | "web" | "both">("notes");
  const [searchMenuOpen, setSearchMenuOpen] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<any | null>(null);

  // Retrieval configuration
  const topK = 15;
  const topN = 5;

  // Materials state
  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Flashcards state
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [flippedCards, setFlippedCards] = useState<{ [id: string]: boolean }>({});
  const [cardScores, setCardScores] = useState<{ [id: string]: "review" | "mastered" }>({});
  const [isGeneratingCards, setIsGeneratingCards] = useState(false);
  const [flashcardTopic, setFlashcardTopic] = useState("");

  const chatEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Stop / interrupt generation mid-stream
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
  };

  // Keyboard shortcut: Escape cancels generation anywhere in the window
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isGenerating) {
        e.preventDefault();
        handleStop();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isGenerating]);

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

    const interval = setInterval(() => {
      fetchStatus();
      fetchDocuments();
    }, 4000);

    const onFocus = () => {
      fetchStatus();
      fetchDocuments();
    };
    window.addEventListener("focus", onFocus);

    return () => {
      clearInterval(interval);
      window.removeEventListener("focus", onFocus);
    };
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
  const handleAsk = async (queryText?: string, options?: { sourceMode?: "notes" | "web" | "both" }) => {
    const q = queryText || inputQuery;
    if (!q.trim() || isGenerating) return;
    const mode = options?.sourceMode ?? sourceMode;
    if (options?.sourceMode) {
      setSourceMode(options.sourceMode);
    }

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
    const controller = new AbortController();
    abortControllerRef.current = controller;

    let streamedAnswer = "";
    let finalSources: string[] = [];
    let finalChunks: any[] = [];
    let finalModel: string | undefined;
    let finalProvider: string | undefined;
    let finalGrounding: Message["grounding"] = "";
    let finalReferences: ReferenceItem[] = [];
    let finalVerification: VerificationPayload | null = null;

    try {
      const url = `/api/ask/stream?query=${encodeURIComponent(q)}&top_k=${topK}&top_n=${topN}&augment_mode=${augmentMode}&source_mode=${mode}&use_web=false&fallback_to_web=false`;
      const response = await fetch(url, { signal: controller.signal });

      if (!response.ok || !response.body) {
        throw new Error("Streaming connection failed.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      while (true) {
        if (controller.signal.aborted) break;
        const { value, done } = await reader.read();
        if (done || controller.signal.aborted) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

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
                finalModel = event.model;
                finalProvider = event.provider;
                if (event.answer) {
                  streamedAnswer = event.answer;
                }
                finalGrounding = event.grounding || "";
                finalReferences = event.references || [];
                finalVerification = event.verification || null;
              }
            } catch {
              // Ignore partial frames
            }
          }
        }
      }

      if (controller.signal.aborted) {
        return;
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
                model: finalModel,
                provider: finalProvider,
                grounding: finalGrounding,
                references: finalReferences,
                verification: finalVerification,
              }
            : msg
        )
      );
    } catch (err: any) {
      // Graceful cancellation handling
      if (controller.signal.aborted || err?.name === "AbortError") {
        const durationMs = Date.now() - startTime;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tutorMsgId
              ? {
                  ...msg,
                  text: streamedAnswer.trim() || "Response stopped.",
                  sources: finalSources,
                  chunks: finalChunks,
                  durationMs,
                  interrupted: true,
                }
              : msg
          )
        );
        return;
      }

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
            source_mode: mode,
            use_web: false,
            fallback_to_web: false,
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
                  provider: data.provider_label,
                  grounding: data.grounding,
                  verification: data.verification || null,
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
      abortControllerRef.current = null;
      setIsGenerating(false);
      fetchStatus();
    }
  };

  // Upload file
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadMessage({
      type: "info",
      text: `Uploading '${file.name}' to server...`,
    });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/ingest/file?background=true", {
        method: "POST",
        body: formData,
      });

      const contentType = res.headers.get("content-type") || "";
      let data: any = {};
      if (contentType.includes("application/json")) {
        data = await res.json();
      } else {
        const rawText = await res.text();
        throw new Error(
          rawText && !rawText.startsWith("<")
            ? rawText
            : `Upload failed (${res.status}). The file may be too large for the proxy, or the study server is still indexing another book.`
        );
      }

      if (!res.ok) {
        throw new Error(data.detail || `Upload failed with status ${res.status}`);
      }

      if (data.job_id) {
        let isDone = false;
        let pollCount = 0;
        const maxPolls = 600; // 10 minutes timeout at 1s intervals

        while (!isDone && pollCount < maxPolls) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          pollCount++;

          try {
            const statusRes = await fetch(`/api/ingest/status/${data.job_id}`);
            const statusContentType = statusRes.headers.get("content-type") || "";
            if (statusRes.ok && statusContentType.includes("application/json")) {
              const job = await statusRes.json();
              if (job.status === "processing") {
                const progressInfo = job.total_chunks > 0
                  ? `${job.progress}% (${job.processed_chunks}/${job.total_chunks} chunks)`
                  : "extracting pages...";
                setUploadMessage({
                  type: "info",
                  text: `Ingesting '${file.name}' (${progressInfo})...`,
                });
              } else if (job.status === "completed") {
                isDone = true;
                const count = job.result?.chunks_count ?? job.total_chunks ?? "all";
                setUploadMessage({
                  type: "success",
                  text: `Successfully ingested '${file.name}' (${count} chunks indexed)`,
                });
                fetchDocuments();
                fetchStatus();
                break;
              } else if (job.status === "failed") {
                isDone = true;
                throw new Error(job.error || "Background ingestion failed.");
              }
            }
          } catch (pollErr: any) {
            if (pollErr.message && !pollErr.message.includes("fetch")) {
              throw pollErr;
            }
          }
        }
      } else {
        const count = data.result?.chunks_count ?? data.total_indexed_chunks ?? "all";
        setUploadMessage({
          type: "success",
          text: `Successfully ingested '${file.name}' (${count} chunks indexed)`,
        });
        fetchDocuments();
        fetchStatus();
      }
    } catch (err: any) {
      setUploadMessage({
        type: "error",
        text: `Ingestion error: ${err.message}`,
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Ingest sample Kepler astronomy document
  const handleIngestSample = async () => {
    setIsUploading(true);
    setUploadMessage({
      type: "info",
      text: "Ingesting Kepler's Planetary Laws sample material...",
    });
    try {
      const res = await fetch("/api/ingest/sample", { method: "POST" });
      const contentType = res.headers.get("content-type") || "";
      let data: any = {};
      if (contentType.includes("application/json")) {
        data = await res.json();
      } else {
        const text = await res.text();
        throw new Error(text || `Server returned status ${res.status}`);
      }

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
        text: `Sample ingestion failed: ${err.message}`,
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

  const isPptReference = (ref?: ReferenceItem | null) => {
    if (!ref) return false;
    const target = (ref.file_path || ref.source || "").toLowerCase();
    return target.endsWith(".pptx") || target.endsWith(".ppt");
  };

  const scrollToBottomReference = (msgId: string, refNum: number) => {
    const el = document.getElementById(`ref-${msgId}-${refNum}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("highlight-ref");
      setTimeout(() => {
        el.classList.remove("highlight-ref");
      }, 2000);
      return true;
    }
    return false;
  };

  const openReference = (ref: ReferenceItem, msgId?: string) => {
    if (isPptReference(ref)) {
      if (msgId && scrollToBottomReference(msgId, ref.n)) {
        return;
      }
      if (ref.preview) {
        setSelectedCitation({
          source: ref.source,
          page: ref.page,
          preview: ref.preview,
          source_type: ref.source_type,
        });
      }
      return;
    }

    if (ref.url) {
      window.open(ref.url, "_blank", "noopener,noreferrer");
      return;
    }
    if (ref.file_path) {
      const page = ref.page || 1;
      window.open(
        `/api/materials/file?path=${encodeURIComponent(ref.file_path)}#page=${page}`,
        "_blank",
        "noopener,noreferrer"
      );
      return;
    }
    setSelectedCitation({
      source: ref.source,
      page: ref.page,
      preview: ref.preview,
      source_type: ref.source_type,
    });
  };

  const searchChoices: { id: "expand" | "rewrite" | "hyde"; label: string }[] = [
    { id: "expand", label: "Broader wording" },
    { id: "rewrite", label: "Rewrite the question" },
    { id: "hyde", label: "Match a passage" },
  ];

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
            <span>Notes and tutoring</span>
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
            <span className="nav-label">Tutor</span>
          </button>

          <button
            className={`nav-item ${activeTab === "materials" ? "active" : ""}`}
            onClick={() => setActiveTab("materials")}
            id="nav-materials"
          >
            <span className="nav-icon">
              <IconMaterials />
            </span>
            <span className="nav-label">Notes</span>
          </button>

          <button
            className={`nav-item ${activeTab === "flashcards" ? "active" : ""}`}
            onClick={() => setActiveTab("flashcards")}
            id="nav-flashcards"
          >
            <span className="nav-icon">
              <IconFlashcards />
            </span>
            <span className="nav-label">Recall</span>
          </button>
        </nav>

        <div className="source-panel">
          <span className="sidebar-kicker">In this session</span>
          {documents.length === 0 ? (
            <button className="source-empty" onClick={() => setActiveTab("materials")}>
              Add a PDF, or load the Kepler sample.
            </button>
          ) : (
            documents.slice(0, 8).map((doc) => (
              <button key={doc.source} className="source-row" onClick={() => setActiveTab("materials")}>
                <IconFile />
                <span>{doc.source}</span>
              </button>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <span>{documents.length === 0 ? "Page citations need notes" : `${documents.length} notes indexed`}</span>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="app-main">
        {/* Top Header */}
        <header className="top-header">
          <div className="header-left">
            <h2 className="header-title">
              {activeTab === "chat" && "Tutor"}
              {activeTab === "materials" && "Notes"}
              {activeTab === "flashcards" && "Recall"}
            </h2>
            {activeTab === "materials" && (
              <span className="header-subtitle">PDF, PowerPoint Slides, Markdown and Text Indexer</span>
            )}
            {activeTab === "flashcards" && (
              <span className="header-subtitle">Exam Preparation and Review Cards</span>
            )}
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
          </div>
        </header>

        {/* --- VIEW 1: TUTOR CHAT --- */}
        {activeTab === "chat" && (
          <div className="view-container">
            <div className="chat-wrapper">
              <div className={`chat-history ${messages.every((m) => m.id === "welcome") && !isGenerating ? "chat-history-empty" : ""}`}>
                {messages.every((m) => m.id === "welcome") && !isGenerating && (
                  <div className="empty-study">
                    <p className="empty-kicker">Study session</p>
                    <h2>What do you want to understand?</h2>
                    <p>
                      The tutor teaches from your notes and names the page. If a fact is not in those notes, it says so
                      instead of citing a random webpage.
                    </p>
                    <div className="empty-actions">
                      <button className="text-action" onClick={() => setActiveTab("materials")}>
                        Add notes
                      </button>
                      <button className="text-action" onClick={() => setActiveTab("flashcards")}>
                        Make recall cards
                      </button>
                    </div>
                    <div className="suggestion-list">
                      <button onClick={() => handleAsk("What is Kepler's First Law and how does eccentricity affect it?")}>
                        Explain Kepler&apos;s first law
                      </button>
                      <button onClick={() => handleAsk("Explain the relationship between orbital period and semi-major axis in Kepler's Third Law")}>
                        Walk through the third law
                      </button>
                      <button onClick={() => handleAsk("What is the difference between perihelion and aphelion speed?")}>
                        Contrast perihelion and aphelion
                      </button>
                      <button
                        onClick={() =>
                          handleAsk(
                            "What is the NASA Kepler space telescope and why was the mission retired?",
                            { sourceMode: "web" }
                          )
                        }
                      >
                        Look up the Kepler mission on the academic web
                      </button>
                    </div>
                  </div>
                )}
                {messages.filter((m) => m.id !== "welcome").map((msg) => (
                  <div key={msg.id} className={`message-card ${msg.sender === "user" ? "from-student" : "from-tutor"}`}>
                    <div className={`message-bubble ${msg.sender === "user" ? "user-bubble" : ""}`}>
                      <div className="message-meta">
                        <span className="author-name">{msg.sender === "user" ? "You" : "Tutor"}</span>
                      </div>

                      {msg.sender === "tutor" && msg.grounding && (
                        <div className={`grounding-banner grounding-${msg.grounding}`}>
                          {msg.grounding === "course" &&
                            "From your course materials. Each citation is a file and page in the index."}
                          {msg.grounding === "web" &&
                            "From academic web results via Tavily, not from your syllabus. Upload notes for page citations."}
                          {msg.grounding === "general" &&
                            "No matching course notes. This is general tutoring, not a page-cited answer. Upload a lecture PDF for syllabus citations."}
                        </div>
                      )}

                      {msg.sender === "tutor" ? (
                        <div className="message-content markdown-body">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
                            components={{
                              h1: ({ children, ...props }) => <h1 {...props}>{cleanHeadingText(children)}</h1>,
                              h2: ({ children, ...props }) => <h2 {...props}>{cleanHeadingText(children)}</h2>,
                              h3: ({ children, ...props }) => <h3 {...props}>{cleanHeadingText(children)}</h3>,
                              h4: ({ children, ...props }) => <h4 {...props}>{cleanHeadingText(children)}</h4>,
                              a: ({ href, children }) => {
                                if (href?.startsWith("#ref-")) {
                                  const refNum = Number(href.slice(5));
                                  const ref = msg.references?.find((item) => item.n === refNum);
                                  return (
                                    <button
                                      type="button"
                                      className="cite-num"
                                      onClick={() => {
                                        if (ref && isPptReference(ref)) {
                                          scrollToBottomReference(msg.id, ref.n);
                                        } else if (ref) {
                                          openReference(ref, msg.id);
                                        } else {
                                          scrollToBottomReference(msg.id, refNum);
                                        }
                                      }}
                                      title={ref ? `${ref.source} (p. ${ref.page || 1})` : `Reference [${refNum}]`}
                                    >
                                      [{href.slice(5)}]
                                    </button>
                                  );
                                }
                                return (
                                  <a href={href} target="_blank" rel="noreferrer">
                                    {children}
                                  </a>
                                );
                              },
                            }}
                          >
                            {linkCitations(prepareTutorMarkdown(msg.text))}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <div className="message-content">{msg.text}</div>
                      )}

                      {/* Interrupted notice */}
                      {msg.interrupted && (
                        <div className="interrupted-notice">
                          <span className="interrupted-dot" />
                          <span>Generation stopped</span>
                        </div>
                      )}

                      {/* Cited Sources List */}
                      {msg.references?.some((ref) => ref.image_url) && (
                        <div className="page-figures">
                          {msg.references
                            .filter((ref) => ref.image_url)
                            .map((ref) => (
                              <figure
                                key={`${ref.file_path}-${ref.page}`}
                                className="page-figure"
                                onClick={() => {
                                  if (isPptReference(ref)) {
                                    scrollToBottomReference(msg.id, ref.n);
                                  } else {
                                    openReference(ref, msg.id);
                                  }
                                }}
                                style={{ cursor: "pointer" }}
                                title={isPptReference(ref) ? "Referansa git" : "Görüntülemek için tıklayın"}
                              >
                                <img src={ref.image_url} alt={ref.label} />
                                <figcaption>{ref.label}</figcaption>
                              </figure>
                            ))}
                        </div>
                      )}

                      {msg.references && msg.references.length > 0 && (
                        <ol className="reference-list">
                          {msg.references.map((ref) => (
                            <li key={ref.n} id={`ref-${msg.id}-${ref.n}`} className="reference-item">
                              <button
                                type="button"
                                className="reference-link"
                                onClick={() => openReference(ref, msg.id)}
                              >
                                {ref.label}
                              </button>
                            </li>
                          ))}
                        </ol>
                      )}

                      {/* Symbolic CAS Math Verification */}
                      {msg.verification && <VerificationCard verification={msg.verification} />}

                      {/* Latency & Model Footer */}
                      {msg.durationMs !== undefined && msg.model && (
                        <div className="answer-footnote">
                          {msg.model}
                          {msg.provider ? ` · ${msg.provider}` : ""}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="message-card from-tutor">
                    <div className="message-bubble">
                      <div className="thinking-line">
                        <span className="dot-pulse" />
                        <span>Reading your notes</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input Controls */}
              <div className="composer">
                <textarea
                  className="chat-textarea"
                  placeholder={
                    sourceMode === "notes"
                      ? "Ask about your notes"
                      : sourceMode === "web"
                        ? "Ask the academic web"
                        : "Ask your notes and the academic web"
                  }
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape" && isGenerating) {
                      e.preventDefault();
                      handleStop();
                    } else if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleAsk();
                    }
                  }}
                  rows={2}
                  disabled={isGenerating}
                  id="chat-input"
                />
                <div className="composer-bar">
                  <div className="source-switch" role="group" aria-label="Where to search">
                    {(["notes", "web", "both"] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={sourceMode === mode ? "on" : ""}
                        onClick={() => setSourceMode(mode)}
                      >
                        {mode === "notes" ? "Notes" : mode === "web" ? "Web" : "Both"}
                      </button>
                    ))}
                  </div>
                  <div className="search-menu">
                    <button
                      type="button"
                      className="search-menu-button"
                      aria-expanded={searchMenuOpen}
                      onClick={() => setSearchMenuOpen((open) => !open)}
                    >
                      {searchChoices.find((choice) => choice.id === augmentMode)?.label}
                    </button>
                    {searchMenuOpen && (
                      <div className="search-menu-list" role="listbox">
                        {searchChoices.map((choice) => (
                          <button
                            key={choice.id}
                            type="button"
                            role="option"
                            aria-selected={augmentMode === choice.id}
                            className={augmentMode === choice.id ? "on" : ""}
                            onClick={() => {
                              setAugmentMode(choice.id);
                              setSearchMenuOpen(false);
                            }}
                          >
                            {choice.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  {isGenerating ? (
                    <button
                      className="stop-btn"
                      onClick={handleStop}
                      id="stop-button"
                      type="button"
                      title="Stop generating (Esc)"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="4" y="4" width="16" height="16" rx="2" />
                      </svg>
                      <span>Stop</span>
                    </button>
                  ) : (
                    <button
                      className="send-btn"
                      onClick={() => handleAsk()}
                      disabled={!inputQuery.trim()}
                      id="send-button"
                      type="button"
                    >
                      Ask
                    </button>
                  )}
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
                  background:
                    uploadMessage.type === "success"
                      ? "var(--accent-success-subtle)"
                      : uploadMessage.type === "info"
                      ? "rgba(99, 102, 241, 0.12)"
                      : "var(--accent-danger-subtle)",
                  border: `1px solid ${
                    uploadMessage.type === "success"
                      ? "var(--accent-success-border)"
                      : uploadMessage.type === "info"
                      ? "rgba(99, 102, 241, 0.35)"
                      : "var(--accent-danger-border)"
                  }`,
                  color:
                    uploadMessage.type === "success"
                      ? "var(--accent-success)"
                      : uploadMessage.type === "info"
                      ? "#6366f1"
                      : "var(--accent-danger)",
                  fontSize: "13px",
                  fontWeight: 500,
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                }}
              >
                {uploadMessage.type === "info" && (
                  <span
                    style={{
                      width: "8px",
                      height: "8px",
                      borderRadius: "50%",
                      backgroundColor: "#6366f1",
                      display: "inline-block",
                      animation: "pulse 1.5s infinite",
                    }}
                  />
                )}
                <span>{uploadMessage.text}</span>
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
                accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.pptx,.ppt"
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
                Supported: PDF & PowerPoint slides (.pptx, .ppt), diagrams (.png, .jpg), TXT transcripts, and Markdown notes
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
                      {doc.has_visuals && (
                        <span
                          style={{
                            fontSize: "11px",
                            fontWeight: 600,
                            padding: "2px 8px",
                            borderRadius: "4px",
                            background: "rgba(99, 102, 241, 0.15)",
                            color: "#6366f1",
                            border: "1px solid rgba(99, 102, 241, 0.3)",
                          }}
                        >
                          {doc.visual_count ? `${doc.visual_count} Diagram(s)` : "Diagrams"}
                        </span>
                      )}
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

            <div style={{ marginBottom: "12px", fontSize: "12.5px", color: "var(--accent-primary)", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "6px" }}>
              <span>
                Document: {selectedCitation.source || selectedCitation.metadata?.source || "Unknown"}
                {selectedCitation.page && ` · Page ${selectedCitation.page}`}
              </span>
              {(selectedCitation.source_type === "web" || String(selectedCitation.source || "").startsWith("http")) && (
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: "4px",
                    background: "rgba(16, 185, 129, 0.15)",
                    color: "#10b981",
                    border: "1px solid rgba(16, 185, 129, 0.3)",
                  }}
                >
                  Tavily Web
                </span>
              )}
              {(selectedCitation.metadata?.has_visuals || (selectedCitation.preview || selectedCitation.text || "").includes("[Visual Diagram") || (selectedCitation.preview || selectedCitation.text || "").includes("[Lecture Slide")) && (
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: "4px",
                    background: "rgba(99, 102, 241, 0.15)",
                    color: "#6366f1",
                    border: "1px solid rgba(99, 102, 241, 0.3)",
                  }}
                >
                  Diagram / Visual Context
                </span>
              )}
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
