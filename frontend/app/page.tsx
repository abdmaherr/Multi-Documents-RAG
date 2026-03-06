"use client";

import { useCallback, useEffect, useState } from "react";
import { clearSession, listDocuments, streamCompare, streamQuery } from "@/lib/api";
import type { ChatMessage, DocumentInfo } from "@/lib/types";
import ChatPanel from "@/components/ChatPanel";
import DocumentList from "@/components/DocumentList";
import DocumentStats from "@/components/DocumentStats";
import Toast from "@/components/Toast";
import UploadZone from "@/components/UploadZone";

let msgCounter = 0;
const uid = () => String(++msgCounter);

export default function Home() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeView, setActiveView] = useState<"chat" | "graph">("chat");
  const [compareMode, setCompareMode] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set());
  const [topK, setTopK] = useState(5);
  const [dark, setDark] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const prefersDark = stored === "dark" || (!stored && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setDark(prefersDark);
    document.documentElement.classList.toggle("dark", prefersDark);
  }, []);

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  }

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {
      // backend not up yet
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  function toggleDocSelect(id: string) {
    setSelectedDocIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleCompareMode() {
    setCompareMode((prev) => {
      if (prev) setSelectedDocIds(new Set());
      return !prev;
    });
  }

  async function handleSend(question: string) {
    if (compareMode && selectedDocIds.size >= 2) {
      return handleCompare(question);
    }

    const userMsg: ChatMessage = { id: uid(), role: "user", content: question };
    const assistantId = uid();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    try {
      await streamQuery(
        question,
        sessionId,
        ({ citations, session_id }) => {
          if (!sessionId) setSessionId(session_id);
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, citations } : m))
          );
        },
        (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            )
          );
        },
        (sid) => {
          setSessionId(sid);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m
            )
          );
        },
        topK
      );
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: "Error: " + (e instanceof Error ? e.message : "Unknown error"),
                streaming: false,
              }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  }

  async function handleCompare(question: string) {
    const docNames = Array.from(selectedDocIds)
      .map((id) => documents.find((d) => d.id === id)?.filename ?? id)
      .join(", ");

    const userMsg: ChatMessage = {
      id: uid(),
      role: "user",
      content: `[Compare] ${question}\n(${docNames})`,
    };
    const assistantId = uid();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
      streaming: true,
      isComparison: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    try {
      await streamCompare(
        question,
        Array.from(selectedDocIds),
        ({ citations }) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, citations } : m))
          );
        },
        (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            )
          );
        },
        () => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m
            )
          );
        },
        topK
      );
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: "Error: " + (e instanceof Error ? e.message : "Unknown error"),
                streaming: false,
              }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  }

  async function handleClearSession() {
    if (sessionId) await clearSession(sessionId);
    setSessionId(null);
    setMessages([]);
  }

  const totalChunks = documents.reduce((sum, d) => sum + (d.chunk_count || 0), 0);

  return (
    <div className="flex flex-col h-screen bg-warm-white text-charcoal overflow-hidden">
      <Toast message={toast} type="error" onDismiss={() => setToast(null)} />
      {/* Top nav bar */}
      <header className="flex items-center h-12 border-b border-border bg-card shrink-0">
        {/* Left: hamburger + branding (matches sidebar width when open) */}
        <div className={`flex items-center gap-3 px-4 h-full shrink-0 transition-all duration-200 ${sidebarOpen ? "w-60 border-r border-border" : ""}`}>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-charcoal/50 hover:text-charcoal transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50 rounded"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-white bg-coral rounded-md px-1.5 py-0.5 tracking-wide">RAG</span>
            <span className="font-heading text-deep text-base whitespace-nowrap">Multi-Doc Pipeline</span>
          </div>
        </div>

        {/* Center: nav + Right: actions */}
        <div className="flex-1 flex items-center justify-between px-4 h-full">
          <div className="flex-1" />
          <nav className="hidden sm:flex items-center gap-1">
            <button
              onClick={() => { setActiveView("chat"); if (compareMode) toggleCompareMode(); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeView === "chat" && !compareMode
                  ? "bg-coral-light text-coral"
                  : "text-charcoal/50 hover:text-charcoal hover:bg-warm-white"
              }`}
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" /></svg>
              Chat
            </button>
            {documents.length >= 2 && (
              <button
                onClick={() => { setActiveView("chat"); toggleCompareMode(); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  compareMode
                    ? "bg-teal-light text-teal"
                    : "text-charcoal/50 hover:text-charcoal hover:bg-warm-white"
                }`}
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" /></svg>
                Compare
              </button>
            )}
            <button
              onClick={() => { setActiveView("graph"); if (compareMode) toggleCompareMode(); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeView === "graph"
                  ? "bg-amber-light text-amber"
                  : "text-charcoal/50 hover:text-charcoal hover:bg-warm-white"
              }`}
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg>
              Graph
            </button>
          </nav>

          <div className="flex-1 flex items-center justify-end gap-3">
            {messages.length > 0 && (
              <button
                onClick={handleClearSession}
                className="text-xs text-charcoal/60 hover:text-coral transition-colors font-medium"
              >
                Clear
              </button>
            )}
            <button
              onClick={toggleTheme}
              className="text-charcoal/50 hover:text-charcoal transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50 rounded p-1"
              title={dark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {dark ? (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
                </svg>
              ) : (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar */}
        <aside
          style={{ width: sidebarOpen ? 240 : 0 }}
          className={`
            shrink-0 flex flex-col
            bg-card transition-all duration-200 overflow-hidden
            ${sidebarOpen ? "border-r border-border" : ""}
          `}
        >
          {/* Documents header */}
          <div className="px-4 pt-4 pb-2 min-w-60">
            <div className="flex items-center gap-2">
              <span className="text-base">📁</span>
              <h2 className="font-heading text-slate-deep text-lg">Documents</h2>
            </div>
            <p className="text-xs text-teal mt-0.5">
              {documents.length} file{documents.length !== 1 ? "s" : ""} uploaded
            </p>
          </div>

          {/* Upload */}
          <div className="px-4 py-3">
            <UploadZone onUploadComplete={fetchDocuments} onToast={setToast} />
          </div>

          {/* Documents list */}
          <div className="flex-1 overflow-y-auto px-4">
            <DocumentList
              documents={documents}
              onDeleted={(id) => {
                setDocuments((prev) => prev.filter((d) => d.id !== id));
                setSelectedDocIds((prev) => {
                  const next = new Set(prev);
                  next.delete(id);
                  return next;
                });
              }}
              compareMode={compareMode}
              selectedDocIds={selectedDocIds}
              onToggleSelect={toggleDocSelect}
            />
          </div>

          {/* Corpus stats footer */}
          <div className="px-4 py-3 border-t border-border">
            <p className="text-xs font-semibold text-charcoal/50 uppercase tracking-wider mb-2 flex items-center gap-1">
              <span className="text-xs">📊</span> Corpus Stats
            </p>
            <div className="flex justify-between">
              <div className="text-center">
                <p className="text-lg font-mono font-semibold text-slate-deep">{documents.length}</p>
                <p className="text-xs text-charcoal/50 uppercase">Docs</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-mono font-semibold text-slate-deep">{totalChunks}</p>
                <p className="text-xs text-charcoal/50 uppercase">Chunks</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-mono font-semibold text-slate-deep">
                  {totalChunks > 0 ? `${(totalChunks * 100 / 1000).toFixed(1)}k` : "0"}
                </p>
                <p className="text-xs text-charcoal/50 uppercase">Tokens</p>
              </div>
            </div>
          </div>
        </aside>

        {/* Main content area */}
        <main className="flex-1 flex flex-col min-w-0 dot-grid">
          {activeView === "chat" ? (
            <ChatPanel
              messages={messages}
              isStreaming={isStreaming}
              onSend={handleSend}
              onClearSession={handleClearSession}
              onOpenSidebar={() => setSidebarOpen(true)}
              hasDocuments={documents.length > 0}
              compareMode={compareMode}
              compareDocCount={selectedDocIds.size}
              topK={topK}
              onTopKChange={setTopK}
            />
          ) : (
            <DocumentStats documents={documents} />
          )}
        </main>
      </div>
    </div>
  );
}
