"use client";

import { useState, useEffect, useCallback } from "react";
import { DocRecord, Message } from "@/types";
import { API_BASE } from "@/lib/api";
import { UploadPanel } from "@/components/upload-panel";
import { DocumentsPanel } from "@/components/documents-panel";
import { ChatPanel } from "@/components/chat-panel";

// ─── Root Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [docs, setDocs] = useState<DocRecord[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [webSearch, setWebSearch] = useState(false);

  // EDGE CASE: conversation_id threads every /query call together so the
  // backend can resolve follow-ups ("where does it occur?") using history.
  // It is NOT used to merge citations — each response still only carries
  // its own message's citations, attached to that one message object.
  const [conversationId, setConversationId] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) return;
      const data: DocRecord[] = await res.json();
      setDocs(data);
    } catch {
      // Silently ignore — list will retry on next upload/poll
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  // Poll while any document is still processing/indexing, so status/pages update live
  useEffect(() => {
    const hasPending = docs.some((d) => d.status === "processing" || d.status === "indexing");
    if (!hasPending) return;
    const interval = setInterval(refreshDocuments, 2500);
    return () => clearInterval(interval);
  }, [docs, refreshDocuments]);

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      await refreshDocuments();
      setSelectedIds((prev) => new Set(prev).add(data.doc_id));
      setMessages((p) => [
        ...p,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.duplicate
            ? `**"${file.name}"** matches an already-uploaded document. Reusing the existing index.`
            : data.warning === "empty_content"
              ? `**"${file.name}"** was uploaded, but no searchable text was found in it (it may be blank or fully scanned images). You can still view page images.`
              : `Document **"${file.name}"** uploaded and indexed successfully. It's now selected — ask me anything about it!`,
          timestamp: new Date(),
        },
      ]);
    } catch (err: any) {
      alert("Upload failed: " + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const toggleDoc = (docId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });
  };

  const selectAllDocs = () => {
    setSelectedIds(new Set(docs.filter((d) => d.status === "ready").map((d) => d.doc_id)));
  };

  const selectNoneDocs = () => setSelectedIds(new Set());

  const handleDelete = async (docId: string, filename: string) => {
    const ok = window.confirm(`Delete "${filename}"? This removes it from the index permanently.`);
    if (!ok) return;

    setDeletingIds((prev) => new Set(prev).add(docId));
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());

      setDocs((prev) => prev.filter((d) => d.doc_id !== docId));
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });

      // NOTE: we deliberately do NOT scrub this doc's citations out of past
      // chat messages — those citations are a historical record of what was
      // actually used to answer that specific past question, and stay
      // correct/legible even after the source document is later deleted.

      setMessages((p) => [
        ...p,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Deleted **"${filename}"** and removed it from the index.`,
          timestamp: new Date(),
        },
      ]);
    } catch (err: any) {
      alert("Delete failed: " + err.message);
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });
    }
  };

  const handleQuery = async (query: string) => {
    setMessages((p) => [...p, { id: crypto.randomUUID(), role: "user", content: query, timestamp: new Date() }]);
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          doc_ids: Array.from(selectedIds),
          is_web_search_enabled: webSearch,
          // EDGE CASE: omit conversation_id entirely on the very first turn
          // so the backend creates a fresh conversation; from then on we
          // always send the id it returned, so follow-ups thread correctly.
          conversation_id: conversationId,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      // The backend always returns the conversation_id (creating one if we
      // didn't send one) and a unique message_id for this exact answer.
      if (data.conversation_id) setConversationId(data.conversation_id);

      setMessages((p) => [
        ...p,
        {
          id: data.message_id || crypto.randomUUID(),
          role: "assistant",
          content: data.answer,
          timestamp: new Date(),
          // EDGE CASE: citations/page_images are attached directly to THIS
          // message object, from THIS response only — never written to any
          // shared array, so later queries can't retroactively change what
          // this message shows.
          citations: data.citations ?? [],
          pageImages: data.page_images ?? [],
        },
      ]);
    } catch (err: any) {
      setMessages((p) => [
        ...p,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Error: " + err.message,
          timestamp: new Date(),
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    // EDGE CASE: clearing conversationId means the NEXT query creates a
    // brand-new conversation server-side — old messages/citations stay
    // exactly as they were in the (now-abandoned) prior conversation's
    // history table, they just stop being sent as context.
    setMessages([]);
    setConversationId(null);
  };

  const disabledReason =
    selectedIds.size === 0 && !webSearch
      ? "Select a document or enable Web Search to ask a question…"
      : null;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">◈</span>
          <span className="brand-name">Multi-Rag</span>
        </div>

        <button className="new-chat-btn" onClick={handleNewChat} disabled={messages.length === 0}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Chat
        </button>

        <UploadPanel
          onUpload={handleUpload}
          isUploading={isUploading}
          webSearchEnabled={webSearch}
          onToggleWebSearch={() => setWebSearch((v) => !v)}
        />
        <DocumentsPanel
          docs={docs}
          selectedIds={selectedIds}
          deletingIds={deletingIds}
          onToggle={toggleDoc}
          onSelectAll={selectAllDocs}
          onSelectNone={selectNoneDocs}
          onDelete={handleDelete}
        />
      </aside>
      <main className="main">
        <ChatPanel messages={messages} isLoading={isLoading} onSend={handleQuery} disabledReason={disabledReason} />
      </main>
    </div>
  );
}