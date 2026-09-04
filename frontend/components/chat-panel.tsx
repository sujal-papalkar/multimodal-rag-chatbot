"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Message } from "@/types";
import { MessageCitations } from "@/components/message-citations";
import { MarkdownRenderer } from "@/components/markdown-renderer";

export function ChatPanel({
  messages,
  isLoading,
  onSend,
  disabledReason,
}: {
  messages: Message[];
  isLoading: boolean;
  onSend: (q: string) => void;
  disabledReason: string | null;
}) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const send = () => {
    const q = input.trim();
    if (!q || isLoading) return;
    onSend(q);
    setInput("");
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const onInput = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  };

  return (
    <div className="chat-panel">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <div className="empty-icon">◈</div>
            <p className="empty-title">Upload documents to begin</p>
            <p className="empty-sub">Select one or more PDFs and ask questions — tables, images, and text understood.</p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`msg msg-${m.role}`}>
            <div className="avatar">{m.role === "user" ? "U" : "◈"}</div>
            <div className="msg-body">
              {m.role === "assistant" ? (
                // EDGE CASE (table rendering): remarkGfm enables GitHub-flavored
                // Markdown, which is what adds pipe-table (| a | b |) support to
                // ReactMarkdown. Without it, a well-formed markdown table renders
                // as a single run-on paragraph instead of an actual <table>.
                // This is the frontend half of the table fix — the backend half
                // is main.py's prompt now asking the model for GFM tables
                // instead of raw HTML in the first place.
                <div className="msg-md">
                  <MarkdownRenderer content={m.content} />
                </div>
              ) : (
                <p className="msg-text">{m.content}</p>
              )}
              <span className="msg-time">
                {m.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
              {/* EDGE CASE: each message renders ONLY its own citations/pageImages,
                  read directly off this message object — never off any shared state. */}
              {m.role === "assistant" && !m.isError && (
                <MessageCitations citations={m.citations ?? []} pageImages={m.pageImages ?? []} />
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="msg msg-assistant">
            <div className="avatar">◈</div>
            <div className="msg-body">
              <div className="thinking"><span /><span /><span /></div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <textarea
          ref={taRef}
          className="chat-input"
          placeholder={disabledReason ?? "Ask about your documents…"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          onInput={onInput}
          rows={1}
          disabled={isLoading}
        />
        <button className="send-btn" onClick={send} disabled={!input.trim() || isLoading}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </div>
  );
}
