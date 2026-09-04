"use client";

import { useState, useRef, DragEvent } from "react";

export function UploadPanel({
  onUpload,
  isUploading,
  webSearchEnabled,
  onToggleWebSearch,
}: {
  onUpload: (file: File) => void;
  isUploading: boolean;
  webSearchEnabled: boolean;
  onToggleWebSearch: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file: File) => {
    if (file.type === "application/pdf") onUpload(file);
    else alert("Only PDF files are supported.");
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="upload-panel">
      <p className="panel-label">Add Document</p>
      <div
        className={`drop-zone ${dragging ? "dz-active" : ""}`}
        onClick={() => !isUploading && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }}
        />
        {isUploading ? (
          <div className="dz-inner">
            <div className="spinner" />
            <span>Processing…</span>
          </div>
        ) : (
          <div className="dz-inner">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--t3)" }}>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span>Drop PDF here</span>
            <span style={{ fontSize: 11, color: "var(--t3)" }}>or click to browse</span>
          </div>
        )}
      </div>

      <div className="toggle-row">
        <div>
          <p className="toggle-label">Web Search</p>
          <p className="toggle-sub">Supplement with live results</p>
        </div>
        <button
          className={`toggle ${webSearchEnabled ? "toggle-on" : ""}`}
          onClick={onToggleWebSearch}
          aria-pressed={webSearchEnabled}
        >
          <span className="toggle-thumb" />
        </button>
      </div>
    </div>
  );
}
