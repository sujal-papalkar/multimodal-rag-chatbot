"use client";

import { DocRecord } from "@/types";

// ─── Documents Panel (multi-select + delete) ──────────────────────────────────
// EDGE CASE (theme): each row now renders as a self-contained "card" (see
// theme.css .doc-row) with its own left status-stub, rather than a plain
// list item — this is what brings it into visual parity with the rest of
// the app instead of reading like a leftover default file-list.

export function DocumentsPanel({
  docs,
  selectedIds,
  deletingIds,
  onToggle,
  onSelectAll,
  onSelectNone,
  onDelete,
}: {
  docs: DocRecord[];
  selectedIds: Set<string>;
  deletingIds: Set<string>;
  onToggle: (docId: string) => void;
  onSelectAll: () => void;
  onSelectNone: () => void;
  onDelete: (docId: string, filename: string) => void;
}) {
  if (docs.length === 0) {
    return (
      <div className="documents-panel">
        <p className="panel-label">Documents</p>
        <p className="docs-empty">No documents uploaded yet.</p>
      </div>
    );
  }

  return (
    <div className="documents-panel">
      <div className="docs-header-row">
        <p className="panel-label">Documents ({selectedIds.size}/{docs.length})</p>
        <div className="docs-header-actions">
          <button className="docs-link-btn" onClick={onSelectAll}>All</button>
          <span className="docs-sep">·</span>
          <button className="docs-link-btn" onClick={onSelectNone}>None</button>
        </div>
      </div>
      <div className="docs-list">
        {docs.map((d) => {
          const checked = selectedIds.has(d.doc_id);
          const disabled = d.status !== "ready";
          const deleting = deletingIds.has(d.doc_id);
          return (
            <div
              key={d.doc_id}
              className={`doc-row ${checked ? "doc-row-checked" : ""} ${disabled ? "doc-row-disabled" : ""} ${deleting ? "doc-row-deleting" : ""}`}
              title={d.filename}
            >
              <label className="doc-row-main">
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled || deleting}
                  onChange={() => onToggle(d.doc_id)}
                />
                <span className="doc-row-icon">📄</span>
                <span className="doc-row-name">{d.filename}</span>
                {(d.status === "processing" || d.status === "indexing") && (
                  <span className="doc-row-status">…</span>
                )}
                {d.status === "failed" && <span className="doc-row-status doc-row-failed">failed</span>}
                {d.status === "ready" && d.num_pages != null && (
                  <span className="doc-row-pages">{d.num_pages}p</span>
                )}
              </label>
              <button
                className="doc-row-delete"
                title={`Delete ${d.filename}`}
                disabled={deleting}
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(d.doc_id, d.filename);
                }}
              >
                {deleting ? (
                  <span className="doc-row-spinner" />
                ) : (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    <line x1="10" y1="11" x2="10" y2="17" />
                    <line x1="14" y1="11" x2="14" y2="17" />
                  </svg>
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
