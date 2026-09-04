"use client";

import { useState } from "react";
import { Citation, PageImage } from "@/types";

// ─── Per-Message Citation Panel ────────────────────────────────────────────────
// EDGE CASE: this now renders scoped to a single message's own citations
// and page_images, passed in as props from that exact message object.
// There is no shared/global citation state anywhere for it to read from,
// so it is structurally impossible for it to show another query's sources.

export function MessageCitations({ citations, pageImages }: { citations: Citation[]; pageImages: PageImage[] }) {
  const [expanded, setExpanded] = useState(false);
  const [tab, setTab] = useState<"sources" | "pages">("sources");
  const [lightbox, setLightbox] = useState<string | null>(null);

  const total = citations.length;
  if (total === 0 && pageImages.length === 0) return null;

  const docCitations = citations.filter((c) => c.type === "text");
  const webCitations = citations.filter((c) => c.type === "web");

  return (
    <div className="msg-citations">
      <button
        className="view-citations-btn"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "Hide" : "View"} {total} {total === 1 ? "Citation" : "Citations"}
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round"
          style={{ marginLeft: 4, transform: expanded ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {expanded && (
        <div className="citation-panel citation-panel-inline">
          <div className="ctabs">
            <button className={`ctab ${tab === "sources" ? "ctab-active" : ""}`} onClick={() => setTab("sources")}>
              Sources <span className="badge">{citations.length}</span>
            </button>
            {pageImages.length > 0 && (
              <button className={`ctab ${tab === "pages" ? "ctab-active" : ""}`} onClick={() => setTab("pages")}>
                Pages <span className="badge">{pageImages.length}</span>
              </button>
            )}
          </div>

          {tab === "sources" && (
            <div className="clist">
              {docCitations.length > 0 && (
                <>
                  <p className="cgroup">From Documents</p>
                  {docCitations.map((c) => (
                    <div key={c.id} className="ccard ccard-doc">
                      <div className="ccard-header">
                        <span className="ccard-type">📄 {c.filename ?? "Document"}</span>
                        {c.page != null && (
                          <span className="ccard-page">
                            p.{c.page}{c.page_end != null && c.page_end !== c.page ? `-${c.page_end}` : ""}
                          </span>
                        )}
                      </div>
                      {c.content && <p className="ccard-text">{c.content}</p>}
                    </div>
                  ))}
                </>
              )}
              {webCitations.length > 0 && (
                <>
                  <p className="cgroup">From Web</p>
                  {webCitations.map((c) => (
                    <a key={c.id} href={c.url ?? "#"} target="_blank" rel="noopener noreferrer" className="ccard ccard-web">
                      <div className="ccard-header">
                        <span className="ccard-type">🌐 Web</span>
                      </div>
                      {c.title && <p className="ccard-title">{c.title}</p>}
                      {c.content && <p className="ccard-text">{c.content}</p>}
                    </a>
                  ))}
                </>
              )}
              {citations.length === 0 && <p className="docs-empty">No sources for this answer.</p>}
            </div>
          )}

          {tab === "pages" && (
            <div className="clist">
              {pageImages.map((pi) => (
                <div key={`${pi.doc_id ?? "doc"}-${pi.page}`} className="page-card" onClick={() => setLightbox(pi.image)}>
                  <img src={`data:image/png;base64,${pi.image}`} alt={`Page ${pi.page}`} className="page-thumb" />
                  <span className="page-label">Page {pi.page}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img src={`data:image/png;base64,${lightbox}`} alt="Page" className="lightbox-img" />
          <button className="lightbox-close">✕</button>
        </div>
      )}
    </div>
  );
}
