"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkBreaks from "remark-breaks";
import rehypeKatex from "rehype-katex";

// ─── Code Block Component with Copy to Clipboard ─────────────────────────────
interface CodeBlockProps {
  language: string;
  code: string;
}

function CodeBlock({ language, code }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code to clipboard:", err);
    }
  };

  return (
    <div className="code-block">
      <div className="code-header">
        <span className="code-lang">{language || "code"}</span>
        <button
          type="button"
          className="code-copy-btn"
          onClick={handleCopy}
          aria-label="Copy code to clipboard"
        >
          {copied ? (
            <>
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Copied!</span>
            </>
          ) : (
            <>
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="code-content">
        <code className={language ? `language-${language}` : ""}>{code}</code>
      </pre>
    </div>
  );
}

// ─── Preprocessing Helpers ───────────────────────────────────────────────────

/**
 * Normalizes LaTeX math delimiters:
 * - Converts display math `\[ ... \]` to `$$ ... $$`
 * - Converts inline math `\( ... \)` to `$ ... $`
 */
function normalizeLatex(text: string): string {
  // Convert \[ ... \] to \n$$\n...\n$$\n
  let out = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
    return `\n$$\n${math.trim()}\n$$\n`;
  });

  // Convert \( ... \) to $...$
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => {
    return `$${math.trim()}$`;
  });

  return out;
}

/**
 * Protects in-text citations such as `[annual_report_2023.pdf, Page 12]`
 * from being parsed as italics due to underscores in filenames.
 */
function protectCitations(text: string): string {
  return text.replace(/\[([^\]\n]+?,\s*Pages?\s*[\d\-–toTO ]+)\]/gi, (match, inner) => {
    // Replace underscores with HTML character entity &#95; inside citations
    const safeInner = inner.replace(/_/g, "&#95;");
    return `[${safeInner}]`;
  });
}

export function preprocessMarkdown(content: string): string {
  if (!content) return "";
  let processed = normalizeLatex(content);
  processed = protectCitations(processed);
  return processed;
}

// ─── MarkdownRenderer Component ──────────────────────────────────────────────

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const processedContent = preprocessMarkdown(content);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
      components={{
        pre({ children }) {
          if (React.isValidElement(children)) {
            const childProps = children.props as any;
            const rawCode = String(childProps?.children || "").replace(/\n$/, "");
            const match = /language-(\w+)/.exec(childProps?.className || "");
            const language = match ? match[1] : "";
            return <CodeBlock language={language} code={rawCode} />;
          }
          return <pre className="code-fallback">{children}</pre>;
        },
        code({ className, children, ...props }) {
          return (
            <code className={`inline-code ${className || ""}`} {...props}>
              {children}
            </code>
          );
        },
        table({ children }) {
          return (
            <div className="table-wrapper">
              <table className="content-table">{children}</table>
            </div>
          );
        },
        a({ href, children, ...props }) {
          const isExternal =
            typeof href === "string" &&
            (href.startsWith("http://") || href.startsWith("https://"));
          return (
            <a
              href={href}
              target={isExternal ? "_blank" : undefined}
              rel={isExternal ? "noopener noreferrer" : undefined}
              className="msg-link"
              {...props}
            >
              {children}
              {isExternal && (
                <svg
                  className="link-ext-icon"
                  width="11"
                  height="11"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                  <polyline points="15 3 21 3 21 9" />
                  <line x1="10" y1="14" x2="21" y2="3" />
                </svg>
              )}
            </a>
          );
        },
        h1({ children }) {
          return <h1 className="md-h1">{children}</h1>;
        },
        h2({ children }) {
          return <h2 className="md-h2">{children}</h2>;
        },
        h3({ children }) {
          return <h3 className="md-h3">{children}</h3>;
        },
        h4({ children }) {
          return <h4 className="md-h4">{children}</h4>;
        },
        h5({ children }) {
          return <h5 className="md-h5">{children}</h5>;
        },
        h6({ children }) {
          return <h6 className="md-h6">{children}</h6>;
        },
        blockquote({ children }) {
          return <blockquote className="md-blockquote">{children}</blockquote>;
        },
        hr() {
          return <hr className="md-hr" />;
        },
      }}
    >
      {processedContent}
    </ReactMarkdown>
  );
}
