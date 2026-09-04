export interface Citation {
  id: string;
  doc_id?: string | null;
  filename?: string | null;
  page?: number | null;
  page_end?: number | null;
  type: "text" | "web";
  content?: string | null;
  title?: string | null;
  url?: string | null;
}

export interface PageImage {
  doc_id?: string | null;
  page: number;
  image: string;
}

// EDGE CASE: citations and page_images now live ON the message itself,
// not in a separate global piece of state. This is what makes per-query
// separation structurally guaranteed on the frontend — there is no shared
// array to accidentally merge into or read the wrong entries from.
export interface Message {
  id: string; // server-issued message_id once known, else a temp client id
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  citations?: Citation[];
  pageImages?: PageImage[];
  isError?: boolean;
}

export interface DocRecord {
  doc_id: string;
  filename: string;
  num_pages: number | null;
  status: "processing" | "ready" | "failed" | "indexing";
  upload_timestamp: number;
}
