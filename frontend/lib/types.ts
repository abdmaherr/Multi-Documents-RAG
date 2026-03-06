export interface DocumentInfo {
  id: string;
  filename: string;
  content_type: string;
  chunk_count: number;
  status: "ready" | "error" | "processing";
}

export interface SourceCitation {
  document: string;
  chunk_text: string;
  score: number;
  page: number | null;
  section: string | null;
}

export interface ProcessingEvent {
  step: string;
  status: string;
  detail: string | null;
  progress: number | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: SourceCitation[];
  streaming?: boolean;
  isComparison?: boolean;
}
