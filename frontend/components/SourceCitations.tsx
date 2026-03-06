"use client";

import { useState } from "react";
import type { SourceCitation } from "@/lib/types";

interface Props {
  citations: SourceCitation[];
}

export default function SourceCitations({ citations }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (citations.length === 0) return null;

  return (
    <div className="mt-1.5 space-y-1.5">
      <p className="text-xs font-semibold text-charcoal/50 uppercase tracking-wider">
        Sources
      </p>
      {citations.map((c, i) => (
        <div key={i} className="rounded-md border border-border bg-card overflow-hidden">
          <button
            onClick={() => setExpanded(expanded === i ? null : i)}
            className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-coral-light transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-coral/50 focus-visible:ring-inset"
          >
            <span className="flex-1 min-w-0">
              <span className="text-xs font-medium text-coral truncate block">
                {c.document}
              </span>
              {(c.page != null || c.section) && (
                <span className="text-xs text-charcoal/50">
                  {c.page != null && `p.${c.page}`}
                  {c.page != null && c.section && " · "}
                  {c.section}
                </span>
              )}
            </span>
            <span
              className="text-xs font-mono px-1.5 py-0.5 rounded bg-warm-white text-charcoal/60 shrink-0"
              title="Similarity score (higher = more relevant)"
            >
              {c.score.toFixed(2)}
            </span>
            <svg
              className={`h-3.5 w-3.5 text-charcoal/50 shrink-0 transition-transform ${expanded === i ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expanded === i && (
            <div className="px-3 pb-3 pt-0">
              <p className="text-xs text-charcoal/60 leading-relaxed border-t border-border pt-2">
                {c.chunk_text}
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
