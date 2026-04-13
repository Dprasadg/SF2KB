import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, X } from "lucide-react";
import api from "../api";

const TEMPLATE_LABELS = {
  solution: { label: "Solution", color: "bg-purple-50 text-purple-700" },
  how_to: { label: "How To", color: "bg-green-50 text-green-700" },
  qa: { label: "Q&A", color: "bg-sky-50 text-sky-700" },
};

function toList(value) {
  if (Array.isArray(value)) {
    return value.filter((item) => String(item || "").trim());
  }
  if (typeof value === "string" && value.trim()) {
    return [value.trim()];
  }
  return [];
}

function highlightText(text, terms) {
  const value = String(text || "");
  if (!value || !Array.isArray(terms) || terms.length === 0) {
    return value;
  }

  const escaped = terms
    .map((t) => String(t || "").trim())
    .filter(Boolean)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (escaped.length === 0) {
    return value;
  }

  const regex = new RegExp(`(${escaped.join("|")})`, "ig");
  const parts = value.split(regex);

  return parts.map((part, idx) =>
    regex.test(part)
      ? <mark key={`${part}-${idx}`} className="rounded bg-yellow-200 px-0.5">{part}</mark>
      : <span key={`${part}-${idx}`}>{part}</span>,
  );
}

function ResultCard({ result, queryTerms }) {
  const [open, setOpen] = useState(true);
  const kb = result?.kb || result;

  const title = kb?.title || "Untitled article";
  const templateType = kb?.template_type || "solution";
  const templateMeta = TEMPLATE_LABELS[templateType] || TEMPLATE_LABELS.solution;
  const summary = kb?.problem_summary || kb?.summary || kb?.objective || kb?.answer || "";
  const symptoms = toList(kb?.symptoms);
  const appliesTo = toList(kb?.applies_to || kb?.appliesTo);
  const rootCause = kb?.root_cause || kb?.cause || "";
  const resolutionSteps = toList(kb?.resolution_steps || kb?.resolution || kb?.steps);
  const keywordVariations = toList(kb?.keyword_variations || kb?.keywordVariations);
  const legacyKeywords = toList(kb?.tags?.keywords || kb?.keywords);
  const keywords = useMemo(
    () => Array.from(new Set([...keywordVariations, ...legacyKeywords])),
    [keywordVariations, legacyKeywords],
  );

  const additionalInfo = kb?.additional_info || kb?.additionalInformation || "";
  const visibility = kb?.visibility || "";
  const validationState = kb?.validation_state || "";
  const scorePct = typeof result?.score === "number" ? Math.round(result.score * 100) : null;
  const matchedTerms = toList(result?.matched_terms);

  return (
    <article className="rounded-2xl border bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-lg">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded px-2 py-0.5 text-xs font-semibold ${templateMeta.color}`}>{templateMeta.label}</span>
            {visibility && <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">{visibility}</span>}
            {validationState && <span className="rounded bg-blue-50 px-2 py-0.5 text-xs text-blue-700">{validationState}</span>}
            {scorePct !== null && <span className="rounded bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">Relevance {scorePct}%</span>}
          </div>

          <h3 className="mt-2 text-lg font-semibold">{highlightText(title, queryTerms)}</h3>
          {summary && <p className="mt-1 text-gray-600">{highlightText(summary, queryTerms)}</p>}

          {matchedTerms.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {matchedTerms.map((term, index) => (
                <span key={`${term}-${index}`} className="rounded-full bg-yellow-50 px-2 py-0.5 text-xs text-yellow-700">
                  Match: {term}
                </span>
              ))}
            </div>
          )}
        </div>

        <button type="button" onClick={() => setOpen(!open)} className="rounded-md p-1 hover:bg-gray-100">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {open && (
        <div className="mt-4 space-y-4">
          {appliesTo.length > 0 && (
            <section>
              <p className="text-sm font-medium">Applies To</p>
              <div className="mt-1 flex flex-wrap gap-2">
                {appliesTo.map((item, index) => (
                  <span key={index} className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-700">
                    {item}
                  </span>
                ))}
              </div>
            </section>
          )}

          {symptoms.length > 0 && (
            <section>
              <p className="text-sm font-medium">Symptoms</p>
              <ul className="list-inside list-disc text-sm text-gray-600">
                {symptoms.map((symptom, index) => (
                  <li key={index}>{symptom}</li>
                ))}
              </ul>
            </section>
          )}

          {rootCause && (
            <p className="text-sm">
              <span className="font-medium">Root Cause:</span> {rootCause}
            </p>
          )}

          {resolutionSteps.length > 0 && (
            <section>
              <p className="text-sm font-medium">Resolution Steps</p>
              <ol className="list-inside list-decimal text-sm text-gray-600">
                {resolutionSteps.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ol>
            </section>
          )}

          {additionalInfo && (
            <p className="text-sm text-gray-700">
              <span className="font-medium">Additional Information:</span> {additionalInfo}
            </p>
          )}

          {keywords.length > 0 && (
            <section>
              <p className="text-sm font-medium">Keyword Variations</p>
              <div className="mt-1 flex flex-wrap gap-2">
                {keywords.map((tag, index) => (
                  <span key={index} className="rounded-full bg-gray-200 px-2 py-1 text-xs">
                    {tag}
                  </span>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </article>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);
  const [templateFilter, setTemplateFilter] = useState("all");

  const search = async (value) => {
    const text = String(value || query).trim();
    if (!text) {
      setResults([]);
      setSearched(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.post("/search", { query: text });
      const payload = response.data;
      const normalized = Array.isArray(payload) ? payload : payload?.results || [];
      setResults(normalized);
      setSearched(true);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || "Search failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 3) {
      if (!trimmed.length) {
        setResults([]);
        setSearched(false);
      }
      return undefined;
    }

    const timer = setTimeout(() => {
      search(trimmed);
    }, 350);

    return () => clearTimeout(timer);
  }, [query]);

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      search(query);
    }
  };

  const queryTerms = useMemo(
    () => query.toLowerCase().split(/\s+/).map((t) => t.trim()).filter((t) => t.length > 1),
    [query],
  );

  const templateCounts = useMemo(() => {
    const counts = { all: results.length, solution: 0, how_to: 0, qa: 0 };
    results.forEach((result) => {
      const kb = result?.kb || result;
      const type = kb?.template_type || "solution";
      if (counts[type] !== undefined) {
        counts[type] += 1;
      }
    });
    return counts;
  }, [results]);

  const filteredResults = useMemo(() => {
    if (templateFilter === "all") {
      return results;
    }
    return results.filter((result) => {
      const kb = result?.kb || result;
      return (kb?.template_type || "solution") === templateFilter;
    });
  }, [results, templateFilter]);

  return (
    <section className="w-full py-2 lg:p-8">
      <h2 className="mb-3 text-3xl font-semibold">Search Knowledge Base</h2>
      <p className="mb-6 text-sm text-gray-500">Type at least 3 characters for live search, or press Enter to search instantly.</p>

      <div className="mb-4 flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search your knowledge base..."
            className="w-full rounded-xl border border-gray-300 p-3 pr-10 focus:outline-none focus:ring-2 focus:ring-black"
          />
          {query && (
            <button
              type="button"
              onClick={() => {
                setQuery("");
                setResults([]);
                setSearched(false);
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-gray-500 hover:bg-gray-100"
            >
              <X size={16} />
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={() => search(query)}
          disabled={loading || !query.trim()}
          className="rounded-xl bg-black px-6 py-3 text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-400"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {results.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {[
            { key: "all", label: `All (${templateCounts.all})` },
            { key: "solution", label: `Solution (${templateCounts.solution})` },
            { key: "how_to", label: `How To (${templateCounts.how_to})` },
            { key: "qa", label: `Q&A (${templateCounts.qa})` },
          ].map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setTemplateFilter(item.key)}
              className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                templateFilter === item.key
                  ? "bg-black text-white"
                  : "border border-gray-300 bg-white text-gray-700 hover:bg-gray-100"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}

      {error && <p className="mb-4 text-sm text-red-700">{error}</p>}
      {searched && !loading && results.length === 0 && (
        <p className="mb-4 text-sm text-gray-500">No matching approved articles found.</p>
      )}

      {!loading && filteredResults.length > 0 && (
        <p className="mb-3 text-sm text-gray-500">Showing {filteredResults.length} result(s)</p>
      )}

      <div className="grid gap-4">
        {filteredResults.map((result, index) => (
          <ResultCard key={result?.kb?.kb_id || result?.kb?.title || index} result={result} queryTerms={queryTerms} />
        ))}
      </div>
    </section>
  );
}
