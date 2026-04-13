import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";
import api from "../api";

const TEMPLATE_LABELS = {
  solution: { label: "Solution", color: "bg-purple-50 text-purple-700" },
  how_to: { label: "How To", color: "bg-green-50 text-green-700" },
  qa: { label: "Q&A", color: "bg-sky-50 text-sky-700" },
};

const PRIORITY_LABELS = {
  critical: { label: "Critical", color: "bg-red-100 text-red-700" },
  high: { label: "High", color: "bg-orange-100 text-orange-700" },
  medium: { label: "Medium", color: "bg-yellow-100 text-yellow-700" },
  low: { label: "Low", color: "bg-gray-100 text-gray-600" },
};

const APPLIES_TO_OPTIONS = ["All Products", "Digital Safe", "Enterprise Archive"];
const VALIDATION_OPTIONS = ["all", "Validated", "Not Validated"];
const VISIBILITY_OPTIONS = ["all", "Visible in Internal App", "Visible in Public KB", "Visible to Customer"];
const APPROVAL_OPTIONS = [
  { key: "approved", label: "Approved" },
  { key: "pending", label: "Pending" },
  { key: "needs_edits", label: "Needs Edits" },
  { key: "all", label: "All" },
];

function toList(value) {
  if (Array.isArray(value)) return value.filter((item) => String(item || "").trim());
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function highlightText(text, terms) {
  const value = String(text || "");
  if (!value || !Array.isArray(terms) || terms.length === 0) return value;

  const escaped = terms
    .map((t) => String(t || "").trim())
    .filter(Boolean)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (escaped.length === 0) return value;

  const regex = new RegExp(`(${escaped.join("|")})`, "ig");
  const parts = value.split(regex);

  return parts.map((part, idx) => (
    idx % 2 === 1
      ? <mark key={`${part}-${idx}`} className="rounded bg-yellow-200 px-0.5">{part}</mark>
      : <span key={`${part}-${idx}`}>{part}</span>
  ));
}

function TemplatePreview({ kb, queryTerms }) {
  const type = kb?.template_type || "solution";
  const appliesTo = toList(kb?.applies_to);

  if (type === "how_to") {
    return (
      <div className="space-y-3">
        {kb?.objective && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Objective</p>
            <p className="mt-1 text-sm text-gray-700">{highlightText(kb.objective, queryTerms)}</p>
          </section>
        )}
        {appliesTo.length > 0 && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Applies To</p>
            <div className="mt-1 flex flex-wrap gap-2">
              {appliesTo.map((item, index) => <span key={index} className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-700">{item}</span>)}
            </div>
          </section>
        )}
        {toList(kb?.steps).length > 0 && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Steps</p>
            <ol className="mt-1 list-inside list-decimal text-sm text-gray-600">
              {toList(kb.steps).map((step, index) => <li key={index}>{highlightText(step, queryTerms)}</li>)}
            </ol>
          </section>
        )}
        {kb?.additional_info && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Additional Information</p>
            <p className="mt-1 text-sm text-gray-700">{highlightText(kb.additional_info, queryTerms)}</p>
          </section>
        )}
      </div>
    );
  }

  if (type === "qa") {
    return (
      <div className="space-y-3">
        {kb?.answer && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Answer</p>
            <p className="mt-1 text-sm text-gray-700">{highlightText(kb.answer, queryTerms)}</p>
          </section>
        )}
        {appliesTo.length > 0 && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Applies To</p>
            <div className="mt-1 flex flex-wrap gap-2">
              {appliesTo.map((item, index) => <span key={index} className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-700">{item}</span>)}
            </div>
          </section>
        )}
        {kb?.additional_info && (
          <section>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Additional Information</p>
            <p className="mt-1 text-sm text-gray-700">{highlightText(kb.additional_info, queryTerms)}</p>
          </section>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {kb?.summary && (
        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Summary</p>
          <p className="mt-1 text-sm text-gray-700">{highlightText(kb.summary, queryTerms)}</p>
        </section>
      )}
      {toList(kb?.symptoms).length > 0 && (
        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Symptoms</p>
          <ul className="mt-1 list-inside list-disc text-sm text-gray-600">
            {toList(kb.symptoms).map((symptom, index) => <li key={index}>{highlightText(symptom, queryTerms)}</li>)}
          </ul>
        </section>
      )}
      {appliesTo.length > 0 && (
        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Applies To</p>
          <div className="mt-1 flex flex-wrap gap-2">
            {appliesTo.map((item, index) => <span key={index} className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-700">{item}</span>)}
          </div>
        </section>
      )}
      {toList(kb?.resolution).length > 0 && (
        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Resolution</p>
          <ol className="mt-1 list-inside list-decimal text-sm text-gray-600">
            {toList(kb.resolution).map((step, index) => <li key={index}>{highlightText(step, queryTerms)}</li>)}
          </ol>
        </section>
      )}
      {kb?.cause && (
        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Cause</p>
          <p className="mt-1 text-sm text-gray-700">{highlightText(kb.cause, queryTerms)}</p>
        </section>
      )}
      {kb?.additional_info && (
        <section>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Additional Information</p>
          <p className="mt-1 text-sm text-gray-700">{highlightText(kb.additional_info, queryTerms)}</p>
        </section>
      )}
    </div>
  );
}

function MatchDetails({ result }) {
  const [open, setOpen] = useState(false);
  const fields = toList(result?.matched_fields);
  const terms = toList(result?.matched_terms);

  return (
    <section className="rounded-xl border border-gray-200 bg-gray-50 p-3">
      <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wide text-gray-600">
        Why This Matched
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {open && (
        <div className="mt-2 space-y-2 text-xs text-gray-700">
          <div className="flex flex-wrap gap-2">
            <span className="rounded bg-emerald-100 px-2 py-1">Final: {Math.round((result?.score || 0) * 100)}%</span>
            <span className="rounded bg-blue-100 px-2 py-1">Semantic: {Math.round((result?.semantic_score || 0) * 100)}%</span>
            <span className="rounded bg-purple-100 px-2 py-1">Keyword: {Math.round((result?.keyword_score || 0) * 100)}%</span>
            <span className="rounded bg-amber-100 px-2 py-1">BM25: {Math.round((result?.bm25_score || 0) * 100)}%</span>
          </div>

          {fields.length > 0 && <p>Matched fields: {fields.join(", ")}</p>}
          {terms.length > 0 && <p>Matched terms: {terms.join(", ")}</p>}
        </div>
      )}
    </section>
  );
}

function ResultCard({ result, queryTerms, isFirst }) {
  const [open, setOpen] = useState(isFirst);
  const kb = result?.kb || result;

  const title = kb?.title || "Untitled article";
  const templateType = kb?.template_type || "solution";
  const templateMeta = TEMPLATE_LABELS[templateType] || TEMPLATE_LABELS.solution;
  const priorityMeta = PRIORITY_LABELS[String(kb?.priority || "").toLowerCase()];
  const confidenceScore = kb?.confidence_score != null ? Math.round(kb.confidence_score * 100) : null;

  return (
    <article className="rounded-2xl border bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-lg">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded px-2 py-0.5 text-xs font-semibold ${templateMeta.color}`}>{templateMeta.label}</span>
            {priorityMeta && <span className={`rounded px-2 py-0.5 text-xs font-semibold ${priorityMeta.color}`}>{priorityMeta.label}</span>}
            {kb?.visibility && <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">{kb.visibility}</span>}
            {kb?.review_status && <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-700">{kb.review_status.replace("_", " ")}</span>}
            {confidenceScore !== null && <span className="rounded bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700">Confidence {confidenceScore}%</span>}
          </div>

          <h3 className="mt-2 text-lg font-semibold leading-snug">{highlightText(title, queryTerms)}</h3>
          <p className="mt-1 text-gray-600">{highlightText(kb?.summary || kb?.objective || kb?.answer || "", queryTerms)}</p>
        </div>

        <button type="button" onClick={() => setOpen(!open)} className="rounded-md p-1 hover:bg-gray-100">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {open && (
        <div className="mt-4 space-y-4 border-t pt-4">
          <TemplatePreview kb={kb} queryTerms={queryTerms} />
          <MatchDetails result={result} />
          {toList(kb?.validation_issues).length > 0 && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-red-700">Validation Issues</p>
              <ul className="mt-1 list-inside list-disc text-sm text-red-700">
                {toList(kb.validation_issues).map((issue, i) => <li key={i}>{issue}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);

  const [templateFilter, setTemplateFilter] = useState("all");
  const [appliesToFilter, setAppliesToFilter] = useState("All Products");
  const [validationFilter, setValidationFilter] = useState("all");
  const [visibilityFilter, setVisibilityFilter] = useState("all");
  const [approvalFilter, setApprovalFilter] = useState("approved");

  const doSearch = useCallback(async (text) => {
    const trimmed = String(text || "").trim();
    if (trimmed.length < 3) return;

    setLoading(true);
    setError(null);

    const payload = {
      query: trimmed,
      top_k: 10,
      candidate_k: 40,
      approval_status: approvalFilter,
    };

    if (templateFilter !== "all") payload.template_type = templateFilter;
    if (appliesToFilter !== "All Products") payload.applies_to = appliesToFilter;
    if (validationFilter !== "all") payload.validation_state = validationFilter;
    if (visibilityFilter !== "all") payload.visibility = visibilityFilter;

    try {
      const response = await api.post("/search", payload);
      const payload2 = response.data;
      const normalized = Array.isArray(payload2) ? payload2 : payload2?.results || [];
      setResults(normalized);
      setTotal(payload2?.total ?? normalized.length);
      setSearched(true);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || "Search failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [templateFilter, appliesToFilter, validationFilter, visibilityFilter, approvalFilter]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 3) {
      if (!trimmed.length) {
        setResults([]);
        setSearched(false);
        setTotal(0);
      }
      return undefined;
    }

    const timer = setTimeout(() => doSearch(trimmed), 350);
    return () => clearTimeout(timer);
  }, [query, doSearch]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length >= 3 && searched) {
      doSearch(trimmed);
    }
  }, [templateFilter, appliesToFilter, validationFilter, visibilityFilter, approvalFilter]);

  const clearSearch = () => {
    setQuery("");
    setResults([]);
    setSearched(false);
    setTotal(0);
    setError(null);
  };

  const queryTerms = useMemo(
    () => query.toLowerCase().split(/\s+/).map((token) => token.trim()).filter((token) => token.length > 1),
    [query],
  );

  const TEMPLATE_TABS = [
    { key: "all", label: "All" },
    { key: "solution", label: "Solution" },
    { key: "how_to", label: "How To" },
    { key: "qa", label: "Q&A" },
  ];

  return (
    <section className="w-full py-2 lg:p-8">
      <div className="mb-3 flex flex-wrap items-baseline gap-3">
        <h2 className="text-3xl font-semibold">Search Knowledge Base</h2>
        {searched && !loading && <span className="text-sm text-gray-500">{total} result{total !== 1 ? "s" : ""}</span>}
      </div>
      <p className="mb-4 text-sm text-gray-500">Type at least 3 characters and use filters to narrow results.</p>

      <div className="mb-4 flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && doSearch(query)}
            placeholder="Search your knowledge base..."
            className="w-full rounded-xl border border-gray-300 py-3 pl-9 pr-10 focus:outline-none focus:ring-2 focus:ring-black"
          />
          {query && (
            <button type="button" onClick={clearSearch} className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-gray-500 hover:bg-gray-100">
              <X size={16} />
            </button>
          )}
        </div>
        <button type="button" onClick={() => doSearch(query)} disabled={loading || !query.trim()} className="rounded-xl bg-black px-6 py-3 text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-400">
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      <div className="mb-4 grid gap-3 rounded-2xl border bg-white p-4 sm:grid-cols-2 lg:grid-cols-5">
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Type</p>
          <div className="flex flex-wrap gap-1">
            {TEMPLATE_TABS.map((item) => (
              <button key={item.key} type="button" onClick={() => setTemplateFilter(item.key)} className={`rounded-full px-2 py-1 text-xs ${templateFilter === item.key ? "bg-black text-white" : "border border-gray-300 text-gray-700"}`}>
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Product</p>
          <select value={appliesToFilter} onChange={(e) => setAppliesToFilter(e.target.value)} className="w-full rounded-lg border border-gray-300 bg-white px-2 py-1.5 text-xs">
            {APPLIES_TO_OPTIONS.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Validation</p>
          <select value={validationFilter} onChange={(e) => setValidationFilter(e.target.value)} className="w-full rounded-lg border border-gray-300 bg-white px-2 py-1.5 text-xs">
            {VALIDATION_OPTIONS.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Visibility</p>
          <select value={visibilityFilter} onChange={(e) => setVisibilityFilter(e.target.value)} className="w-full rounded-lg border border-gray-300 bg-white px-2 py-1.5 text-xs">
            {VISIBILITY_OPTIONS.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </div>

        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Approval Status</p>
          <div className="flex flex-wrap gap-1">
            {APPROVAL_OPTIONS.map((opt) => (
              <button key={opt.key} type="button" onClick={() => setApprovalFilter(opt.key)} className={`rounded-full px-2 py-1 text-xs ${approvalFilter === opt.key ? "bg-black text-white" : "border border-gray-300 text-gray-700"}`}>
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error} Check API health and try again.
        </div>
      )}

      {loading && (
        <div className="mb-4 rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-500">
          Searching semantic + keyword indexes...
        </div>
      )}

      {searched && !loading && results.length === 0 && (
        <div className="mt-8 flex flex-col items-center gap-2 rounded-2xl border border-gray-200 bg-white p-6 text-gray-400">
          <Search size={36} strokeWidth={1.5} />
          <p className="text-sm font-medium text-gray-700">No matching articles found.</p>
          <p className="text-xs">Try broader terms, set approval to All, or approve pending KBs from the Dashboard review workspace.</p>
        </div>
      )}

      <div className="grid gap-4">
        {results.map((result, index) => (
          <ResultCard key={result?.kb?.kb_id || result?.kb?.title || index} result={result} queryTerms={queryTerms} isFirst={index === 0} />
        ))}
      </div>
    </section>
  );
}
