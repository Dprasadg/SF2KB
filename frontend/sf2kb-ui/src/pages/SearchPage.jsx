import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import api from "../api";

function ResultCard({ result }) {
  const [open, setOpen] = useState(false);
  const kb = result?.kb || result;
  const title = kb?.title || "Untitled article";
  const summary = kb?.problem_summary || kb?.summary;
  const symptoms = Array.isArray(kb?.symptoms) ? kb.symptoms : [];
  const rootCause = kb?.root_cause || kb?.cause;
  const resolutionSteps = Array.isArray(kb?.resolution_steps)
    ? kb.resolution_steps
    : Array.isArray(kb?.resolution)
      ? kb.resolution
      : [];
  const keywords = Array.isArray(kb?.tags?.keywords)
    ? kb.tags.keywords
    : Array.isArray(kb?.keywords)
      ? kb.keywords
      : [];
  const confidence = kb?.confidence;

  return (
    <article className="rounded-2xl border bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-lg">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">{title}</h3>
          {summary && <p className="mt-1 text-gray-600">{summary}</p>}
        </div>

        <button type="button" onClick={() => setOpen(!open)} className="rounded-md p-1 hover:bg-gray-100">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {open && (
        <div className="mt-4 space-y-3">
          {symptoms.length > 0 && (
            <section>
              <p className="text-sm font-medium">Symptoms</p>
              <ul className="list-inside list-disc text-sm text-gray-600">
                {symptoms.map((symptom, index) => (
                  <li key={index}>{typeof symptom === "string" ? symptom : symptom?.name || ""}</li>
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

          {keywords.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {keywords.map((tag, index) => (
                <span key={index} className="rounded-full bg-gray-200 px-2 py-1 text-xs">
                  {tag}
                </span>
              ))}
            </div>
          )}

          {confidence && <p className="text-xs text-green-600">Confidence: {confidence}</p>}
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

  const search = async () => {
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.post("/search", { query });
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

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      search();
    }
  };

  return (
    <section className="w-full py-2 lg:p-8">
      <h2 className="mb-6 text-3xl font-semibold">Search Knowledge Base</h2>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search your knowledge base..."
          className="flex-1 rounded-xl border border-gray-300 p-3 focus:outline-none focus:ring-2 focus:ring-black"
        />
        <button
          type="button"
          onClick={search}
          disabled={loading || !query.trim()}
          className="rounded-xl bg-black px-6 py-3 text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-400"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <p className="mb-4 text-sm text-red-700">{error}</p>}
      {searched && !loading && results.length === 0 && (
        <p className="mb-4 text-sm text-gray-500">No matching articles found.</p>
      )}

      <div className="grid gap-4">
        {results.map((result, index) => (
          <ResultCard key={index} result={result} />
        ))}
      </div>
    </section>
  );
}