import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import api from "../api";

const TEMPLATE_LABELS = {
  solution: { label: "Solution", color: "bg-purple-50 text-purple-700" },
  how_to: { label: "How To", color: "bg-green-50 text-green-700" },
  qa: { label: "Q&A", color: "bg-sky-50 text-sky-700" },
};

function toList(value) {
  if (Array.isArray(value)) return value.filter((item) => String(item || "").trim());
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function KBCard({ kb, onApprove, approving }) {
  const [open, setOpen] = useState(false);

  const title = kb?.title || "Untitled article";
  const templateType = kb?.template_type || "solution";
  const templateMeta = TEMPLATE_LABELS[templateType] || TEMPLATE_LABELS.solution;
  const summary = kb?.summary || kb?.problem_summary || "";
  const objective = kb?.objective || "";
  const answer = kb?.answer || "";
  const symptoms = toList(kb?.symptoms);
  const appliesTo = toList(kb?.applies_to);
  const rootCause = kb?.cause || kb?.root_cause || "";
  const resolutionSteps = toList(kb?.resolution_steps || kb?.resolution);
  const steps = toList(kb?.steps);
  const additionalInfo = kb?.additional_info || "";
  const visibility = kb?.visibility || "";
  const validationState = kb?.validation_state || "";
  const isApproved = validationState === "Validated";
  const keywords = useMemo(
    () => Array.from(new Set(toList(kb?.keyword_variations).concat(toList(kb?.keywords)))),
    [kb],
  );

  return (
    <article className="rounded-2xl border bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded px-2 py-0.5 text-xs font-semibold ${templateMeta.color}`}>
              {templateMeta.label}
            </span>
            {visibility && (
              <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">{visibility}</span>
            )}
            {validationState && (
              <span className="rounded bg-blue-50 px-2 py-0.5 text-xs text-blue-700">{validationState}</span>
            )}
          </div>
          <h3 className="mt-2 text-base font-semibold leading-snug">{title}</h3>
          {!open && (summary || objective || answer) && (
            <p className="mt-1 line-clamp-2 text-sm text-gray-500">{summary || objective || answer}</p>
          )}
        </div>
        <button type="button" onClick={() => setOpen(!open)} className="shrink-0 rounded-md p-1 hover:bg-gray-100">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {!isApproved && (
        <div className="mt-3 flex items-center justify-between rounded-xl border border-amber-200 bg-amber-50 px-3 py-2">
          <p className="text-xs font-medium text-amber-800">Pending approval</p>
          <button
            type="button"
            onClick={() => onApprove?.(kb?.kb_id)}
            disabled={approving || !kb?.kb_id}
            className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-gray-400"
          >
            {approving ? "Approving..." : "Approve"}
          </button>
        </div>
      )}

      {open && (
        <div className="mt-4 space-y-4 border-t pt-4">
          {appliesTo.length > 0 && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Applies To</p>
              <div className="mt-1 flex flex-wrap gap-2">
                {appliesTo.map((item, i) => (
                  <span key={i} className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-700">{item}</span>
                ))}
              </div>
            </section>
          )}

          {(summary || objective) && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                {templateType === "how_to" ? "Objective" : "Summary"}
              </p>
              <p className="mt-1 text-sm text-gray-700">{objective || summary}</p>
            </section>
          )}

          {answer && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Answer</p>
              <p className="mt-1 text-sm text-gray-700">{answer}</p>
            </section>
          )}

          {symptoms.length > 0 && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Symptoms</p>
              <ul className="mt-1 list-inside list-disc text-sm text-gray-600">
                {symptoms.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </section>
          )}

          {rootCause && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Root Cause</p>
              <p className="mt-1 text-sm text-gray-700">{rootCause}</p>
            </section>
          )}

          {steps.length > 0 && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Steps</p>
              <ol className="mt-1 list-inside list-decimal text-sm text-gray-600">
                {steps.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
            </section>
          )}

          {resolutionSteps.length > 0 && templateType !== "how_to" && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Resolution Steps</p>
              <ol className="mt-1 list-inside list-decimal text-sm text-gray-600">
                {resolutionSteps.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
            </section>
          )}

          {additionalInfo && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Additional Information</p>
              <p className="mt-1 text-sm text-gray-700">{additionalInfo}</p>
            </section>
          )}

          {keywords.length > 0 && (
            <section>
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">Keywords</p>
              <div className="mt-1 flex flex-wrap gap-2">
                {keywords.map((tag, i) => (
                  <span key={i} className="rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-600">{tag}</span>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </article>
  );
}

const FILTER_OPTIONS = [
  { value: "all", label: "All" },
  { value: "solution", label: "Solution" },
  { value: "how_to", label: "How To" },
  { value: "qa", label: "Q&A" },
];

export default function DashboardPage() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all");
  const [approvingKbId, setApprovingKbId] = useState(null);

  const fetchArticles = () => {
    setLoading(true);
    setError(null);
    api
      .get("/kb")
      .then((response) => setArticles(response.data?.kb || []))
      .catch(() => setError("Failed to load KB articles."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchArticles();
  }, []);

  const counts = useMemo(() => {
    const c = { solution: 0, how_to: 0, qa: 0 };
    articles.forEach((kb) => {
      const t = kb.template_type || "solution";
      if (t in c) c[t]++;
    });
    return c;
  }, [articles]);

  const approvalCounts = useMemo(() => {
    const approved = articles.filter((kb) => kb?.validation_state === "Validated").length;
    const pending = articles.length - approved;
    return { approved, pending };
  }, [articles]);

  const approvedArticles = useMemo(
    () => articles.filter((kb) => kb?.validation_state === "Validated"),
    [articles],
  );

  const filtered = useMemo(
    () => filter === "all"
      ? approvedArticles
      : approvedArticles.filter((kb) => (kb.template_type || "solution") === filter),
    [approvedArticles, filter],
  );

  const pendingArticles = useMemo(
    () => articles.filter((kb) => kb?.validation_state !== "Validated"),
    [articles],
  );

  const approveKb = async (kbId) => {
    if (!kbId || approvingKbId) return;

    setApprovingKbId(kbId);
    setError(null);
    try {
      await api.post(`/kb/${kbId}/approval`, { approved: true });
      await fetchArticles();
    } catch {
      setError("Failed to approve KB article.");
    } finally {
      setApprovingKbId(null);
    }
  };

  return (
    <section className="w-full py-2 lg:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-3xl font-semibold">Dashboard</h2>
          <p className="mt-2 text-sm text-gray-500">Browse and review all generated KB articles.</p>
        </div>
        <button
          type="button"
          onClick={fetchArticles}
          className="rounded-xl bg-black px-4 py-2 text-sm font-semibold text-white transition hover:bg-gray-800"
        >
          Refresh
        </button>
      </div>

      {loading && <p className="mt-6 text-sm text-gray-500">Loading articles...</p>}
      {error && <p className="mt-6 text-sm text-red-700">{error}</p>}

      {!loading && !error && (
        <>
          {/* Stats row */}
          <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { label: "Total", value: articles.length, color: "text-black" },
              { label: "Approved", value: approvalCounts.approved, color: "text-emerald-700" },
              { label: "Pending", value: approvalCounts.pending, color: "text-amber-700" },
              { label: "Solution", value: counts.solution, color: "text-purple-700" },
              { label: "How To", value: counts.how_to, color: "text-green-700" },
              { label: "Q&A", value: counts.qa, color: "text-sky-700" },
            ].map((stat) => (
              <div key={stat.label} className="rounded-2xl border bg-white p-4 shadow-sm">
                <p className="text-xs uppercase tracking-widest text-gray-500">{stat.label}</p>
                <p className={`mt-1 text-4xl font-bold ${stat.color}`}>{stat.value}</p>
              </div>
            ))}
          </div>

          {pendingArticles.length > 0 && (
            <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-4">
              <h3 className="text-lg font-semibold text-amber-900">Pending Approval</h3>
              <p className="mt-1 text-sm text-amber-800">
                Review and approve newly generated KBs. Approved KBs move to the top and are available to others in search.
              </p>
              <div className="mt-3 grid gap-3">
                {pendingArticles.map((kb) => (
                  <KBCard
                    key={`pending-${kb.kb_id || kb.title}`}
                    kb={kb}
                    onApprove={approveKb}
                    approving={approvingKbId === kb.kb_id}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Filter tabs */}
          <div className="mt-6 flex flex-wrap gap-2">
            {FILTER_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setFilter(opt.value)}
                className={`rounded-full px-4 py-1.5 text-sm font-medium transition ${
                  filter === opt.value
                    ? "bg-black text-white"
                    : "border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Article cards */}
          <div className="mt-4 grid gap-4">
            {filtered.length > 0 ? (
              filtered.map((kb) => (
                <KBCard
                  key={kb.kb_id || kb.title}
                  kb={kb}
                  onApprove={approveKb}
                  approving={approvingKbId === kb.kb_id}
                />
              ))
            ) : (
              <p className="mt-4 text-sm text-gray-500">
                {articles.length === 0
                  ? "No KB articles yet. Run a CSV upload to generate content."
                  : "No approved articles match the selected filter."}
              </p>
            )}
          </div>
        </>
      )}
    </section>
  );
}