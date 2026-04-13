import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, CheckCircle, AlertCircle, FileText } from "lucide-react";
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

const SORT_OPTIONS = [
  { value: "priority", label: "Priority" },
  { value: "confidence", label: "Confidence" },
  { value: "date", label: "Date" },
  { value: "title", label: "Title" },
];

const PRIORITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };
const APPLIES_TO_OPTIONS = ["All Products", "Digital Safe", "Enterprise Archive"];

function toList(value) {
  if (Array.isArray(value)) return value.filter((item) => String(item || "").trim());
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

// ─── Toast notification ───────────────────────────────────────────────────────

function Toast({ message, type, onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, 3500);
    return () => clearTimeout(t);
  }, [onClose]);

  const base = "fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-2xl px-5 py-3 shadow-xl text-sm font-medium transition-all";
  const style = type === "success"
    ? `${base} bg-emerald-600 text-white`
    : `${base} bg-red-600 text-white`;

  return (
    <div className={style} role="alert">
      {type === "success" ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
      {message}
    </div>
  );
}

// ─── Loading skeletons ────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-2xl border bg-white p-5 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="flex-1 space-y-2">
          <div className="flex gap-2">
            <div className="h-4 w-16 rounded bg-gray-200" />
            <div className="h-4 w-12 rounded bg-gray-200" />
          </div>
          <div className="h-5 w-3/4 rounded bg-gray-200" />
          <div className="h-3 w-full rounded bg-gray-100" />
          <div className="h-3 w-2/3 rounded bg-gray-100" />
        </div>
        <div className="h-7 w-7 rounded bg-gray-200" />
      </div>
    </div>
  );
}

// ─── KB Card ──────────────────────────────────────────────────────────────────

function KBCard({ kb, onApprove, onUnapprove, onSaveEdit, approving, unapproving, saving }) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({
    title: "",
    summary: "",
    objective: "",
    answer: "",
    symptoms: "",
    applies_to: "",
    steps: "",
    resolution: "",
    cause: "",
    additional_info: "",
    keyword_variations: "",
  });

  const title = kb?.title || "Untitled article";
  const templateType = kb?.template_type || "solution";
  const templateMeta = TEMPLATE_LABELS[templateType] || TEMPLATE_LABELS.solution;
  const priority = (kb?.priority || "").toLowerCase();
  const priorityMeta = PRIORITY_LABELS[priority];
  const confidenceScore = kb?.confidence_score != null ? Math.round(kb.confidence_score * 100) : null;
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

  const startEditing = () => {
    setOpen(true);
    setEditing(true);
    setForm({
      title: title || "",
      summary: summary || "",
      objective: objective || "",
      answer: answer || "",
      symptoms: symptoms.join("\n"),
      applies_to: appliesTo.join("\n"),
      steps: steps.join("\n"),
      resolution: resolutionSteps.join("\n"),
      cause: rootCause || "",
      additional_info: additionalInfo || "",
      keyword_variations: toList(kb?.keyword_variations).join("\n"),
    });
  };

  const payloadFromForm = () => ({
    title: String(form.title || "").trim(),
    summary: String(form.summary || "").trim(),
    objective: String(form.objective || "").trim(),
    answer: String(form.answer || "").trim(),
    symptoms: toList(String(form.symptoms || "").split("\n")),
    applies_to: toList(String(form.applies_to || "").split("\n")),
    steps: toList(String(form.steps || "").split("\n")),
    resolution: toList(String(form.resolution || "").split("\n")),
    cause: String(form.cause || "").trim(),
    additional_info: String(form.additional_info || "").trim(),
    keyword_variations: toList(String(form.keyword_variations || "").split("\n")),
  });

  return (
    <article className="rounded-2xl border bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded px-2 py-0.5 text-xs font-semibold ${templateMeta.color}`}>
              {templateMeta.label}
            </span>
            {priorityMeta && (
              <span className={`rounded px-2 py-0.5 text-xs font-semibold ${priorityMeta.color}`}>
                {priorityMeta.label}
              </span>
            )}
            {confidenceScore !== null && (
              <span className="rounded bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700">
                Confidence {confidenceScore}%
              </span>
            )}
            {visibility && (
              <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">{visibility}</span>
            )}
            {validationState && (
              <span className={`rounded px-2 py-0.5 text-xs ${isApproved ? "bg-emerald-50 text-emerald-700" : "bg-blue-50 text-blue-700"}`}>
                {validationState}
              </span>
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

      {/* Approval action strip */}
      {!isApproved ? (
        <div className="mt-3 flex items-center justify-between rounded-xl border border-amber-200 bg-amber-50 px-3 py-2">
          <p className="text-xs font-medium text-amber-800">Pending approval</p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={startEditing}
              className="rounded-lg border border-gray-300 bg-white px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-50"
            >
              Edit
            </button>
            <button
              type="button"
              onClick={() => onApprove?.(kb?.kb_id)}
              disabled={approving || !kb?.kb_id}
              className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-gray-400"
            >
              {approving ? "Approving…" : "Approve"}
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-3 flex justify-end gap-2">
          <button
            type="button"
            onClick={startEditing}
            className="rounded-lg border border-gray-300 bg-white px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-50"
          >
            Edit
          </button>
          <button
            type="button"
            onClick={() => onUnapprove?.(kb?.kb_id)}
            disabled={unapproving || !kb?.kb_id}
            className="rounded-lg border border-gray-300 px-3 py-1 text-xs text-gray-500 transition hover:border-red-300 hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed"
          >
            {unapproving ? "Revoking…" : "Revoke Approval"}
          </button>
        </div>
      )}

      {open && (
        <div className="mt-4 space-y-4 border-t pt-4">
          {editing && (
            <section className="space-y-3 rounded-xl border border-blue-200 bg-blue-50 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">Edit KB</p>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="text-xs font-semibold text-gray-600">
                  Title
                  <input
                    value={form.title}
                    onChange={(e) => setForm((prev) => ({ ...prev, title: e.target.value }))}
                    className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
                <label className="text-xs font-semibold text-gray-600">
                  {templateType === "how_to" ? "Objective" : templateType === "qa" ? "Answer" : "Summary"}
                  <input
                    value={templateType === "how_to" ? form.objective : templateType === "qa" ? form.answer : form.summary}
                    onChange={(e) => {
                      const value = e.target.value;
                      if (templateType === "how_to") {
                        setForm((prev) => ({ ...prev, objective: value }));
                      } else if (templateType === "qa") {
                        setForm((prev) => ({ ...prev, answer: value }));
                      } else {
                        setForm((prev) => ({ ...prev, summary: value }));
                      }
                    }}
                    className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
              </div>

              {templateType === "solution" && (
                <label className="block text-xs font-semibold text-gray-600">
                  Symptoms (one per line)
                  <textarea
                    value={form.symptoms}
                    onChange={(e) => setForm((prev) => ({ ...prev, symptoms: e.target.value }))}
                    rows={3}
                    className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
              )}

              <label className="block text-xs font-semibold text-gray-600">
                Applies To (one per line)
                <textarea
                  value={form.applies_to}
                  onChange={(e) => setForm((prev) => ({ ...prev, applies_to: e.target.value }))}
                  rows={2}
                  className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                />
              </label>

              {templateType === "how_to" ? (
                <label className="block text-xs font-semibold text-gray-600">
                  Steps (one step per line)
                  <textarea
                    value={form.steps}
                    onChange={(e) => setForm((prev) => ({ ...prev, steps: e.target.value }))}
                    rows={4}
                    className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
              ) : templateType === "solution" ? (
                <label className="block text-xs font-semibold text-gray-600">
                  Resolution (one step per line)
                  <textarea
                    value={form.resolution}
                    onChange={(e) => setForm((prev) => ({ ...prev, resolution: e.target.value }))}
                    rows={4}
                    className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
              ) : null}

              {templateType === "solution" && (
                <label className="block text-xs font-semibold text-gray-600">
                  Cause
                  <textarea
                    value={form.cause}
                    onChange={(e) => setForm((prev) => ({ ...prev, cause: e.target.value }))}
                    rows={2}
                    className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                  />
                </label>
              )}

              <label className="block text-xs font-semibold text-gray-600">
                Additional Information
                <textarea
                  value={form.additional_info}
                  onChange={(e) => setForm((prev) => ({ ...prev, additional_info: e.target.value }))}
                  rows={2}
                  className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                />
              </label>

              <label className="block text-xs font-semibold text-gray-600">
                Keyword Variations (one per line)
                <textarea
                  value={form.keyword_variations}
                  onChange={(e) => setForm((prev) => ({ ...prev, keyword_variations: e.target.value }))}
                  rows={2}
                  className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                />
              </label>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setEditing(false)}
                  className="rounded-lg border border-gray-300 bg-white px-3 py-1 text-xs font-semibold text-gray-700 transition hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    const ok = await onSaveEdit?.(kb?.kb_id, payloadFromForm(), false);
                    if (ok) setEditing(false);
                  }}
                  disabled={saving || !kb?.kb_id}
                  className="rounded-lg bg-gray-900 px-3 py-1 text-xs font-semibold text-white transition hover:bg-black disabled:cursor-not-allowed disabled:bg-gray-400"
                >
                  {saving ? "Saving…" : "Save"}
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    const ok = await onSaveEdit?.(kb?.kb_id, payloadFromForm(), true);
                    if (ok) setEditing(false);
                  }}
                  disabled={saving || !kb?.kb_id}
                  className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-gray-400"
                >
                  {saving ? "Saving…" : "Save & Approve"}
                </button>
              </div>
            </section>
          )}

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
  const [appliesToFilter, setAppliesToFilter] = useState("All Products");
  const [sortBy, setSortBy] = useState("priority");
  const [actioningKbId, setActioningKbId] = useState(null);  // covers both approve + unapprove
  const [savingEditKbId, setSavingEditKbId] = useState(null);
  const [toast, setToast] = useState(null);  // { message, type }

  const showToast = (message, type = "success") => setToast({ message, type });
  const clearToast = () => setToast(null);

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
    return { approved, pending: articles.length - approved };
  }, [articles]);

  const approvedArticles = useMemo(
    () => articles.filter((kb) => kb?.validation_state === "Validated"),
    [articles],
  );

  const pendingArticles = useMemo(
    () => articles.filter((kb) => kb?.validation_state !== "Validated"),
    [articles],
  );

  // Apply product + template filters
  const filteredApproved = useMemo(() => {
    let list = approvedArticles;

    if (appliesToFilter !== "All Products") {
      list = list.filter((kb) =>
        toList(kb.applies_to).some((a) =>
          a.toLowerCase().includes(appliesToFilter.toLowerCase()),
        ),
      );
    }

    if (filter !== "all") {
      list = list.filter((kb) => (kb.template_type || "solution") === filter);
    }

    return list;
  }, [approvedArticles, filter, appliesToFilter]);

  // Sort approved articles
  const sortedFiltered = useMemo(() => {
    const sorted = [...filteredApproved];
    if (sortBy === "priority") {
      sorted.sort((a, b) =>
        (PRIORITY_ORDER[a.priority?.toLowerCase()] ?? 9) -
        (PRIORITY_ORDER[b.priority?.toLowerCase()] ?? 9)
      );
    } else if (sortBy === "confidence") {
      sorted.sort((a, b) => (b.confidence_score ?? 0) - (a.confidence_score ?? 0));
    } else if (sortBy === "date") {
      sorted.sort((a, b) =>
        (b.approved_at || b.created_at || "").localeCompare(a.approved_at || a.created_at || ""),
      );
    } else if (sortBy === "title") {
      sorted.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
    }
    return sorted;
  }, [filteredApproved, sortBy]);

  const setApproval = async (kbId, approved) => {
    if (!kbId || actioningKbId) return;
    setActioningKbId(kbId);
    try {
      await api.post(`/kb/${kbId}/approval`, { approved });
      showToast(approved ? "KB article approved." : "Approval revoked.");
      await fetchArticles();
    } catch {
      showToast(approved ? "Failed to approve KB article." : "Failed to revoke approval.", "error");
    } finally {
      setActioningKbId(null);
    }
  };

  const bulkApproveAll = async () => {
    if (!pendingArticles.length || actioningKbId) return;
    for (const kb of pendingArticles) {
      if (kb.kb_id) {
        try {
          await api.post(`/kb/${kb.kb_id}/approval`, { approved: true });
        } catch {
          // continue approving others even if one fails
        }
      }
    }
    showToast(`${pendingArticles.length} KB article${pendingArticles.length !== 1 ? "s" : ""} approved.`);
    await fetchArticles();
  };

  const saveKbEdit = async (kbId, updates, approve = false) => {
    if (!kbId || savingEditKbId) return false;
    setSavingEditKbId(kbId);
    try {
      const response = await api.patch(`/kb/${kbId}`, { ...(updates || {}), approve });
      const issues = toList(response?.data?.validation_issues);

      if (approve && issues.length > 0) {
        showToast("Saved, but cannot approve until validation issues are fixed.", "error");
        return false;
      } else {
        showToast(approve ? "KB article updated and approved." : "KB article updated.");
      }
      await fetchArticles();
      return true;
    } catch {
      showToast("Failed to save KB edits.", "error");
      return false;
    } finally {
      setSavingEditKbId(null);
    }
  };

  return (
    <section className="w-full py-2 lg:p-8">
      {toast && <Toast message={toast.message} type={toast.type} onClose={clearToast} />}

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

      {error && <p className="mt-6 text-sm text-red-700">{error}</p>}

      {loading ? (
        <>
          {/* Stats skeleton */}
          <div className="mt-6 grid grid-cols-3 gap-4 sm:grid-cols-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="animate-pulse rounded-2xl border bg-white p-4 shadow-sm">
                <div className="h-3 w-12 rounded bg-gray-200" />
                <div className="mt-2 h-8 w-8 rounded bg-gray-200" />
              </div>
            ))}
          </div>
          {/* Card skeletons */}
          <div className="mt-6 grid gap-4">
            {Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        </>
      ) : (
        <>
          {/* Stats row — 6-column grid */}
          <div className="mt-6 grid grid-cols-3 gap-4 sm:grid-cols-6">
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
                <p className={`mt-1 text-3xl font-bold ${stat.color}`}>{stat.value}</p>
              </div>
            ))}
          </div>

          {/* Pending Approval section */}
          {pendingArticles.length > 0 && (
            <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-lg font-semibold text-amber-900">Pending Approval</h3>
                  <p className="mt-0.5 text-sm text-amber-800">
                    Approved KBs move to the main list and are available in search.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={bulkApproveAll}
                  disabled={!!actioningKbId}
                  className="rounded-xl bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-gray-400"
                >
                  Approve All ({pendingArticles.length})
                </button>
              </div>
              <div className="mt-3 grid gap-3">
                {pendingArticles.map((kb) => (
                  <KBCard
                    key={`pending-${kb.kb_id || kb.title}`}
                    kb={kb}
                    onApprove={(id) => setApproval(id, true)}
                    onUnapprove={(id) => setApproval(id, false)}
                    onSaveEdit={saveKbEdit}
                    approving={actioningKbId === kb.kb_id}
                    unapproving={false}
                    saving={savingEditKbId === kb.kb_id}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Product + Template filter bar */}
          <div className="mt-6 flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">Product</span>
              <div className="flex gap-1">
                {APPLIES_TO_OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => setAppliesToFilter(opt)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                      appliesToFilter === opt
                        ? "bg-black text-white"
                        : "border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">Type</span>
              <div className="flex gap-1">
                {FILTER_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setFilter(opt.value)}
                    className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${
                      filter === opt.value
                        ? "bg-black text-white"
                        : "border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sort control */}
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">Sort</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="rounded-lg border border-gray-300 bg-white px-2 py-1 text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-black"
              >
                {SORT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Article cards */}
          <div className="mt-4 grid gap-4">
            {sortedFiltered.length > 0 ? (
              sortedFiltered.map((kb) => (
                <KBCard
                  key={kb.kb_id || kb.title}
                  kb={kb}
                  onApprove={(id) => setApproval(id, true)}
                  onUnapprove={(id) => setApproval(id, false)}
                  onSaveEdit={saveKbEdit}
                  approving={false}
                  unapproving={actioningKbId === kb.kb_id}
                  saving={savingEditKbId === kb.kb_id}
                />
              ))
            ) : (
              <div className="mt-10 flex flex-col items-center gap-3 text-gray-400">
                <FileText size={40} strokeWidth={1.2} />
                <p className="text-sm font-medium">
                  {articles.length === 0
                    ? "No KB articles yet."
                    : "No approved articles match the selected filters."}
                </p>
                {articles.length === 0 && (
                  <p className="text-xs">Run a CSV upload from the Upload page to get started.</p>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
