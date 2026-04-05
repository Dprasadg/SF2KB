import { useEffect, useState } from "react";
import api from "../api";

export default function DashboardPage() {
  const [data, setData] = useState({ total_kb: 0, titles: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDashboard = () => {
    setLoading(true);
    setError(null);
    api
      .get("/dashboard")
      .then((response) => setData(response.data))
      .catch(() => setError("Failed to load dashboard data."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  return (
    <section className="w-full py-2 lg:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-3xl font-semibold">Dashboard</h2>
          <p className="mt-2 text-sm text-gray-500">Quick snapshot of generated KB inventory.</p>
        </div>
        <button
          type="button"
          onClick={fetchDashboard}
          className="rounded-xl bg-black px-4 py-2 text-sm font-semibold text-white transition hover:bg-gray-800"
        >
          Refresh
        </button>
      </div>

      {loading && <p className="mt-6 text-sm text-gray-500">Loading dashboard...</p>}
      {error && <p className="mt-6 text-sm text-red-700">{error}</p>}

      {!loading && !error && (
        <>
          <div className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
            <p className="text-xs uppercase tracking-[0.18em] text-gray-500">Total Articles</p>
            <p className="mt-2 text-5xl font-bold">{data.total_kb || 0}</p>
          </div>

          <div className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold">Published Titles</h3>
            {Array.isArray(data.titles) && data.titles.length > 0 ? (
              <ul className="mt-4 space-y-2">
                {data.titles.map((title, index) => (
                  <li key={index} className="rounded-xl border border-gray-200 bg-white p-3 text-sm text-gray-700">
                    {title || "Untitled"}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-gray-500">No KB articles yet. Run a CSV upload to generate content.</p>
            )}
          </div>
        </>
      )}
    </section>
  );
}