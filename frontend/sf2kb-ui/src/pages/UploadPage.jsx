import { useState } from "react";
import api from "../api";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    if (selected && !selected.name.toLowerCase().endsWith(".csv")) {
      setError("Only CSV files are supported.");
      setFile(null);
      return;
    }
    setError(null);
    setResult(null);
    setFile(selected || null);
  };

  const uploadFile = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post("/process-cases", formData);
      setResult(response.data.result);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || "Processing failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="w-full py-2 lg:p-8">
      <h2 className="text-3xl font-semibold">File Upload</h2>
      <p className="mt-2 max-w-3xl text-sm text-gray-500">
        Upload your Salesforce case CSV file to run ingestion, deduplication, clustering, and KB generation.
      </p>

      <div className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
        <label className="block text-sm font-medium text-gray-700">CSV File</label>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="mt-3 w-full rounded-xl border border-gray-300 bg-white px-3 py-2 text-sm text-gray-700 file:mr-4 file:rounded-lg file:border-0 file:bg-black file:px-3 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-gray-800"
        />

        <button
          type="button"
          onClick={uploadFile}
          disabled={loading || !file}
          className="mt-5 rounded-xl bg-black px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-400"
        >
          {loading ? "Processing..." : "Run Pipeline"}
        </button>

        {error && (
          <p className="mt-4 text-sm text-red-700">{error}</p>
        )}
      </div>

      {result && (
        <div className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
          <h3 className="text-xl font-semibold">Pipeline Summary</h3>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <Metric label="Loaded" value={result.loaded} color="bg-green-500" />
            <Metric label="Clusters" value={result.clusters} color="bg-blue-500" />
            <Metric label="Created" value={result.created} color="bg-black" />
            <Metric label="Skipped" value={result.skipped} color="bg-amber-400" />
            <Metric label="Failed" value={result.failed} color="bg-red-500" />
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value, color }) {
  return (
    <article className="rounded-xl border border-gray-200 bg-white p-4">
      <p className="text-xs uppercase tracking-wider text-gray-500">{label}</p>
      <div className="mt-2 flex items-end gap-2">
        <p className="text-3xl font-bold">{value ?? 0}</p>
        <span className={`mb-1 h-2.5 w-2.5 rounded-full ${color}`} />
      </div>
    </article>
  );
}