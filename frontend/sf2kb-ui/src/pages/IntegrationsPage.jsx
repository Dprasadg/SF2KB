import { useState } from "react";
import api from "../api";

export default function IntegrationsPage() {
  const [sfConfig, setSfConfig] = useState({
    baseUrl: "https://login.salesforce.com",
    username: "",
    password: "",
    securityToken: "",
  });
  const [smarshConfig, setSmarshConfig] = useState({
    baseUrl: "https://docs.smarsh.com",
    username: "",
    password: "",
  });
  const [integrationLoading, setIntegrationLoading] = useState({
    salesforce: false,
    smarsh: false,
  });
  const [integrationMessage, setIntegrationMessage] = useState({
    salesforce: "",
    smarsh: "",
  });
  const [integrationError, setIntegrationError] = useState({
    salesforce: "",
    smarsh: "",
  });

  const validateConfig = (config, options = { requireToken: false }) => {
    if (!config.baseUrl?.trim() || !config.baseUrl.startsWith("https://")) {
      return "Base URL must start with https://";
    }
    if (!config.username?.trim()) {
      return "Username is required.";
    }
    if (!config.password?.trim()) {
      return "Password is required.";
    }
    if (options.requireToken && !config.securityToken?.trim()) {
      return "Salesforce security token is required.";
    }
    return "";
  };

  const connectSalesforce = async () => {
    const validationError = validateConfig(sfConfig, { requireToken: true });
    if (validationError) {
      setIntegrationError((prev) => ({ ...prev, salesforce: validationError }));
      setIntegrationMessage((prev) => ({ ...prev, salesforce: "" }));
      return;
    }

    setIntegrationLoading((prev) => ({ ...prev, salesforce: true }));
    setIntegrationError((prev) => ({ ...prev, salesforce: "" }));
    setIntegrationMessage((prev) => ({ ...prev, salesforce: "" }));

    try {
      const response = await api.post("/integrations/salesforce/connect", sfConfig);
      setIntegrationMessage((prev) => ({
        ...prev,
        salesforce: response?.data?.message || "Salesforce credentials validated successfully.",
      }));
    } catch (requestError) {
      if (requestError?.response?.status === 404) {
        setIntegrationError((prev) => ({
          ...prev,
          salesforce:
            "Backend endpoint /integrations/salesforce/connect is not available yet. UI is ready for integration.",
        }));
      } else {
        setIntegrationError((prev) => ({
          ...prev,
          salesforce: requestError.response?.data?.detail || "Salesforce connection failed.",
        }));
      }
    } finally {
      setIntegrationLoading((prev) => ({ ...prev, salesforce: false }));
    }
  };

  const connectSmarshDocs = async () => {
    const validationError = validateConfig(smarshConfig);
    if (validationError) {
      setIntegrationError((prev) => ({ ...prev, smarsh: validationError }));
      setIntegrationMessage((prev) => ({ ...prev, smarsh: "" }));
      return;
    }

    setIntegrationLoading((prev) => ({ ...prev, smarsh: true }));
    setIntegrationError((prev) => ({ ...prev, smarsh: "" }));
    setIntegrationMessage((prev) => ({ ...prev, smarsh: "" }));

    try {
      const response = await api.post("/integrations/smarsh/connect", smarshConfig);
      setIntegrationMessage((prev) => ({
        ...prev,
        smarsh: response?.data?.message || "docs.smarsh.com credentials validated successfully.",
      }));
    } catch (requestError) {
      if (requestError?.response?.status === 404) {
        setIntegrationError((prev) => ({
          ...prev,
          smarsh:
            "Backend endpoint /integrations/smarsh/connect is not available yet. UI is ready for integration.",
        }));
      } else {
        setIntegrationError((prev) => ({
          ...prev,
          smarsh: requestError.response?.data?.detail || "docs.smarsh.com connection failed.",
        }));
      }
    } finally {
      setIntegrationLoading((prev) => ({ ...prev, smarsh: false }));
    }
  };

  return (
    <section className="w-full py-2 lg:p-8">
      <h2 className="text-3xl font-semibold">Integrations</h2>
      <p className="mt-2 max-w-3xl text-sm text-gray-500">
        Configure Salesforce and docs.smarsh.com credentials for future live integrations.
      </p>

      <div className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-gray-200 p-4">
            <h4 className="font-semibold">Salesforce</h4>

            <label className="mt-4 block text-xs font-medium uppercase tracking-wide text-gray-600">Base URL</label>
            <input
              type="text"
              value={sfConfig.baseUrl}
              onChange={(event) => setSfConfig((prev) => ({ ...prev, baseUrl: event.target.value }))}
              placeholder="https://login.salesforce.com"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-gray-600">Username</label>
            <input
              type="text"
              value={sfConfig.username}
              onChange={(event) => setSfConfig((prev) => ({ ...prev, username: event.target.value }))}
              placeholder="name@company.com"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-gray-600">Password</label>
            <input
              type="password"
              value={sfConfig.password}
              onChange={(event) => setSfConfig((prev) => ({ ...prev, password: event.target.value }))}
              placeholder="Enter Salesforce password"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-gray-600">
              Security Token
            </label>
            <input
              type="password"
              value={sfConfig.securityToken}
              onChange={(event) => setSfConfig((prev) => ({ ...prev, securityToken: event.target.value }))}
              placeholder="Enter Salesforce token"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <button
              type="button"
              onClick={connectSalesforce}
              disabled={integrationLoading.salesforce}
              className="mt-4 rounded-lg bg-black px-4 py-2 text-sm font-semibold text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-400"
            >
              {integrationLoading.salesforce ? "Connecting..." : "Connect Salesforce"}
            </button>

            {integrationError.salesforce && <p className="mt-3 text-sm text-red-700">{integrationError.salesforce}</p>}
            {integrationMessage.salesforce && (
              <p className="mt-3 text-sm text-green-700">{integrationMessage.salesforce}</p>
            )}
          </div>

          <div className="rounded-xl border border-gray-200 p-4">
            <h4 className="font-semibold">docs.smarsh.com</h4>

            <label className="mt-4 block text-xs font-medium uppercase tracking-wide text-gray-600">Base URL</label>
            <input
              type="text"
              value={smarshConfig.baseUrl}
              onChange={(event) => setSmarshConfig((prev) => ({ ...prev, baseUrl: event.target.value }))}
              placeholder="https://docs.smarsh.com"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-gray-600">Username</label>
            <input
              type="text"
              value={smarshConfig.username}
              onChange={(event) => setSmarshConfig((prev) => ({ ...prev, username: event.target.value }))}
              placeholder="Enter docs username"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-gray-600">Password</label>
            <input
              type="password"
              value={smarshConfig.password}
              onChange={(event) => setSmarshConfig((prev) => ({ ...prev, password: event.target.value }))}
              placeholder="Enter docs password"
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />

            <button
              type="button"
              onClick={connectSmarshDocs}
              disabled={integrationLoading.smarsh}
              className="mt-4 rounded-lg bg-black px-4 py-2 text-sm font-semibold text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-400"
            >
              {integrationLoading.smarsh ? "Connecting..." : "Connect docs.smarsh.com"}
            </button>

            {integrationError.smarsh && <p className="mt-3 text-sm text-red-700">{integrationError.smarsh}</p>}
            {integrationMessage.smarsh && <p className="mt-3 text-sm text-green-700">{integrationMessage.smarsh}</p>}
          </div>
        </div>
      </div>
    </section>
  );
}