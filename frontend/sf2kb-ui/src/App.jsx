import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import TopNav from "./components/TopNav";
import UploadPage from "./pages/UploadPage";
import IntegrationsPage from "./pages/IntegrationsPage";
import SearchPage from "./pages/SearchPage";
import DashboardPage from "./pages/DashboardPage";

function App() {
  const [view, setView] = useState("search");

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="page-shell">
        <Sidebar setView={setView} currentView={view} />

        <main className="min-h-screen flex-1 bg-gray-50 px-4 py-6 sm:px-6 lg:px-10">
          <TopNav currentView={view} setView={setView} />
          {view === "upload" && <UploadPage />}
          {view === "integrations" && <IntegrationsPage />}
          {view === "search" && <SearchPage />}
          {view === "dashboard" && <DashboardPage />}
        </main>
      </div>
    </div>
  );
}

export default App;