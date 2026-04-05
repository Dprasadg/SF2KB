const navItems = [
  { key: "search", label: "Search" },
  { key: "dashboard", label: "Dashboard" },
  { key: "upload", label: "File Upload" },
  { key: "integrations", label: "Integrations" },
];

export default function TopNav({ currentView, setView }) {
  return (
    <div className="mb-6 flex items-center justify-between rounded-xl border border-gray-200 bg-white p-3 shadow-sm lg:hidden">
      <p className="text-sm font-semibold tracking-wide">KB Intelligence</p>
      <div className="flex gap-2">
        {navItems.map((item) => (
          <button
            key={item.key}
            type="button"
            onClick={() => setView(item.key)}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
              currentView === item.key
                ? "bg-gray-900 text-white"
                : "bg-gray-100 text-gray-700"
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}