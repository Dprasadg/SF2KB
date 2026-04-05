import { Database, LayoutDashboard, Search } from "lucide-react";

const navItems = [
  { key: "search", label: "Search KB", icon: Search },
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "upload", label: "File Upload", icon: Database },
  { key: "integrations", label: "Integrations", icon: Database },
];

export default function Sidebar({ setView, currentView }) {
  return (
    <aside className="hidden h-screen w-64 bg-gradient-to-b from-gray-900 to-gray-800 p-5 text-white shadow-xl lg:block">
      <h1 className="mb-8 text-2xl font-bold tracking-tight">KB Intelligence</h1>

      <nav>
        {navItems.map((item) => {
          const active = currentView === item.key;
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setView(item.key)}
              className={`mb-2 flex w-full cursor-pointer items-center gap-3 rounded-xl p-3 text-left transition-all duration-200 ${
                active ? "bg-gray-700" : "hover:bg-gray-800"
              }`}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}