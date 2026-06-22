import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { getHealth } from "../api";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/train", label: "Train" },
  { to: "/predict", label: "Predict" },
];

const STATUS = {
  ready:      { dot: "bg-emerald-400", label: "Model ready" },
  "no-model": { dot: "bg-amber-400",   label: "No model trained" },
  unknown:    { dot: "bg-slate-500",   label: "API unreachable" },
};

export default function Layout() {
  const [modelStatus, setModelStatus] = useState("unknown");

  useEffect(() => {
    getHealth()
      .then((r) => setModelStatus(r.data.model_available ? "ready" : "no-model"))
      .catch(() => setModelStatus("unknown"));
  }, []);

  const { dot, label } = STATUS[modelStatus];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              Customer Churn Prediction
            </h1>
            <p className="text-xs text-slate-400">
              Multi-Agent System &middot; LangGraph &middot; MLflow
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400">
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />
              {label}
            </div>
            <nav className="flex gap-1">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-indigo-500 text-white"
                        : "text-slate-300 hover:bg-slate-800 hover:text-white"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
