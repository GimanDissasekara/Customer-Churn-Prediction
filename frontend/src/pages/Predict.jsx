import { useMemo, useState } from "react";
import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { predictChurn } from "../api";
import Card from "../components/Card";
import ProbabilityBar from "../components/ProbabilityBar";
import RiskBadge from "../components/RiskBadge";
import Spinner from "../components/Spinner";
import { CHART_TOOLTIP_STYLE, RISK_COLORS } from "../lib/constants";

const PAGE_SIZE = 25;
const RISK_FILTERS = ["All", "High", "Medium", "Low"];

function downloadCSV(rows) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const lines = [
    keys.join(","),
    ...rows.map((r) => keys.map((k) => JSON.stringify(r[k] ?? "")).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  Object.assign(document.createElement("a"), {
    href: url,
    download: "churn_predictions.csv",
  }).click();
  URL.revokeObjectURL(url);
}

export default function Predict() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);

  const handlePredict = async () => {
    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setPage(0);
    try {
      const res = await predictChurn(file);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const pieData = useMemo(
    () =>
      Object.entries(result?.risk_level_counts || {}).map(([level, count]) => ({
        name: level,
        value: count,
      })),
    [result]
  );

  const filteredPredictions = useMemo(() => {
    if (!result) return [];
    let rows = result.predictions;
    if (filter !== "All") rows = rows.filter((p) => p.risk_level === filter);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((p) => String(p.customerID).toLowerCase().includes(q));
    }
    return [...rows].sort((a, b) =>
      sortDir === "desc"
        ? b.churn_probability - a.churn_probability
        : a.churn_probability - b.churn_probability
    );
  }, [result, filter, search, sortDir]);

  const pageCount = Math.max(1, Math.ceil(filteredPredictions.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = filteredPredictions.slice(
    safePage * PAGE_SIZE,
    (safePage + 1) * PAGE_SIZE
  );
  const hasActual = pageRows[0]?.actual_churn !== undefined;

  const setFilterAndReset = (v) => { setFilter(v); setPage(0); };
  const setSearchAndReset = (v) => { setSearch(v); setPage(0); };
  const toggleSort = () => { setSortDir((d) => (d === "desc" ? "asc" : "desc")); setPage(0); };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold">Predict customer churn</h2>
        <p className="text-slate-400 text-sm mt-1">
          Upload a CSV of customers in the Telco Customer Churn schema. The pipeline
          re-runs ingestion, cleaning, and feature engineering before scoring.
        </p>
      </div>

      <Card title="Customer data">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="text-sm text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-md
                       file:border-0 file:text-sm file:font-medium
                       file:bg-indigo-500 file:text-white hover:file:bg-indigo-400 cursor-pointer"
          />
          <button
            onClick={handlePredict}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-indigo-500
                       hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed
                       text-sm font-medium text-white transition-colors"
          >
            {loading && <Spinner size={16} />}
            {loading ? "Scoring…" : "Predict churn"}
          </button>
        </div>
      </Card>

      {error && (
        <Card title="Prediction failed">
          <p className="text-red-400 text-sm whitespace-pre-wrap">
            {Array.isArray(error) ? error.join("\n") : String(error)}
          </p>
        </Card>
      )}

      {result && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card title="Customers scored">
              <div className="text-3xl font-semibold text-indigo-400 text-center py-2">
                {result.n_customers.toLocaleString()}
              </div>
            </Card>
            <Card title="Avg. churn probability">
              <div className="text-3xl font-semibold text-indigo-400 text-center py-2">
                {(result.avg_churn_probability * 100).toFixed(1)}%
              </div>
            </Card>
            <Card title="Risk distribution">
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={28}
                      outerRadius={48}
                    >
                      {pieData.map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={RISK_COLORS[entry.name] || "#94a3b8"}
                        />
                      ))}
                    </Pie>
                    <Tooltip {...CHART_TOOLTIP_STYLE} />
                    <Legend iconSize={10} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Results table */}
          <Card title="Customer results">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              {RISK_FILTERS.map((level) => (
                <button
                  key={level}
                  onClick={() => setFilterAndReset(level)}
                  className={`px-3 py-1 rounded-md text-xs font-medium border transition-colors ${
                    filter === level
                      ? "bg-indigo-500 border-indigo-500 text-white"
                      : "border-slate-700 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {level}
                </button>
              ))}
              <input
                type="search"
                placeholder="Search customer ID…"
                value={search}
                onChange={(e) => setSearchAndReset(e.target.value)}
                className="ml-auto px-3 py-1 text-xs bg-slate-800 border border-slate-700
                           rounded-md text-slate-300 placeholder-slate-500
                           focus:outline-none focus:ring-1 focus:ring-indigo-500 w-44"
              />
              <button
                onClick={() => downloadCSV(filteredPredictions)}
                disabled={!filteredPredictions.length}
                className="px-3 py-1 text-xs border border-slate-700 text-slate-300
                           rounded-md hover:bg-slate-800 disabled:opacity-40 transition-colors"
              >
                Export CSV
              </button>
              <span className="text-xs text-slate-500 whitespace-nowrap">
                {filteredPredictions.length} customers
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-800">
                    <th className="py-2 pr-4 font-medium">Customer ID</th>
                    <th className="py-2 pr-4 font-medium">
                      <button
                        onClick={toggleSort}
                        className="flex items-center gap-1 hover:text-slate-200 transition-colors"
                      >
                        Churn probability
                        <span className="text-slate-500 text-xs">
                          {sortDir === "desc" ? "↓" : "↑"}
                        </span>
                      </button>
                    </th>
                    <th className="py-2 pr-4 font-medium">Risk level</th>
                    {hasActual && (
                      <th className="py-2 pr-4 font-medium">Actual churn</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {pageRows.length === 0 ? (
                    <tr>
                      <td
                        colSpan={hasActual ? 4 : 3}
                        className="py-10 text-center text-slate-500 text-sm"
                      >
                        No customers match the current filter.
                      </td>
                    </tr>
                  ) : (
                    pageRows.map((row) => (
                      <tr
                        key={row.customerID}
                        className="border-b border-slate-900 hover:bg-slate-900/50"
                      >
                        <td className="py-2.5 pr-4 font-mono text-xs text-slate-300">
                          {row.customerID}
                        </td>
                        <td className="py-2.5 pr-4">
                          <ProbabilityBar value={row.churn_probability} />
                        </td>
                        <td className="py-2.5 pr-4">
                          <RiskBadge level={row.risk_level} />
                        </td>
                        {hasActual && (
                          <td className="py-2.5 pr-4">
                            <span
                              className={
                                row.actual_churn
                                  ? "text-red-400"
                                  : "text-slate-500"
                              }
                            >
                              {row.actual_churn ? "Yes" : "No"}
                            </span>
                          </td>
                        )}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-4 text-sm">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={safePage === 0}
                className="px-3 py-1 rounded-md border border-slate-700 text-slate-300
                           disabled:opacity-40 hover:bg-slate-800 transition-colors"
              >
                Previous
              </button>
              <span className="text-slate-500 text-xs">
                Page {safePage + 1} of {pageCount}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                disabled={safePage >= pageCount - 1}
                className="px-3 py-1 rounded-md border border-slate-700 text-slate-300
                           disabled:opacity-40 hover:bg-slate-800 transition-colors"
              >
                Next
              </button>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
