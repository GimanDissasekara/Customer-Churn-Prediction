import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getModelMetadata } from "../api";
import Card from "../components/Card";
import MetricGrid from "../components/MetricGrid";
import Spinner from "../components/Spinner";
import { CHART_TOOLTIP_STYLE, MODEL_LABELS } from "../lib/constants";

export default function Dashboard() {
  const [metadata, setMetadata] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getModelMetadata()
      .then((res) => setMetadata(res.data))
      .catch((err) =>
        setError(err.response?.status === 404 ? "no-model" : err.message)
      )
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return (
      <div className="flex items-center gap-3 text-slate-400">
        <Spinner /> Loading model status…
      </div>
    );
  }

  if (error === "no-model") {
    return (
      <Card title="No model trained yet">
        <p className="text-slate-300">
          Head over to the{" "}
          <span className="text-indigo-400 font-medium">Train</span> page to run
          the full training pipeline.
        </p>
      </Card>
    );
  }

  if (error) {
    return (
      <Card title="Error">
        <p className="text-red-400">Failed to load model metadata: {error}</p>
      </Card>
    );
  }

  const candidateData = Object.entries(metadata.candidate_results || {}).map(
    ([name, m]) => ({
      name: MODEL_LABELS[name] || name,
      "ROC-AUC": m.roc_auc,
      F1: m.f1,
      Accuracy: m.accuracy,
    })
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Current Model</h2>
          <p className="text-slate-400 text-sm mt-1">
            {MODEL_LABELS[metadata.model_name] || metadata.model_name} &middot; trained{" "}
            {new Date(metadata.trained_at).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium border ${
              metadata.passed_quality_gate
                ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                : "bg-red-500/15 text-red-400 border-red-500/30"
            }`}
          >
            {metadata.passed_quality_gate ? "Quality gate passed" : "Quality gate failed"}
          </span>
          <button
            onClick={load}
            className="px-3 py-1.5 text-sm border border-slate-700 text-slate-300
                       rounded-md hover:bg-slate-800 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      <Card title="Evaluation metrics (selected model)">
        <MetricGrid metrics={metadata.metrics} />
      </Card>

      {candidateData.length > 0 && (
        <Card title="Model comparison — ROC-AUC / F1 / Accuracy">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={candidateData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis
                  domain={[0, 1]}
                  stroke="#94a3b8"
                  fontSize={12}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip {...CHART_TOOLTIP_STYLE} formatter={(v) => v.toFixed(4)} />
                <Legend />
                <Bar dataKey="ROC-AUC" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="F1" fill="#22c55e" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Accuracy" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      <Card title="Feature columns used by the model">
        <div className="flex flex-wrap gap-2">
          {(metadata.feature_columns || []).map((col) => (
            <span
              key={col}
              className="px-2 py-1 bg-slate-800 border border-slate-700 rounded text-xs text-slate-300"
            >
              {col}
            </span>
          ))}
        </div>
      </Card>
    </div>
  );
}
