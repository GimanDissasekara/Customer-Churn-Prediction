import { useState } from "react";
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
import { trainModel } from "../api";
import AgentConversation from "../components/AgentConversation";
import AgentTimeline from "../components/AgentTimeline";
import Card from "../components/Card";
import MetricGrid from "../components/MetricGrid";
import Spinner from "../components/Spinner";
import {
  CHART_TOOLTIP_STYLE,
  MODEL_LABELS,
  TRAINING_AGENTS,
} from "../lib/constants";

export default function Train() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  // Raw errors/logs are kept separately so AgentTimeline can parse them by agent key
  const [pipelineLogs, setPipelineLogs] = useState([]);
  const [pipelineErrors, setPipelineErrors] = useState([]);
  const [agentMessages, setAgentMessages] = useState([]);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPipelineLogs([]);
    setPipelineErrors([]);
    setAgentMessages([]);
    try {
      const res = await trainModel(file);
      setResult(res.data);
      setPipelineLogs(res.data.logs ?? []);
      setAgentMessages(res.data.agent_messages ?? []);
    } catch (err) {
      const detail = err.response?.data?.detail;
      const errList = Array.isArray(detail)
        ? detail
        : detail
        ? [detail]
        : [err.message];
      setError(errList.join("\n"));
      if (Array.isArray(detail)) setPipelineErrors(detail);
      // Partial messages may exist on the error response
      setAgentMessages(err.response?.data?.agent_messages ?? []);
    } finally {
      setLoading(false);
    }
  };

  const candidateData = result
    ? Object.entries(result.candidate_results || {}).map(([name, m]) => ({
        name: MODEL_LABELS[name] || name,
        "ROC-AUC": m.roc_auc,
        F1: m.f1,
        Accuracy: m.accuracy,
      }))
    : [];

  const showTimeline = loading || pipelineLogs.length > 0 || pipelineErrors.length > 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold">Train the churn model</h2>
        <p className="text-slate-400 text-sm mt-1">
          Runs the full LangGraph pipeline: ingestion → cleaning → feature engineering
          → training (LR, RF, XGBoost) → evaluation → registry. Each run is logged to
          MLflow.
        </p>
      </div>

      {/* ── Controls ── */}
      <Card title="Training data">
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
            onClick={handleTrain}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-indigo-500
                       hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed
                       text-sm font-medium text-white transition-colors"
          >
            {loading && <Spinner size={16} />}
            {loading ? "Training…" : "Run training pipeline"}
          </button>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          {file
            ? `Selected: ${file.name}`
            : "No file selected — the bundled Telco Customer Churn dataset will be used."}
        </p>
      </Card>

      {/* ── Agent message-passing timeline ── */}
      {showTimeline && (
        <Card title="Pipeline execution — agent message passing">
          <AgentTimeline
            agents={TRAINING_AGENTS}
            loading={loading}
            logs={pipelineLogs}
            errors={pipelineErrors}
          />
        </Card>
      )}

      {/* ── Agent conversation ── */}
      <Card title="Agent conversations — what each agent said to the next">
        <AgentConversation
          messages={agentMessages}
          loading={loading}
          agents={TRAINING_AGENTS}
        />
      </Card>

      {/* ── Top-level API error (non-pipeline) ── */}
      {error && pipelineErrors.length === 0 && (
        <Card title="Training failed">
          <p className="text-red-400 text-sm whitespace-pre-wrap">{error}</p>
        </Card>
      )}

      {/* ── Results ── */}
      {result && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-lg font-semibold">
              Best model:{" "}
              <span className="text-indigo-400">
                {MODEL_LABELS[result.model_name] || result.model_name}
              </span>
            </h3>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium border ${
                result.passed_quality_gate
                  ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                  : "bg-red-500/15 text-red-400 border-red-500/30"
              }`}
            >
              {result.passed_quality_gate ? "Quality gate passed" : "Quality gate failed"}
            </span>
          </div>

          <Card title="Evaluation metrics">
            <MetricGrid metrics={result.metrics} />
          </Card>

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
                  <Tooltip
                    {...CHART_TOOLTIP_STYLE}
                    formatter={(v) => v.toFixed(4)}
                  />
                  <Legend />
                  <Bar dataKey="ROC-AUC" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="F1"       fill="#22c55e" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Accuracy" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
