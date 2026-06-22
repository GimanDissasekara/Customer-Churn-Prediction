const LABELS = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1 Score",
  roc_auc: "ROC-AUC",
};

export default function MetricGrid({ metrics }) {
  if (!metrics) return null;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
      {Object.entries(LABELS).map(([key, label]) => (
        <div
          key={key}
          className="bg-slate-800/60 border border-slate-700 rounded-lg p-3 text-center"
        >
          <div className="text-2xl font-semibold text-indigo-400">
            {metrics[key] !== undefined ? metrics[key].toFixed(3) : "—"}
          </div>
          <div className="text-xs text-slate-400 mt-1">{label}</div>
        </div>
      ))}
    </div>
  );
}
