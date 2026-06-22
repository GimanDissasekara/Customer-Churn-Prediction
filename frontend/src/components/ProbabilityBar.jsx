export default function ProbabilityBar({ value }) {
  const pct = Math.round(value * 100);
  const barColor =
    pct >= 70 ? "bg-red-500" : pct >= 40 ? "bg-amber-400" : "bg-emerald-500";

  return (
    <div className="flex items-center gap-2 min-w-[130px]">
      <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden flex-shrink-0">
        <div
          className={`h-full rounded-full ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-sm tabular-nums text-slate-200">{pct}%</span>
    </div>
  );
}
