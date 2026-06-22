const STYLES = {
  High: "bg-red-500/15 text-red-400 border-red-500/30",
  Medium: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  Low: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
};

export default function RiskBadge({ level }) {
  const style = STYLES[level] || "bg-slate-500/15 text-slate-300 border-slate-500/30";
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${style}`}>
      {level}
    </span>
  );
}
