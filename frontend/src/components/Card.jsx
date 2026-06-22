export default function Card({ title, children, className = "" }) {
  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-xl p-5 ${className}`}>
      {title && (
        <h3 className="text-sm font-medium text-slate-400 mb-3 uppercase tracking-wide">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}
