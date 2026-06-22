import { useEffect, useRef, useState } from "react";

/**
 * Visualises the LangGraph pipeline as a vertical step-by-step timeline.
 *
 * During loading  → animates the active agent using per-step timing estimates.
 * After loading   → parses the real log/error strings returned by the API and
 *                   displays them under the agent that produced them.
 *
 * Log format expected: "[agent_key] message text"
 * Error format:        "[agent_key] error message"
 */

// ─── sub-components ──────────────────────────────────────────────────────────

function Dot({ status }) {
  const cls = {
    pending: "w-3 h-3 rounded-full bg-slate-700 flex-shrink-0",
    active:  "w-3 h-3 rounded-full bg-indigo-400 flex-shrink-0 ring-4 ring-indigo-400/20 animate-pulse",
    done:    "w-3 h-3 rounded-full bg-emerald-400 flex-shrink-0",
    error:   "w-3 h-3 rounded-full bg-red-400 flex-shrink-0",
    skipped: "w-3 h-3 rounded-full bg-slate-800 border border-slate-700 flex-shrink-0",
  };
  return <span className={cls[status] ?? cls.pending} />;
}

function StatusPill({ status }) {
  const cfg = {
    pending: { cls: "text-slate-600",  text: "waiting"  },
    active:  { cls: "text-indigo-400", text: "running…" },
    done:    { cls: "text-emerald-400",text: "done"      },
    error:   { cls: "text-red-400",    text: "error"     },
    skipped: { cls: "text-slate-600",  text: "skipped"   },
  }[status] ?? { cls: "text-slate-600", text: "" };

  return (
    <span className={`text-[10px] font-mono tracking-wide uppercase ${cfg.cls}`}>
      {cfg.text}
    </span>
  );
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/** Extract all log/error lines that belong to a given agent key. */
function linesFor(key, lines) {
  return lines.filter((l) => l.startsWith(`[${key}]`));
}

// ─── main component ───────────────────────────────────────────────────────────

/**
 * @param {Object[]} agents     Ordered list from TRAINING_AGENTS / PREDICTION_AGENTS
 * @param {boolean}  loading    True while the API call is in flight
 * @param {string[]} logs       Log strings from the API response
 * @param {string[]} errors     Error strings from the API response
 */
export default function AgentTimeline({ agents, loading, logs = [], errors = [] }) {
  const [activeStep, setActiveStep] = useState(0);
  const timerRef = useRef(null);

  // Advance the step animation while loading is true.
  // Uses each agent's `stepMs` estimate for realistic pacing.
  useEffect(() => {
    clearTimeout(timerRef.current);
    setActiveStep(0);
    if (!loading) return;

    let step = 0;

    function advance() {
      step += 1;
      if (step >= agents.length) return;
      setActiveStep(step);
      timerRef.current = setTimeout(advance, agents[step]?.stepMs ?? 5000);
    }

    timerRef.current = setTimeout(advance, agents[0]?.stepMs ?? 2000);
    return () => clearTimeout(timerRef.current);
  }, [loading]); // eslint-disable-line react-hooks/exhaustive-deps

  function statusOf(index, key) {
    if (loading) {
      if (index < activeStep) return "done";   // simulated — already passed
      if (index === activeStep) return "active";
      return "pending";
    }
    if (linesFor(key, errors).length > 0) return "error";
    if (linesFor(key, logs).length > 0)   return "done";
    return "skipped";
  }

  return (
    <div className="select-text">
      {agents.map(({ key, label, description, stateKey }, index) => {
        const status   = statusOf(index, key);
        const isLast   = index === agents.length - 1;
        const agentLogs   = loading ? [] : linesFor(key, logs);
        const agentErrors = loading ? [] : linesFor(key, errors);
        const hasMessages = agentLogs.length > 0 || agentErrors.length > 0;
        const nextAgent   = agents[index + 1];

        return (
          <div key={key} className="flex gap-3">
            {/* ── Left: dot + connector ── */}
            <div className="flex flex-col items-center pt-[3px]">
              <Dot status={status} />
              {!isLast && (
                <div className="flex flex-col items-center flex-1 mt-1">
                  <div
                    className={`w-px flex-1 min-h-[20px] ${
                      status === "done" ? "bg-emerald-500/30" : "bg-slate-800"
                    }`}
                  />
                  {/* State key passed to next agent */}
                  {status === "done" && nextAgent && (
                    <span className="text-[9px] font-mono text-slate-600 my-0.5 whitespace-nowrap">
                      {stateKey} →
                    </span>
                  )}
                  <div
                    className={`w-px flex-1 min-h-[20px] ${
                      status === "done" ? "bg-emerald-500/30" : "bg-slate-800"
                    }`}
                  />
                </div>
              )}
            </div>

            {/* ── Right: content ── */}
            <div className={`flex-1 min-w-0 ${isLast ? "pb-0" : "pb-4"}`}>
              {/* Header row */}
              <div className="flex items-center gap-2 flex-wrap">
                <span
                  className={`text-sm font-semibold ${
                    status === "active"  ? "text-indigo-300" :
                    status === "done"    ? "text-slate-100"  :
                    status === "error"   ? "text-red-400"    :
                                          "text-slate-500"
                  }`}
                >
                  {label}
                </span>
                <StatusPill status={status} />
              </div>

              {/* Description */}
              <p className="text-xs text-slate-600 mt-0.5 mb-2">{description}</p>

              {/* Active pulse bar */}
              {status === "active" && (
                <div className="w-28 h-0.5 rounded-full bg-slate-800 overflow-hidden mb-2">
                  <div className="h-full w-full bg-indigo-500/60 animate-pulse rounded-full" />
                </div>
              )}

              {/* Log / error messages */}
              {hasMessages && (
                <div className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 space-y-1">
                  {agentLogs.map((msg, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-emerald-500/60 text-xs mt-0.5 flex-shrink-0">▸</span>
                      <p className="text-xs font-mono text-slate-400 leading-relaxed break-all">
                        {msg}
                      </p>
                    </div>
                  ))}
                  {agentErrors.map((msg, i) => (
                    <div key={i} className="flex gap-2">
                      <span className="text-red-500/70 text-xs mt-0.5 flex-shrink-0">✕</span>
                      <p className="text-xs font-mono text-red-400 leading-relaxed break-all">
                        {msg}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
