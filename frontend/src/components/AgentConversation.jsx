/**
 * AgentConversation
 *
 * Renders inter-agent messages as a chat-style conversation thread.
 * Each message bubble shows the sender, the receiver (if any), and the
 * full handoff content written by that agent.
 *
 * Props:
 *   messages  — array of { sender, receiver, content } from the API
 *   loading   — true while the pipeline is running (shows skeleton placeholders)
 *   agents    — ordered agent config list (TRAINING_AGENTS / PREDICTION_AGENTS)
 *               used to build the skeleton during loading
 */

const SENDER_COLORS = {
  "Ingestion Agent":            { ring: "border-sky-500/40",     dot: "bg-sky-400",     text: "text-sky-300"     },
  "Cleaning Agent":             { ring: "border-violet-500/40",  dot: "bg-violet-400",  text: "text-violet-300"  },
  "Feature Engineering Agent":  { ring: "border-amber-500/40",   dot: "bg-amber-400",   text: "text-amber-300"   },
  "Training Agent":             { ring: "border-indigo-500/40",  dot: "bg-indigo-400",  text: "text-indigo-300"  },
  "Evaluation Agent":           { ring: "border-emerald-500/40", dot: "bg-emerald-400", text: "text-emerald-300" },
  "Registry Agent":             { ring: "border-rose-500/40",    dot: "bg-rose-400",    text: "text-rose-300"    },
  "Prediction Agent":           { ring: "border-teal-500/40",    dot: "bg-teal-400",    text: "text-teal-300"    },
};

const DEFAULT_COLOR = { ring: "border-slate-600", dot: "bg-slate-400", text: "text-slate-300" };

function colorFor(sender) {
  return SENDER_COLORS[sender] ?? DEFAULT_COLOR;
}

// ─── Single message bubble ────────────────────────────────────────────────────

function MessageBubble({ message, index }) {
  const { sender, receiver, content } = message;
  const { ring, dot, text } = colorFor(sender);
  const isLast = !receiver;

  return (
    <div className="flex flex-col gap-2">
      {/* Header */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${dot}`} />
        <span className={`text-sm font-semibold ${text}`}>{sender}</span>
        {receiver && (
          <>
            <span className="text-slate-600 text-xs">→</span>
            <span className="text-xs text-slate-400 font-medium">{receiver}</span>
          </>
        )}
        {isLast && (
          <span className="text-[10px] font-mono uppercase tracking-wide text-slate-600 ml-1">
            pipeline complete
          </span>
        )}
      </div>

      {/* Content bubble */}
      <div
        className={`ml-5 rounded-xl border ${ring} bg-slate-900/70 px-4 py-3
                    shadow-sm`}
      >
        <pre className="text-xs text-slate-300 whitespace-pre-wrap leading-relaxed font-mono">
          {content}
        </pre>
      </div>

      {/* Connector arrow to next message */}
      {!isLast && (
        <div className="ml-6 flex items-center gap-1.5 text-slate-700 text-xs select-none">
          <div className="w-px h-4 bg-slate-800 ml-[3px]" />
        </div>
      )}
    </div>
  );
}

// ─── Skeleton while loading ───────────────────────────────────────────────────

function SkeletonBubble({ agent }) {
  const { dot, text } = colorFor(agent.label);
  return (
    <div className="flex flex-col gap-2 animate-pulse">
      <div className="flex items-center gap-2">
        <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${dot} opacity-40`} />
        <span className={`text-sm font-semibold ${text} opacity-40`}>{agent.label}</span>
        <span className="text-slate-700 text-xs">→ …</span>
      </div>
      <div className="ml-5 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
        <div className="space-y-2">
          <div className="h-2.5 bg-slate-800 rounded w-3/4" />
          <div className="h-2.5 bg-slate-800 rounded w-1/2" />
          <div className="h-2.5 bg-slate-800 rounded w-5/6" />
        </div>
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function AgentConversation({ messages = [], loading = false, agents = [] }) {
  if (!loading && messages.length === 0) return null;

  return (
    <div className="space-y-4">
      {loading ? (
        // Show skeleton placeholders for all agents while waiting
        agents.map((agent) => (
          <SkeletonBubble key={agent.key} agent={agent} />
        ))
      ) : (
        messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} index={i} />
        ))
      )}
    </div>
  );
}
