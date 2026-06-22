"""
Training Graph
----------------
LangGraph workflow that wires together the agents responsible for
training a churn classifier from raw CSV data:

    Ingestion -> Cleaning -> Feature Engineering -> Training -> Evaluation -> Registry
"""

from langgraph.graph import END, StateGraph

from src.agents.cleaning import cleaning_agent
from src.agents.evaluation import evaluation_agent
from src.agents.feature_engineering import feature_engineering_agent
from src.agents.ingestion import ingestion_agent
from src.agents.registry import registry_agent
from src.agents.training import training_agent
from src.state import ChurnPipelineState


def _has_errors(state: ChurnPipelineState) -> str:
    """Conditional edge: stop the graph early if a prior agent recorded an error."""
    return "error" if state.get("errors") else "continue"


def build_training_graph():
    graph = StateGraph(ChurnPipelineState)

    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("cleaning", cleaning_agent)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("training", training_agent)
    graph.add_node("evaluation", evaluation_agent)
    graph.add_node("registry", registry_agent)

    graph.set_entry_point("ingestion")

    graph.add_conditional_edges("ingestion", _has_errors, {"continue": "cleaning", "error": END})
    graph.add_conditional_edges("cleaning", _has_errors, {"continue": "feature_engineering", "error": END})
    graph.add_conditional_edges("feature_engineering", _has_errors, {"continue": "training", "error": END})
    graph.add_conditional_edges("training", _has_errors, {"continue": "evaluation", "error": END})
    graph.add_conditional_edges("evaluation", _has_errors, {"continue": "registry", "error": END})
    graph.add_edge("registry", END)

    return graph.compile()


def run_training(input_path: str) -> ChurnPipelineState:
    """Convenience helper: run the full training graph on a CSV file."""
    app = build_training_graph()
    initial_state: ChurnPipelineState = {
        "input_path": input_path,
        "mode": "train",
        "errors": [],
        "logs": [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys

    from src.config import DEFAULT_TRAIN_FILE

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRAIN_FILE
    final_state = run_training(path)

    for line in final_state.get("logs", []):
        print(line)

    if final_state.get("errors"):
        print("\nERRORS:")
        for err in final_state["errors"]:
            print(" -", err)
    else:
        print("\nEvaluation report:")
        print(final_state["evaluation_report"])
