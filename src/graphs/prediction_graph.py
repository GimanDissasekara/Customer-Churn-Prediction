"""
Prediction Graph
------------------
LangGraph workflow used for scoring NEW customer CSV uploads with the
already-trained model:

    Ingestion -> Cleaning -> Feature Engineering -> Prediction
"""

from langgraph.graph import END, StateGraph

from src.agents.cleaning import cleaning_agent
from src.agents.feature_engineering import feature_engineering_agent
from src.agents.ingestion import ingestion_agent
from src.agents.prediction import prediction_agent
from src.state import ChurnPipelineState


def _has_errors(state: ChurnPipelineState) -> str:
    return "error" if state.get("errors") else "continue"


def build_prediction_graph():
    graph = StateGraph(ChurnPipelineState)

    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("cleaning", cleaning_agent)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("prediction", prediction_agent)

    graph.set_entry_point("ingestion")

    graph.add_conditional_edges("ingestion", _has_errors, {"continue": "cleaning", "error": END})
    graph.add_conditional_edges("cleaning", _has_errors, {"continue": "feature_engineering", "error": END})
    graph.add_conditional_edges("feature_engineering", _has_errors, {"continue": "prediction", "error": END})
    graph.add_edge("prediction", END)

    return graph.compile()


def run_prediction(input_path: str) -> ChurnPipelineState:
    """Convenience helper: run the prediction graph on a new CSV file."""
    app = build_prediction_graph()
    initial_state: ChurnPipelineState = {
        "input_path": input_path,
        "mode": "predict",
        "errors": [],
        "logs": [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.graphs.prediction_graph <path_to_csv>")
        sys.exit(1)

    final_state = run_prediction(sys.argv[1])

    for line in final_state.get("logs", []):
        print(line)

    if final_state.get("errors"):
        print("\nERRORS:")
        for err in final_state["errors"]:
            print(" -", err)
    else:
        print("\nPredictions (head):")
        print(final_state["predictions_df"].head())
