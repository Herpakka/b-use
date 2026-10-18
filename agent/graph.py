"""StateGraph assembly.

START → plan → observe → decide ─┬─ act ──┬─ observe …        (normal step)
                                  │        ├─ reflect → observe (error / 3x repeat)
                                  │        └─ finalize          (limits)
                                  ├─ human ─┬─ act              (approved)
                                  │         └─ observe          (denied / answered)
                                  └─ finalize → END             (Done / limits)
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.context import AgentContext
from agent.nodes.act import act_node
from agent.nodes.decide import decide_node
from agent.nodes.finalize import finalize_node
from agent.nodes.human import human_node
from agent.nodes.observe import observe_node
from agent.nodes.plan import plan_node
from agent.nodes.reflect import reflect_node
from agent.routing import route_after_act, route_after_decide, route_after_human
from agent.state import AgentState

AgentGraph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def build_graph() -> StateGraph[AgentState, AgentContext, AgentState, AgentState]:
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("plan", plan_node)
    g.add_node("observe", observe_node)
    g.add_node("decide", decide_node)
    g.add_node("human", human_node)
    g.add_node("act", act_node)
    g.add_node("reflect", reflect_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "observe")
    g.add_edge("observe", "decide")
    g.add_conditional_edges("decide", route_after_decide, ["human", "act", "finalize"])
    g.add_conditional_edges("human", route_after_human, ["act", "observe"])
    g.add_conditional_edges("act", route_after_act, ["observe", "reflect", "finalize"])
    g.add_edge("reflect", "observe")
    g.add_edge("finalize", END)
    return g


def compile_graph(checkpointer: BaseCheckpointSaver[Any] | None) -> AgentGraph:
    return build_graph().compile(checkpointer=checkpointer)


def run_config(run_id: str, max_steps: int) -> RunnableConfig:
    # Each step is ~3 supersteps (observe/decide/act) plus reflections and human pauses;
    # LangGraph's default recursion limit of 25 would cut runs short.
    return {"configurable": {"thread_id": run_id}, "recursion_limit": max_steps * 8 + 25}
