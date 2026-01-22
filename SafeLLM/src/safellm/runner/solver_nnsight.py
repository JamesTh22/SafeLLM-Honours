from __future__ import annotations

from pathlib import Path
from typing import Optional

from inspect_ai.solver import solver, Solver, TaskState, Generate, react
from inspect_ai.tool import submit

from .tools_inspect import (
    list_files, read_file, read_csv, write_file, make_dir, append_file, write_csv, done
)
from .schema import EpisodeTrace, MessageRecord, ToolCallRecord
from .trace_writer import write_episode_trace


@solver
def safellm_react_with_trace(
    trace_path: Optional[str] = None,
    max_steps: int = 20,
) -> Solver:
    agent = react(
        tools=[
            list_files, read_file, read_csv,
            write_file, make_dir, append_file, write_csv,
            done,
        ],
        system_prompt=lambda state: state.metadata.get("system_prompt", ""),
        max_steps=max_steps,
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # run the agent
        state = await agent(state, generate)

        # build trace from state
        scenario_id = str(state.metadata.get("scenario_id", "unknown"))
        case_id = str(state.metadata.get("case_id", state.sample_id))
        variant = str(state.metadata.get("variant", "unknown"))

        messages = []
        for m in state.messages:
            messages.append(
                MessageRecord(role=m.role, content=m.text, tool_calls=None)
            )

        model_metadata = None
        if state.output is not None and getattr(state.output, "metadata", None):
            model_metadata = dict(state.output.metadata)

        trace = EpisodeTrace(
            scenario_id=scenario_id,
            case_id=case_id,
            variant=variant,
            model=str(state.model),
            sample_id=str(state.sample_id),
            messages=messages,
            output=(state.output.completion if state.output else None),
            model_metadata=model_metadata,
        )

        out_path = Path(trace_path) if trace_path else None
        write_episode_trace(trace, path=out_path)

        return state

    return solve