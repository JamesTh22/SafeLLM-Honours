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


