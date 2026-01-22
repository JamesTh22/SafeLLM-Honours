# this file basically build the scenario from yaml file and designed to be generic for all yaml files
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from .solver_nnsight import safellm_react_with_trace

@dataclass
class ScenarioSpec:
    scenario_id: str
    cases: List[Dict[str, Any]]
    sandbox_files: Dict[str, str]


def load_yaml_scenario(path: str) -> ScenarioSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ScenarioSpec(
        scenario_id=data["scenario_id"],
        cases=data.get("cases", []) or [],
        sandbox_files=data.get("sandbox_files", {}) or {},
    )
