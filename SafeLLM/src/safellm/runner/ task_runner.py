# this file basically build the scenario from yaml file and designed to be generic for all yaml files
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from .react_solver import safellm_react_with_trace

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

def build_samples(spec: ScenarioSpec) -> List[Sample]:
    samples: List[Sample] = []

    for c in spec.cases:
        case_id = c["case_id"]
        samples.append(
            Sample(
                id=case_id,
                input=c.get("user_prompt", ""),
                metadata={
                    "scenario_id": spec.scenario_id,
                    "case_id": case_id,
                    "variant": c.get("variant", "benign"),
                    "system_prompt": c.get("system_prompt", ""),
                },
                # Inspect will materialise these into the sandbox working directory
                files=spec.sandbox_files,
            )
        )

    return samples

@task 
def run_yaml(yaml_path: str, trace_path: str = "data/runs/raw/runs_dev.jsonl") -> Task:
    spec = load_yaml_scenario(yaml_path)
    dataset = build_samples(spec)

    return Task(
        dataset=dataset,
        solver=safellm_react_with_trace(trace_path=trace_path),
    )