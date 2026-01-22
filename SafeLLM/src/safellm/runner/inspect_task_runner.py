# this file basically build the scenario from yaml file and designed to be generic for all yaml files
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from .solver_nnsight import safellm_react_with_trace