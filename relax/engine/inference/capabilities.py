# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Internal engine capabilities, independent of user configuration and Ray."""

from enum import Enum


class WeightSource(str, Enum):
    POLICY = "policy"
    CHECKPOINT = "checkpoint"
    EXTERNAL = "external"
