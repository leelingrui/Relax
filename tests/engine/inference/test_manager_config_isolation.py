# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Registration owns independent copies of mutable launch configuration."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from relax.engine.inference.config import EngineGroupConfig, ModelConfig
from relax.engine.inference.manager import InferenceManager
from relax.engine.inference.types import Role


def test_register_model_isolates_resolve_and_nested_overrides() -> None:
    manager = InferenceManager(Role.ROLLOUT)
    config = ModelConfig("policy", engine_groups=[EngineGroupConfig("regular", 2, overrides={"nested": [1]})])
    original = deepcopy(config)
    result = manager.register_model(config, operation_id="register")

    config.resolve(SimpleNamespace(rollout_num_gpus_per_engine=2, sglang_hf_checkpoint="resolved", hf_checkpoint=None))
    config.engine_groups[0].overrides["nested"].append(2)
    result.engine_groups[0].overrides["nested"].append(3)
    result.name = "changed-return"

    assert manager._models["policy"] == original
    assert manager._operations["register"] == ("register", original)
    assert manager._models["policy"] is not manager._operations["register"][1]
    assert manager.snapshot().models[0].model_id == "policy"
    with pytest.raises(ValueError, match="Operation conflict"):
        manager.register_model(config, operation_id="register")
    with pytest.raises(ValueError, match="Model configuration conflict"):
        manager.register_model(config, operation_id="changed")

    replay = manager.register_model(original, operation_id="register")
    replay.engine_groups[0].overrides["nested"].append(4)
    assert manager.register_model(original, operation_id="register") == original
    assert manager._models["policy"] == original
    assert manager._operations["register"] == ("register", original)


def test_register_model_separate_operation_returns_do_not_alias_definition() -> None:
    manager = InferenceManager(Role.ROLLOUT)
    config = ModelConfig("policy", engine_groups=[EngineGroupConfig("regular", 1)])
    manager.register_model(config, operation_id="first")
    result = manager.register_model(config, operation_id="second")
    result.engine_groups[0].overrides["new"] = {"values": [1]}
    assert manager._models["policy"] == config
    assert manager._operations["first"] == manager._operations["second"] == ("register", config)
    assert manager._operations["first"][1] is not manager._operations["second"][1]
