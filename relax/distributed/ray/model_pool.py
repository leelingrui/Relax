# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Common model-pool facade for inference backends.

The facade owns the public pool contract.  A role-specific implementation is
only a backend adapter: it supplies placement, address allocation, runtime
environment and recovery policy, while lifecycle calls are routed through this
class.
"""

from typing import Any

from relax.engine.inference.manager import InferenceManager
from relax.engine.inference.types import Role


class ModelPool:
    """Uniform lifecycle facade over a static backend or a supplied runtime."""

    def __init__(
        self, backend: Any, *, inference_manager: InferenceManager, model_id: str, runtime: bool = False
    ) -> None:
        self.backend = backend
        self.inference_manager = inference_manager
        self.model_id = model_id
        self._runtime = runtime

    @classmethod
    def from_runtime(cls, inference_manager: InferenceManager, model_id: str, runtime: Any) -> "ModelPool":
        """Create a pool for an already-created engine runtime and bind it."""
        inference_manager.bind_pool(model_id, runtime)
        return cls(runtime, inference_manager=inference_manager, model_id=model_id, runtime=True)

    @classmethod
    def from_backend(cls, backend: Any, *, inference_manager: InferenceManager, model_id: str) -> "ModelPool":
        """Wrap a role adapter whose constructor already bound its backend."""
        return cls(backend, inference_manager=inference_manager, model_id=model_id)

    def initialize(self) -> Any:
        return self.backend.initialize()

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # InferenceManager.dispatch invokes the host adapter, so routing back
        # through it here would recurse. The host supplies admission and
        # serialization; this facade supplies the backend operation.
        if self._runtime:
            return getattr(self.inference_manager, method)(self.model_id, *args, **kwargs)
        return getattr(self.backend, method)(*args, **kwargs)

    def health_check(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("health_check", *args, **kwargs)

    def recover(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("recover", *args, **kwargs)

    def onload(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("onload", *args, **kwargs)

    def offload(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("offload", *args, **kwargs)

    def is_onloaded(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.is_onloaded(*args, **kwargs)

    def shutdown(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("shutdown", *args, **kwargs)

    def get_urls(self) -> Any:
        return self.backend.get_urls()

    def get_engine_hosts_ports(self) -> Any:
        return self.backend.get_engine_hosts_ports()

    def get_genrm_engines_and_lock(self) -> Any:
        return self.backend.get_genrm_engines_and_lock()

    def get_discovery_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.get_discovery_snapshot(*args, **kwargs)

    def fanout(self, method: str, **kwargs: Any) -> Any:
        return self.backend.fanout(method, **kwargs)

    def retire(self, ranks: list[int]) -> Any:
        return self.backend.retire(ranks)

    def set_onloaded(self, value: bool) -> Any:
        return self.backend.set_onloaded(value)


def create_model_pool(
    role: Role | str,
    args: Any,
    *role_args: Any,
    inference_manager: InferenceManager | None = None,
    model_id: str = "default",
    defer_init: bool = False,
    **role_kwargs: Any,
) -> ModelPool:
    """Construct the common facade and the role-specific static backend."""
    role = Role(role)
    if role is Role.GENRM:
        from relax.distributed.ray.genrm import GenRMEngineAdapter

        backend_cls = GenRMEngineAdapter
    elif role is Role.TEACHER:
        from relax.distributed.ray.teacher_manager import TeacherEngineAdapter

        backend_cls = TeacherEngineAdapter
    else:
        raise ValueError(f"Unsupported static model-pool role: {role.value}")

    backend = backend_cls(
        args,
        *role_args,
        inference_manager=inference_manager,
        model_id=model_id,
        defer_init=defer_init,
        **role_kwargs,
    )
    return ModelPool.from_backend(backend, inference_manager=backend.inference_manager, model_id=model_id)
