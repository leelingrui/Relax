# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ray-side client for the task's lifecycle coordinator.

Callers name the *phase* they need -- ``genrm``, ``teacher``, ``inference`` --
never the activation group: the group is named after the shared GPU slice and
is therefore only known once placement resolved. Every method degrades to
``None`` or an empty result when the task has no coordinator, which is the
normal case for a layout whose roles have their own GPUs; a caller then keeps
its existing direct path instead of failing.
"""

from typing import Any, Sequence

import ray

from relax.engine.inference.lifecycle import PhaseHandle, PhaseResult
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished by the time a phase is released, so nothing should
# still be in flight. The bound exists so a request that never completes surfaces
# as a failed handover instead of hanging the training step forever.
PHASE_DRAIN_TIMEOUT_S = 600.0


def _resolve(value: Any) -> Any:
    object_ref = getattr(ray, "ObjectRef", ())
    return ray.get(value) if object_ref and isinstance(value, object_ref) else value


class PhaseClient:
    """Thin wrapper over the task inference manager's coordinator methods."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    @property
    def enabled(self) -> bool:
        return self.owner is not None

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.owner, method, None)
        if target is None:
            raise AttributeError(f"Inference owner has no method {method}")
        remote = getattr(target, "remote", None)
        return _resolve(remote(*args, **kwargs) if remote is not None else target(*args, **kwargs))

    def phases(self) -> tuple[str, ...]:
        """The phases this task sequences; empty when there is no
        coordinator."""
        if self.owner is None:
            return ()
        try:
            return tuple(self._call("coordinated_phases"))
        except Exception as exc:
            logger.debug("Inference owner reported no coordinated phases: %s", exc)
            return ()

    def owns(self, phase_id: str) -> bool:
        return phase_id in self.phases()

    def locate(self, phase_id: str) -> tuple[str, str] | None:
        if self.owner is None:
            return None
        located = self._call("locate_phase", phase_id)
        return None if located is None else (located[0], located[1])

    def activation_groups(self) -> tuple[str, ...]:
        if self.owner is None:
            return ()
        return tuple(self._call("activation_groups"))

    def enter(self, phase_id: str, *, operation_id: str, timeout_s: float | None = None) -> PhaseHandle | None:
        """Take exclusive occupancy of the slice ``phase_id`` runs on."""
        located = self.locate(phase_id)
        if located is None:
            return None
        plan_id, _ = located
        return self._call("enter_phase", plan_id, phase_id, operation_id=operation_id, timeout_s=timeout_s)

    def finish(
        self, handle: PhaseHandle, *, operation_id: str, outcome: str = "completed", timeout_s: float | None = None
    ) -> PhaseResult:
        return self._call("finish_phase", handle, operation_id=operation_id, outcome=outcome, timeout_s=timeout_s)

    def activate_phase(
        self, phase_id: str, *, operation_id: str, timeout_s: float | None = None
    ) -> PhaseResult | None:
        """Make ``phase_id`` the occupant without taking an exclusive hold."""
        located = self.locate(phase_id)
        if located is None:
            return None
        _, group = located
        return self._call("transition", group, phase_id, operation_id=operation_id, timeout_s=timeout_s)

    def adopt(self, phase_id: str, *, operation_id: str) -> PhaseResult | None:
        """Tell the coordinator an external owner already activated this
        phase."""
        located = self.locate(phase_id)
        if located is None:
            return None
        _, group = located
        return self._call("adopt_phase", group, phase_id, operation_id=operation_id)

    def release_all(self, *, operation_id: str, timeout_s: float | None = None) -> tuple[PhaseResult, ...]:
        """Empty every activation group and confirm each release.

        This is the training handover: an empty target set drains and
        deactivates whatever occupies the group and activates nothing.
        """
        results = []
        for group in self.activation_groups():
            results.append(
                self._call("switch_model", group, (), operation_id=f"{operation_id}:{group}", timeout_s=timeout_s)
            )
        return tuple(results)

    def confirm_training_release(self, evidence: Any, *, operation_id: str) -> tuple[Any, ...]:
        """Report training's release for every group it shares with
        inference."""
        results = []
        for group in self.activation_groups():
            results.append(
                self._call("confirm_training_release", group, evidence, operation_id=f"{operation_id}:{group}")
            )
        return tuple(results)

    def get_activation_group(self, activation_group: str) -> Any:
        return self._call("get_activation_group", activation_group)


def phase_client(owner: Any, *, phases: Sequence[str] | None = None) -> PhaseClient | None:
    """Return a client only when the task sequences the requested phases."""
    if owner is None:
        return None
    client = PhaseClient(owner)
    owned = client.phases()
    if not owned:
        return None
    if phases is not None and not set(phases) & set(owned):
        return None
    return client


__all__ = ["PhaseClient", "phase_client"]
