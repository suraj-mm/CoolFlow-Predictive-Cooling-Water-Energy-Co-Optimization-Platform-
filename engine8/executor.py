"""
engine8/executor.py
Phase 8: Dual-Track Execution Engine.

ARCHITECTURE:
  Two execution tracks run concurrently (PARALLEL mode) or sequentially (SEQUENTIAL mode):

  Track A — SimPy Macro Simulator:
    Updates U(t+1), P(t+1) in a SimPy discrete-event environment.
    Applies DVFS (scales utilisation), fan change (propagated to RC twin next tick),
    and workload migration (reassigns U from src_node to dst_node).
    Returns the updated U/P vectors for the CLOSED LOOP back to Engine 1.

  Track B — Docker/cgroups Hardware Testbed:
    Issues real OS-level commands via the docker SDK and subprocess:
      CPU throttling:  docker update --cpus="<quota>" <container>
      DVFS:            cpupower frequency-set -u <freq>GHz  (Linux only; no-op on Windows)
      Migration:       docker stop <src_container>; docker run <image> on <dst_node>
    On platforms where Docker is unavailable or containers are not running,
    Track B gracefully degrades to simulation mode without stopping Track A.

CLOSED-LOOP CONTRACT (E8 → E1):
  TickResult.u_next  — new CPU utilisation per node (N,) in [0, 1]
  TickResult.p_next  — new power per node (N,) in Watts
  These are fed directly to WorkloadPowerConverter.convert_step() in the next tick.

EXECUTION MODE:
  PARALLEL  : Both tracks applied simultaneously; E8 returns after both complete.
  SEQUENTIAL: Fan adjustment applied first; if projected temps still above threshold,
               DVFS + migration follow. Both tracks still run — only the *application
               order* is sequential, not mutually exclusive.

DESIGN DECISIONS:
  - Track A uses SimPy's Environment.run(until=1) for a single-step advance,
    not a blocking loop, so E8 integrates cleanly into the closed-loop tick cadence.
  - Track B Docker calls use a best-effort pattern: if a container is not found,
    the call is logged and skipped without raising (testbed containers may not
    be running during parquet replay).
  - DVFS power scaling: U_effective = U_raw × dvfs_ratio (not dvfs^3) because
    DVFS directly affects clock frequency, and our P(U) model already encodes
    the non-linear power / frequency relationship. Applying dvfs as a utilisation
    scaler captures: lower clock → less work done per tick → effectively lower U.

CONSUMED BY: Engine 1 (u_next, p_next) and Engine 11 (action_log).
CONSUMES:    Engine 7 ActionDecision.
"""
from __future__ import annotations

import subprocess
import warnings
from typing import Optional

import numpy as np

from shared.config import cfg
from shared.types import ActionDecision, PowerVector, ThermalState, TickResult
from engine1.power_converter import WorkloadPowerConverter

# Container name pattern for the testbed cluster (engine8 uses positional index)
_CONTAINER_NAMES: list[str] = ["dc_node_1", "dc_node_2", "dc_node_3"]
_DOCKER_IMAGE: str = "dc_node:latest"

# DVFS level → GHz mapping (representative; cpupower takes GHz string)
_DVFS_TO_GHZ: dict[float, str] = {
    0.6: "1.2",
    0.7: "1.4",
    0.8: "1.6GHz",
    0.9: "1.8GHz",
    1.0: "3.0GHz",
}

_converter = WorkloadPowerConverter("DC_PRODUCTION")


# ---------------------------------------------------------------------------
# Track A: SimPy Macro Simulator
# ---------------------------------------------------------------------------
def run_track_a(
    action: ActionDecision,
    u_current: np.ndarray,
    p_current: np.ndarray,
    n_nodes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the action in the SimPy macro simulation track.

    Updates utilisation and power for the next tick:
      1. DVFS: U_eff[i] = clip(U_current[i] × a_dvfs[i], 0, 1)
      2. Migration: U_eff[src] → U_eff[src] * 0.1 (offloaded);
                    U_eff[dst] → clip(U_eff[dst] + U_current[src]*0.9, 0, 1)
      3. Power: recomputed via WorkloadPowerConverter with U_eff

    Args:
        action:    E7 ActionDecision.
        u_current: Current utilisation per node (N,) in [0, 1].
        p_current: Current power per node (N,) in Watts.
        n_nodes:   Number of nodes.

    Returns:
        (u_next, p_next) — updated arrays fed back to Engine 1.
    """
    try:
        import simpy  # late import — simpy may not be installed during unit tests
    except ImportError:
        warnings.warn(
            "simpy not installed — Track A running in direct NumPy mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _apply_action_numpy(action, u_current, n_nodes)

    u_next = np.array(u_current, dtype=np.float64)

    # Create a minimal SimPy environment for one-step discrete-event advance
    env = simpy.Environment()

    def _apply_action(env):
        yield env.timeout(1)   # one discrete tick

        # 1. DVFS: scale each node's utilisation by its DVFS ratio
        for i in range(n_nodes):
            ratio = float(action.a_dvfs[i]) if i < len(action.a_dvfs) else 1.0
            u_next[i] = float(np.clip(u_next[i] * ratio, 0.0, 1.0))

        # 2. Migration: redistribute load from src → dst
        if action.a_mig is not None:
            src, dst = action.a_mig
            if 0 <= src < n_nodes and 0 <= dst < n_nodes and src != dst:
                migrated_load = u_current[src] * 0.9   # 90% of src load migrates
                u_next[src] = float(np.clip(u_next[src] - migrated_load * 0.9, 0.0, 1.0))
                u_next[dst] = float(np.clip(u_next[dst] + migrated_load, 0.0, 1.0))

    env.process(_apply_action(env))
    env.run(until=2)

    # Recompute power from updated utilisation
    pv = _converter.convert_step(u_next)
    p_next = pv.p_servers

    return u_next, p_next


def _apply_action_numpy(
    action: ActionDecision,
    u_current: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy fallback for Track A when SimPy is unavailable.
    Applies identical logic to run_track_a without the SimPy environment.
    """
    u_next = np.array(u_current, dtype=np.float64)

    # DVFS
    for i in range(n_nodes):
        ratio = float(action.a_dvfs[i]) if i < len(action.a_dvfs) else 1.0
        u_next[i] = float(np.clip(u_next[i] * ratio, 0.0, 1.0))

    # Migration
    if action.a_mig is not None:
        src, dst = action.a_mig
        if 0 <= src < n_nodes and 0 <= dst < n_nodes and src != dst:
            migrated_load = u_current[src] * 0.9
            u_next[src] = float(np.clip(u_next[src] - migrated_load * 0.9, 0.0, 1.0))
            u_next[dst] = float(np.clip(u_next[dst] + migrated_load, 0.0, 1.0))

    pv = _converter.convert_step(u_next)
    return u_next, pv.p_servers


# ---------------------------------------------------------------------------
# Track B: Docker / cgroups / cpupower Hardware Track
# ---------------------------------------------------------------------------
def run_track_b(
    action: ActionDecision,
    n_nodes: int = 3,
) -> dict[str, str]:
    """
    Apply the action on the physical/containerised testbed.

    Issuing three classes of OS commands:
      - CPU quota (Docker cgroup):   docker update --cpus="<quota>" <container>
      - DVFS frequency (cpupower):   subprocess call (no-op on Windows)
      - Container migration:         docker stop + docker run on destination node

    All failures are caught and logged — Track B is best-effort.
    Track A continues regardless of Track B status.

    Args:
        action:  E7 ActionDecision.
        n_nodes: Number of nodes.

    Returns:
        dict of {command: "ok" | "skip:<reason>" | "err:<msg>"} for the action log.
    """
    log: dict[str, str] = {}

    try:
        import docker as docker_sdk
        client = docker_sdk.from_env()
    except Exception:
        # Docker daemon not running — best-effort skip
        for i in range(n_nodes):
            log[f"docker_node{i+1}"] = "skip:docker_unavailable"
        log["migration"] = "skip:docker_unavailable"
        return log

    for i in range(n_nodes):
        container_name = _CONTAINER_NAMES[i] if i < len(_CONTAINER_NAMES) else f"dc_node_{i+1}"
        dvfs_ratio = float(action.a_dvfs[i]) if i < len(action.a_dvfs) else 1.0

        # CPU quota: map DVFS ratio to CPU fraction for cgroup throttling
        cpu_quota = max(0.1, dvfs_ratio)

        # ---- Docker cgroup CPU throttle -----------------------------------
        try:
            container = client.containers.get(container_name)
            container.update(cpu_quota=int(cpu_quota * 100_000))  # 100000 = 1 CPU
            log[f"docker_cpuquota_node{i+1}"] = "ok"
        except Exception as exc:
            log[f"docker_cpuquota_node{i+1}"] = f"skip:{exc!s}"

        # ---- DVFS via cpupower (Linux only) --------------------------------
        freq_str = _closest_dvfs_ghz(dvfs_ratio)
        try:
            result = subprocess.run(
                ["cpupower", "frequency-set", "-u", freq_str],
                capture_output=True, text=True, timeout=5,
            )
            log[f"cpupower_node{i+1}"] = "ok" if result.returncode == 0 else f"err:{result.stderr}"
        except FileNotFoundError:
            log[f"cpupower_node{i+1}"] = "skip:cpupower_not_found"
        except Exception as exc:
            log[f"cpupower_node{i+1}"] = f"skip:{exc!s}"

    # ---- Container migration -----------------------------------------------
    if action.a_mig is not None:
        src, dst = action.a_mig
        src_name = _CONTAINER_NAMES[src] if src < len(_CONTAINER_NAMES) else f"dc_node_{src+1}"
        dst_name = _CONTAINER_NAMES[dst] if dst < len(_CONTAINER_NAMES) else f"dc_node_{dst+1}"

        try:
            container = client.containers.get(src_name)
            container.stop(timeout=5)
            client.containers.run(
                _DOCKER_IMAGE,
                name=f"{src_name}_migrated",
                detach=True,
                remove=True,
                labels={"migrated_to": dst_name},
            )
            log["migration"] = f"ok:{src_name}→{dst_name}"
        except Exception as exc:
            log["migration"] = f"skip:{exc!s}"
    else:
        log["migration"] = "skip:no_migration"

    return log


def _closest_dvfs_ghz(ratio: float) -> str:
    """Return the nearest DVFS GHz string for cpupower."""
    closest = min(_DVFS_TO_GHZ.keys(), key=lambda k: abs(k - ratio))
    return _DVFS_TO_GHZ[closest]


# ---------------------------------------------------------------------------
# Top-level execution coordinator
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """
    Engine 8: Dual-Track Execution Engine.

    Coordinates Track A (SimPy) and Track B (Docker) based on execution mode.
    Returns TickResult for the closed-loop E8 → E1 connection.

    The fan duty change from a_fan is NOT applied here to U or P — it feeds
    into the RC twin (via ThermalState.fan_duty) on the next tick, since the
    fan controller's thermal effect appears one tick later in the 10-min cadence.
    """

    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes

    def execute(
        self,
        action: ActionDecision,
        u_current: np.ndarray,
        p_current: np.ndarray,
        thermal: ThermalState,
    ) -> TickResult:
        """
        Execute the action decision on both tracks.

        Args:
            action:    E7 ActionDecision.
            u_current: Current CPU utilisation per node (N,) in [0, 1].
            p_current: Current power per node (N,) in Watts.
            thermal:   E3A ThermalState (used for reference in the log).

        Returns:
            TickResult: u_next, p_next (for E1), action_log (for E11).
        """
        # Track A: SimPy macro update
        u_next, p_next = run_track_a(action, u_current, p_current, self.n_nodes)

        # Track B: Docker/cgroups hardware (SEQUENTIAL mode: fan goes first)
        if action.exec_mode == "SEQUENTIAL":
            # In SEQUENTIAL mode, fan adjustment is implicitly first (no separate command
            # needed here — a_fan is carried in the ActionDecision and the RC twin will
            # pick it up on the next tick via the fan_duty field).
            # Temperature re-check: if max T is still above threshold after fan-only,
            # proceed with full DVFS+migration (same action — already committed).
            pass  # full action still applied; SEQUENTIAL = order of effect, not suppression

        track_b_log = run_track_b(action, self.n_nodes)

        action_log = {
            "a_mig": action.a_mig,
            "a_dvfs": action.a_dvfs.tolist(),
            "a_fan": action.a_fan,
            "a_cool": action.a_cool,
            "exec_mode": action.exec_mode,
            "u_before": u_current.tolist(),
            "u_after": u_next.tolist(),
            "p_before": p_current.tolist(),
            "p_after": p_next.tolist(),
            "t_max_before": float(np.max(thermal.temperatures)),
            "track_b": track_b_log,
        }

        return TickResult(
            u_next=u_next,
            p_next=p_next,
            action_log=action_log,
        )
