"""Simulator replay orchestration helpers."""

from __future__ import annotations

import os
import time
import pickle
from types import SimpleNamespace

from simumax.core.base_struct import (
    BarrierBackend,
    NetworkFabric,
    SimuContext,
    SimuSystem,
    SimuThread,
)
from simumax.core.generate_tracing import write_trace_file
from simumax.core.simu_events import write_debug_log
from simumax.core.simu_artifacts import (
    append_memory_events_to_trace,
    export_simu_memory_artifacts,
    should_enable_simu_memory_timeline,
)
from simumax.core.simu_memory import SimuMemoryTracker
from simumax.core.transformer.pipeline_schedule import OptimizerSimulator, PpSchedule
from simumax.core.utils import get_pp_stage_representative_rank, get_rank_group


def run_simulation(perf_model, save_path, merge_lanes=True):
    """Run simulator replay for a configured PerfLLM-like object."""

    model_base = perf_model.model_chunk_dict["first_stage_chunk"]
    # Resource lanes are computed once from the system config (design doc 4.2)
    # and shared by the SimuSystem, every SimuThread lane dict, and the ctx.
    resource_lanes = perf_model.system.simu_resource_lanes()
    simu = SimuSystem(resource_lanes=resource_lanes)
    t0 = time.time()
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "log.log")
    output_json_path = os.path.join(save_path, "tracing_logs.json")
    ctx = SimuContext(BarrierBackend(), merge_lanes=merge_lanes, log_path=log_path,
                      resource_lanes=resource_lanes)
    # Phase C virtual waiters (network-fabric design doc section 8)
    ctx.collective_skew = getattr(perf_model.strategy, "collective_skew", None)
    ctx.strategy = perf_model.strategy
    ctx.num_per_node = perf_model.system.num_per_node
    if should_enable_simu_memory_timeline(perf_model.strategy, perf_model._vp_size()):
        ctx.memory_tracker = SimuMemoryTracker()

    # Network fabric servers (network-fabric design doc sections 5-6);
    # None = off, which reproduces the current behavior.
    fabric = None
    levels = None
    model = getattr(perf_model.system, "fabric_model", None)
    if model in ("nic", "nic+tor"):
        topo = perf_model.system.topology or {}
        share = topo.get("tor_node_share", "auto")
        if share == "auto":
            share = perf_model.system.num_per_node if merge_lanes else 1
        # ToR capacity defaults to the node uplink (inter_node bandwidth);
        # set topology.tor_capacity_gbps below that to model oversubscription.
        tor_capacity = topo.get("tor_capacity_gbps")
        if tor_capacity is None:
            inter = perf_model.system.networks.get("inter_node")
            tor_capacity = inter.bandwidth.gbps if inter is not None else None
        fabric = NetworkFabric(
            perf_model.system.num_per_node,
            tor_enabled=(model == "nic+tor"),
            tor_node_share=share,
            tor_capacity_gbps=tor_capacity,
        )
    elif model == "nic+levels":
        # Hierarchical fabric (hierarchical-network design doc section 8):
        # per-GPU NIC servers plus one logical link server per (level, unit).
        # topology["levels"] is required by the SystemConfig sanity check.
        levels = perf_model.system.topology["levels"]
        # Per-level link capacity in gbps, resolved from each level's net
        # profile (level["net"] -> networks[net].bandwidth.gbps).
        level_capacities = [
            perf_model.system.networks[level["net"]].bandwidth.gbps
            for level in levels
        ]
        fabric = NetworkFabric(perf_model.system.num_per_node)
        fabric.set_level_topology(levels, level_capacities, merge_lanes)
    ctx.fabric = fabric
    # Level routing context of the DES; set only under "nic+levels". A
    # topology["levels"] list may still exist in the config for the
    # analytical levels cost path (net field "auto") — fabric charging
    # stays off unless fabric_model is set.
    ctx.levels = levels

    if merge_lanes:
        simu_ranks = perf_model.strategy.pp_size
    else:
        simu_ranks = perf_model.strategy.world_size

    for rank_i in range(simu_ranks):
        rank = (
            get_pp_stage_representative_rank(rank_i, perf_model.strategy)
            if merge_lanes
            else rank_i
        )
        thread = SimuThread(rank=rank, lanes=resource_lanes)

        args = SimpleNamespace(thread_state=thread.thread_state, rank=rank, microbatch=0)
        rank_info = get_rank_group(rank, model_base.strategy)
        if rank_info["pp_rank"] == 0:
            model_base = perf_model.model_chunk_dict["first_stage_chunk"]
            model_name = "first_stage_chunk"
            stage_key = "first_stage_chunk"
        elif rank_info["pp_rank"] < model_base.strategy.pp_size - 1:
            model_base = perf_model.model_chunk_dict["middle_stage_chunk"]
            model_name = "middle_stage_chunk"
            stage_key = "middle_stage_chunk"
        else:
            model_base = perf_model.model_chunk_dict["last_stage_chunk"]
            model_name = "last_stage_chunk"
            stage_key = "last_stage_chunk"

        vp_size = perf_model._vp_size()
        if vp_size > 1 and perf_model.vpp_stage_chunk_names.get(stage_key):
            stage_models = [
                perf_model.vpp_chunk_dict[name]
                for name in perf_model.vpp_stage_chunk_names[stage_key]
            ]
        else:
            stage_models = [model_base]

        pp_simu = PpSchedule(perf_model.strategy, perf_model.system, stage_models)
        if ctx.memory_tracker is not None:
            stage_static_bytes = sum(model.get_model_info().all for model in stage_models)
            ctx.memory_tracker.init_rank(rank, stage_static_bytes)

        thread.job = pp_simu.prefill_batch(args, com_buff=None)

        op_block = OptimizerSimulator(perf_model, model_name)
        op_block.prefill(args, com_buff=None)
        # Model-wise FSDP (zero_state >= 3): unshard (all-gather params) runs
        # before the PP forward (prepended), reshard (reduce-scatter grads) +
        # optim_step runs after the PP backward (appended). Otherwise the
        # legacy ZeRO-1 tail (RS -> barrier -> optim -> AG) is appended as a
        # single block. See docs/design_simu_zero3_fsdp.md section 4.2.
        if getattr(perf_model.strategy, 'fsdp_mode', 'model-wise') == 'model-wise' \
                and perf_model.strategy.zero_state >= 3:
            thread.job.insert(0, op_block.prefill_unshard_fwd())
            thread.job.append(op_block.prefill_reshard_step_fwd())
        else:
            thread.job.append(op_block.prefill_fwd())

        simu.threads.append(thread)

    simu.simu(ctx)

    print("wall time", time.time() - t0)

    write_debug_log(ctx.event_sink.events, log_path)
    write_trace_file(ctx.event_sink.events, output_json_path)
    if ctx.memory_tracker is not None:
        append_memory_events_to_trace(output_json_path, ctx.memory_tracker)
        export_simu_memory_artifacts(save_path, ctx.memory_tracker, pickle_module=pickle)
