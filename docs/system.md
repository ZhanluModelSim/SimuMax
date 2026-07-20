<p align="center">
  <a href="system.md">English</a>| 
  <a href="system-zh.md">中文版本</a> 
</p>


# System Config

SimuMax relies on three core input files: `system`, `strategy`, and `model`.
The `system` file describes the machine side of the problem:

- accelerator compute capability
- memory bandwidth
- network bandwidth / latency
- shape-level operator efficiency

A complete `system.json` always has three logical parts:

- basic information: system name and GPUs per node
- `accelerator`: compute and memory-side behavior
- `networks`: intra-node and inter-node communication behavior

Some machine families may also add extra machine-specific fields such as `FC8`, but the three parts above are the main structure you should think about first.

Workflow pointers:

- overview: [README.md](./README.md)
- model fields: [model.md](./model.md)
- strategy fields: [strategy.md](./strategy.md)
- machine measurement: [simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md)

Important practical note:

- the shared public workflow can generate operator-efficiency data automatically
- on supported CUDA/MUSA hardware, the shared workflow also tries to auto-fill `accelerator.backend`, visible `num_per_node`, and `accelerator.mem_gbs`
- communication fitting is still a guided manual write-back into `networks`
- accelerator-bandwidth defaults are still starter values and should be reviewed for timing-quality analysis
- so a newly generated `system.json` should be treated as timing-ready only after you review machine-side fields and replace the starter network values with your fitted communication numbers

## When to measure your own machine

Using a shipped system config is usually enough when:

- the target machine is close to an existing example machine
- communication topology is similar
- the dominant operator shapes are already covered in `accurate_efficient_factor`

You should measure your own data when:

- the hardware is new
- inter-node or intra-node bandwidth / latency is materially different
- `system.miss_efficiency` is non-empty and the goal is timing analysis

Practical rule:

- for OOM feasibility, missing efficiency may still be acceptable
- for `perf vs simulator` or `perf vs real` timing interpretation, fill missing efficiency first

## Fastest way to start

Do not start from an empty file unless you have to.

Recommended path:

1. Copy the nearest existing config under [configs/system](../configs/system).
2. Rename `sys_name`.
3. Update `num_per_node`, `accelerator.backend`, and `accelerator.mem_gbs`.
4. Replace the `networks` section with the topology closest to your machine.
5. Only then start measuring and filling `accurate_efficient_factor` and fitted communication data.

If you only need approximate analysis, copying the nearest shipped system config is usually better than writing a new one from scratch.

## Minimal viable template

The example below is intentionally complete enough to be a real starting point. It includes a `networks` skeleton, which earlier drafts of this doc did not show clearly enough.

```json
{
    "sys_name": "my_system",
    "num_per_node": 8,
    "accelerator": {
        "backend": "cuda",
        "mem_gbs": 80,
        "op": {
            "default": {
                "tflops": 312,
                "efficient_factor": 0.75
            }
        },
        "bandwidth": {
            "default": {
                "efficient_factor": 0.9,
                "gbps": 1600,
                "latency_us": 40
            }
        },
        "mode": "roofline"
    },
    "networks": {
        "intra_with_pcie": false,
        "low_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 300,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        },
        "high_intra_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 300,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        },
        "inter_node": {
            "processor_usage": 0.0,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 200,
                "latency_us": 30
            },
            "op": {
                "all_reduce": {"scale": 2, "offset": -1},
                "all_gather": {"scale": 1, "offset": -1},
                "reduce_scatter": {"scale": 1, "offset": -1},
                "p2p": {"scale": 1, "offset": 0},
                "all2all": {"scale": 1, "offset": -1}
            }
        }
    }
}
```

## Required fields vs common defaults

Fields you should treat as required:

- `sys_name`
- `num_per_node`
- `accelerator.backend`
- `accelerator.mem_gbs`
- `accelerator.op.default`
- `accelerator.bandwidth.default`
- `networks.intra_with_pcie`
- the matching network groups under `networks`

Fields that many users can keep close to an existing config at first:

- `accelerator.mode` (`roofline` is the common default)
- `processor_usage` (currently a reserved field in public configs)
- operation-specific bandwidth overrides under `accelerator.bandwidth`
- operation-specific communication overrides under `networks.*.op`

For approximate analysis, it is acceptable to:

- start from the nearest shipped config
- leave many `accurate_efficient_factor` entries empty
- use approximate network numbers from a similar machine

For timing-quality analysis, you should measure:

- dominant `matmul`, `group_matmul`, and attention shapes
- intra-node and inter-node communication bandwidth / latency

In other words:

- `accelerator.op.*.accurate_efficient_factor` closes the operator-efficiency gap
- `networks.*` closes the communication-timing gap
- `num_per_node`, `accelerator.mem_gbs`, and `accelerator.bandwidth.*` still need to be reviewed against the real machine before you rely on timing

See [simu_tools/efficency_test/README.md](../simu_tools/efficency_test/README.md) for the shared public measurement path.

## accelerator
The accelerator section includes GPU memory size, memory access bandwidth, computing power, and computational efficiency for various operators.

### backend 
Backend description, used for identification only.


### mem_gbs
GPU memory size, unit is GB.

### op
This section defines the default computing power used by various operators and the accurate computational efficiency under different shapes.

One of SimuMax's core features is implementing shape-level computational efficiency modeling, which is key to accurate performance modeling. Therefore, SimuMax supports user-defined descriptions of computational efficiency for multiple core operators under different shapes and defines a set of shape expression rules. Users need to follow these rules to add computational efficiency for different operator shapes.

The currently supported operator list and their shape expression rules are:




|key|Operator|Format|Example|Notes|
|---|---|---|---|---|
|matmul|Matrix Multiplication| b={batch_size}, m={m}, k={k}, n={n}, layout={layout}, accumulate={accumulate}, out_dtype={out_dtype}|`b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16`|`accumulate`: whether gradient accumulation is performed; True during backward pass for w gradient|
|fp8_matmul|	FP8 Matrix Multiplication|Same as above|Same as above|Same as above|
|sdp_fwd|SDP Forward Computation|batch={batchh_size}, seq_len={seq_len}, head_num={head_num}, kv_head_num={kv_head_num}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, qkv_contiguous={qkv_contiguous}|`batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662`| `qkv_contiguous`: whether input qkv is contiguous in memory; this affects performance, so described separately; generally contiguous input on A100|
|sdp_bwd|SDP Backward Computation|Same as above|Same as above|Same as above|
|group_matmul| Grouped matmul for MOE models|ng={num_groups}, M={fwd_M}, N={fwd_N}, K={fwd_k}, dtype={dtype}, out_dtype={out_dtype}, main_grad_dtype={main_grad_dtype}, stage={stage}, grad={grad}, accumulate={accumulate}, use_split_accumulator=False, single_output={single_output}| fwd stage：<br> `ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614`|1. The` M, N, K `shape descriptions for the fwd, bwd_grad_act, bwd_grad_w stages all equal the M, N, K of the fwd stage, differentiated by the stage parameter.<br>2. `single_output` is True only for the fwd stage.<br>3. `accumulate` is True only for the bwd_grad_w stage.<br>4. `grad` and `use_split_accumulator` are True only for the bwd_grad_w stage.|
|fp8_group_matmul|	Grouped matmul for MOE models|Same as above|Same as above|Same as above|



For example, for NVIDIA A100, the computing power used by its various operators and descriptions of computational efficiency under different shapes are:

```json

"op": {
    "default" : {
        "tflops": 312,
        "efficient_factor": 0.75
    },
     "matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "b=1, m=4096, k=5120, n=1536, layout=TN, accumulate=False, out_dtype=bf16": 0.7876672065615554,
                    "b=1, m=4096, k=1536, n=5120, layout=NN, accumulate=False, out_dtype=bf16": 0.737505124681297
                },
    },
     "fp8_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {},
     },
     "sdp_fwd" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0729673001633662,
                    "batch=1, seq_len=4096, head_num=64, kv_head_num=64, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 1.0544285429372056
                },
     },
     "sdp_bwd" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "batch=1, seq_len=4096, head_num=128, kv_head_num=128, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 0.8018473732899901,
                    "batch=1, seq_len=4096, head_num=64, kv_head_num=64, qk_head_dim=192, v_head_dim=128, qkv_contiguous=True": 0.7942592665301026
                },
     },
     "group_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=fwd, grad=False, accumulate=False, use_split_accumulator=False, single_output=True": 0.6313438865579614,
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=bwd_grad_act, grad=True, accumulate=False, use_split_accumulator=True, single_output=False": 0.6790978664070304,
                    "ng=40, M=616, N=3072, K=5120, dtype=bf16, out_dtype=bf16, main_grad_dtype=fp32, stage=bwd_grad_w, grad=True, accumulate=True, use_split_accumulator=True, single_output=False": 0.5196854178569805
                },
     },
     "fp8_group_matmul" : {
                "tflops": 312,
                "efficient_factor": 0.75,
                "accurate_efficient_factor": {},
     },
}
```
Here, `default` represents the default computing power, which is used for unsupported operator types; under each operator, `tflops` represents the nominal computing power, `efficient_factor` represents the default computational efficiency, and `accurate_efficient_factor` also indicates the actual computational efficiency of each operator under different shapes.


### bandwidth
Memory access bandwidth description, including bandwidth for various memory access types. For example, for NVIDIA A100, its memory access bandwidth description is:
```json
"bandwidth": {
    "default" : {
        "efficient_factor": 0.91,
        "gbps": 1600,
        "latency_us": 40
    },
    "permute_fwd":{
        "efficient_factor": 0.1879,
        "gbps": 1600,
        "latency_us": 40
    },
    "permute_bwd":{
        "efficient_factor": 0.1879,
        "gbps": 1600,
        "latency_us": 40
    },
    "ce":{
        "efficient_factor": 0.808,
        "gbps": 1600,
        "latency_us": 40
    }
}
```
Here, `default` represents the default memory access bandwidth and its efficiency. Besides the default, we add operator-specific fine-tuned efficiency for 3 memory-bound operators that users can customize:
- `permute_fwd` represents memory access bandwidth and efficiency for the permute forward pass.
- `permute_bwd` represents memory access bandwidth and efficiency for the permute backward pass.
- `ce` represents memory access bandwidth and efficiency for cross entropy.

## networks
### FC8
Whether it is FC8 (Fully Connected 8) interconnect.
### intra_with_pcie
Whether intra-node connection uses PCIe.
- if intra_with_pcie=True, it means intra-node connection uses PCIe, and the networks section must also include the following network bandwidth configurations:
```json
"intra_node_pcie_8x": {
},
"intra_node_pcie_4x": {
},
"intra_node_pcie_2x": {
},
"inter_node": { 
}
```
- if intra_with_pcie=False, it means intra-node connection uses high-speed NVLink, and the networks section must also include the following network bandwidth configurations:
```json
"low_intra_node": {
},
"high_intra_node": {
},
"inter_node": { 
}
```

### intra_node_pcie_8x/intra_node_pcie_4x/intra_node_pcie_2x/low_intra_node/high_intra_node/inter_node   
Each network bandwidth configuration includes the following parameters:
- processor_usage: unused, reserved field     
- bandwidth: Network bandwidth configuration, includes:
    - efficient_factor: Network bandwidth efficiency
    - gbps: Network bandwidth, GB/s
    - latency_us: Network latency
- op: Network bandwidth efficiency for specific operations, includes:
    - all_reduce: Network bandwidth efficiency for all_reduce operation
        - scale: 2, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - all_gather: Network bandwidth efficiency for all_gather operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - reduce_scatter: Network bandwidth efficiency for reduce_scatter operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional
    - p2p: Network bandwidth efficiency for p2p (point-to-point) operation
        - scale: 1, fixed
        - offset: 0， fixed
        - efficient_factor， optional
        - latency_us，optional
    - all2all: Network bandwidth efficiency for all2all operation
        - scale: 1, fixed
        - offset: -1， fixed
        - efficient_factor， optional
        - latency_us，optional

For example, detailed configuration of continuous two-card communication bandwidth for A100_PCIE:
```json
"intra_node_pcie_2x": {
            "processor_usage": 0.00,
            "bandwidth": {
                "efficient_factor": 0.5,
                "gbps": 30,
                "latency_us": 10
            },
            "op": {
                "all_reduce": {
                    "scale": 2,
                    "offset": -1,
                    "efficient_factor": 0.6965,
                    "latency_us": 15.51
                },
                "all_gather": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6866,
                    "latency_us": 24.84
                },
                "reduce_scatter": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6419,
                    "latency_us": 131.30
                },
                "p2p": {
                    "scale": 1,
                    "offset": 0
                },
                "all2all": {
                    "scale": 1,
                    "offset": -1,
                    "efficient_factor": 0.6969,
                    "latency_us": 24.07
                }
            }

        },
```

## engines

Optional dict declaring extra hardware engine lanes for the DES resource
model, for example:

```json
"engines": {
    "cube": {"peak_tflops": 320},
    "vector": {"peak_tflops": 80}
}
```

- Absent means a single engine, which reproduces the current behavior.
- Engine names must be valid identifiers and must not collide with the
  reserved lane names `comp`, `comm`, `pp_fwd`, `pp_bwd`, and `off`.
- Vector-lane costs currently use peak-scaled analytic estimates (design
  decision 9.1 in
  [design_simu_kind_resource_model.md](./design_simu_kind_resource_model.md)),
  so each engine entry only carries scalar peaks such as `peak_tflops`; a
  measured efficiency table can be added later without interface changes.

## fabric_model / topology (Preview)

Optional fields enabling the network-fabric contention model of the DES
(section 6 of
[design_simu_network_fabric.md](./design_simu_network_fabric.md)):

```json
"fabric_model": "nic",
"topology": {
    "tor_capacity_gbps": 1600,
    "tor_node_share": "auto"
}
```

- `fabric_model`: absent/`null` (default) disables fabric modeling and
  reproduces the current behavior; `"nic"` enables per-GPU NIC servers so
  that `inter_node` ops queue on their rank's NIC; `"nic+tor"`
  additionally activates top-of-rack (ToR) servers (Preview).
- `topology.tor_capacity_gbps`: ToR server capacity; defaults to
  `inter_node.gbps` (the node uplink).
- `topology.tor_node_share`: `"auto"` or a number >= 1. `"auto"` resolves
  to `num_per_node` under `merge_lanes` (only one rank per node is
  simulated, so a ToR server would otherwise see only 1/num_per_node of
  the node's real traffic) and to `1` otherwise.
- `topology` is only meaningful together with `fabric_model`; setting it
  while `fabric_model` is absent triggers a warning.

## operator_efficiency (optional)

Optional per-operator efficiency table for the analytical cost model
(Phase 1 of
[design_simu_cost_model_tunability.md](./design_simu_cost_model_tunability.md)).
It tunes the efficiency of individual operators without touching the
shared, op-name-level `accurate_efficient_factor` entries.

Each key is either a **class key** (a module class name such as
`"LinearCol"` or `"ParallelCE"`, or the instance-level `cost_op_key`
when set) or a **path key** (the module path from the model root, e.g.
`"layer_3.mlp"`). A key maps to either a scalar (the default efficiency
for that key) or a dict with a `default` plus per-shape `shapes`
entries:

```json
"operator_efficiency": {
    "ParallelCE": 0.52,
    "LinearCol": {
        "default": 0.60,
        "shapes": {
            "b=1, m=4096, k=8192, n=10240, layout=TN, accumulate=False, out_dtype=bf16": 0.66
        }
    },
    "layer_3.mlp": {"default": 0.55}
}
```

### Shape descriptors

Per-shape entries (`shapes` blocks and the level-1/3/5 lookups) are
keyed by the shape descriptor each module emits at runtime. The
formats currently produced are:

- GEMM (LinearBase subclasses): the existing
  `b=, m=, k=, n=, layout=, accumulate=, out_dtype=` format, e.g.
  `b=1, m=4096, k=8192, n=10240, layout=TN, accumulate=False, out_dtype=bf16`.
- sdp / attention (CoreAttention, MLA variants):
  `batch=, seq_len=, head_num=, kv_head_num=, qk_head_dim=, v_head_dim=, qkv_contiguous=`,
  e.g. `batch=1, seq_len=4096, head_num=32, kv_head_num=8, qk_head_dim=128, v_head_dim=128, qkv_contiguous=True`.
  This is exactly the shape key that
  `simu_tools/efficency_test/test_fa_efficiency.py` emits, so measured
  entries paste back verbatim and actually hit.
- elementwise / norm ops that cost as the `default` op (LayerNorm,
  Swiglu, Gelu): a light `b=, s=, h=` descriptor, e.g.
  `b=1, s=4096, h=8192`. `h` is the hidden size for LayerNorm and the
  ffn/intermediate dim the activation operates on for Swiglu/Gelu.

### Lookup chain

At cost time the first hit wins:

| Level | Key | Source |
|---|---|---|
| 1 | `(path_key, shape_desc)` | overrides chain |
| 2 | `path_key` | overrides chain |
| 3 | `(class_key, shape_desc)` | overrides chain |
| 4 | `class_key` | overrides chain |
| 5 | `(op_name, shape_desc)` | `accurate_efficient_factor` (existing) |
| 6 | `op_name` | `efficient_factor` (existing) |
| 7 | `default` op | existing fallback |

The "overrides chain" itself is, highest precedence first:

1. API overrides — `PerfBase.configure(..., efficiency_overrides={...})`
2. strategy `efficiency_overrides` (see
   [strategy.md](./strategy.md#efficiency_overrides))
3. system `operator_efficiency` (this field)

Override keys are validated against the built model at `run_estimate()`
time; a key that matches no module raises `ValueError` instead of
silently doing nothing.

### Hit / miss attribution and the measurement loop

Every lookup records the winning level and source in `hit_efficiency`.
Misses are recorded in `miss_efficiency`, grouped by class key / path
key with their level labels, so the report directly names the operator
to benchmark. The closed loop is: run → inspect `miss_efficiency` →
measure the named operator with
[simu_tools/efficency_test](../simu_tools/efficency_test/README.md) →
paste the result into `operator_efficiency` (or into
`accurate_efficient_factor` for op-name-level entries).

Because modules now emit their shape descriptors at runtime, sdp
misses recorded in `miss_efficiency` carry real shape keys in exactly
the format `test_fa_efficiency.py` benchmarks: copy a missed sdp shape
key into the measurement script's shape list, run it, and paste the
resulting entry back into `accurate_efficient_factor` (or a `shapes`
override) — the simulator will produce the identical key on the next
run, so the measured value is guaranteed to hit.
