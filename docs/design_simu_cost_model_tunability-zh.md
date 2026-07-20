<p align="center">
  <a href="design_simu_cost_model_tunability.md">English</a>|
  <a href="design_simu_cost_model_tunability-zh.md">中文版本</a>
</p>

# 设计方案：负载接入与算子级成本模型独立可调

- 状态：**草案 v0.1**（讨论已定稿，尚未实现）
- 日期：2026-07-17
- 范围：解析成本路径（`compute_op_accuracy_time`、
  `_comp_cost_info_impl`）、模块成本规格、model JSON 组合方式、测量
  回灌闭环。建立在（不改变）前两份 DES 设计文档的工作之上。

## 1. 背景与问题

当前的负载接入是代码驱动的，分三层：

1. **结构层**：model JSON 只带超参；层组合硬编码在 Python 模块库
   （`dense_module.py` / `moe_module.py`）。
2. **公式层**：每个模块类手写 fwd / bwd_grad_act / bwd_grad_w 三个
   方向的 FLOPs 与访存字节数公式，通信大小在 `prefill` 里手写。
3. **成本层**：fake-forward 逐模块填 `_compute_info`，再由
   `_comp_cost_info_impl`（`base_struct.py:846`）经
   `compute_op_accuracy_time(op_name, flops, shape_desc)`
   （`config.py:906`）换算：`time = flops / (tflops * eff)`。

效率查找只有两个维度：

- `op_name`：调用点硬编码的粗粒度 op 类型（7 种：`matmul`、
  `fp8_matmul`、`group_matmul`、`fp8_group_matmul`、`sdp_fwd`、
  `sdp_bwd`、`default`），同类算子全部共享。
- `shape_desc`：实测 shape 键（`b=, m=, k=, n=, layout=, accumulate=,
  out_dtype=`），**只有 LinearBase 的 GEMM 有**；sdp / elementwise
  算子传空串，只能吃 op 类型默认效率。

局限：

- **无法按算子独立调整**：所有 Linear 共享 `matmul`；实测 shape 条目
  （`accurate_efficient_factor`）不含算子身份，同 shape 的
  LinearCol 和 LinearRow 共享一条记录。无法做到"只调 ParallelCE"
  或"只调 layer_3 的 MLP"。
- **接入要写 Python**：新结构（如 MLA）就要新模块类 + 三组手推
  FLOPs 公式；公式与拓扑代码缠绕；没有声明式组合入口。

## 2. 目标 / 非目标

目标（编号对应第 12 节决策）：

1. 算子级效率调整支持**两种键粒度**：类名与模块路径。
2. `operator_efficiency` 放 system.json（机器属性）；strategy/API
   overrides 做临时调整——并刷新使用文档。
3. **声明式配方**：model JSON 可声明层组合（引用已注册模板），新
   模型接入变为 JSON（+可选注册 cost spec），而非写 Python 类。
4. 非 GEMM 算子（sdp、elementwise）获得 shape 维度的效率查找。
5. 完全向后兼容：不配置任何键 ⇒ 结果逐位一致。

非目标：

- 图/trace 导入负载（ONNX 输入）；保持静态解析哲学。
- 改动成本公式本身（flops/bytes 推导搬入 spec，内容不变）。
- 真机测量流程重做（只做产出归组）。

## 3. cost_op_key 与查找链

每个模块实例获得：

- **class_key**：默认类名（`"LinearCol"`、`"ParallelCE"`、
  `"ExpertMLP"`），可通过新的 `cost_op_key` 属性按实例覆写。
- **path_key**：从模型根节点出发的属性路径点接，如 `layer_3.mlp`、
  `layer_3.attention.q_proj`。fake-forward 时沿现有父子树计算（与
  `peak_path` 同一棵树）；根前缀省略以保持简短。路径键采用**前缀
  语义**：`layer_3.mlp` 上的覆盖作用于整个子树
  （`layer_3.mlp.linear_fc1` 等）；最长匹配前缀优先，并列时按
  API > strategy > system。

效率查找顺序（首中即止，未中下落）：

| 级别 | 键 | 来源 |
|---|---|---|
| 1 | `(path_key, shape_desc)` | overrides 链 |
| 2 | `path_key` | overrides 链 |
| 3 | `(class_key, shape_desc)` | overrides 链 |
| 4 | `class_key` | overrides 链 |
| 5 | `(op_name, shape_desc)` | 现有 `accurate_efficient_factor` |
| 6 | `op_name` | 现有 `efficient_factor` |
| 7 | `default` op | 现有兜底 |

"overrides 链"本身的优先级：API overrides > strategy
`efficiency_overrides` > system `operator_efficiency`。每次查找把
命中级别与来源记入 `hit_efficiency`；未命中记入
`miss_efficiency` 并**按 class_key/path_key 归组**，让测量闭环知道
该测哪个算子。

## 4. 配置

system.json（机器属性，主）：

```json
"operator_efficiency": {
  "ParallelCE": 0.52,
  "LinearCol": {"default": 0.60, "shapes": {"b=1, m=4096, k=8192, n=10240, layout=TN, accumulate=False, out_dtype=bf16": 0.66}},
  "layer_3.mlp": {"default": 0.55}
}
```

strategy.json（临时 what-if，优先于 system）：

```json
"efficiency_overrides": {
  "layer_3.mlp": 0.48
}
```

API（优先级最高，供搜索/调参循环）：

```python
perf.configure(..., efficiency_overrides={"ParallelCE": 0.50})
```

键语法：裸键（类名或路径）映射到标量（该键默认效率）或含
`default` + `shapes` 的字典。校验：键必须能解析到已知 class_key
或合法路径模式；未知键在 configure 时报错（快速失败，不做静默
no-op）。

## 5. 非 GEMM 算子的 shape 维度

- `sdp_fwd` / `sdp_bwd`：注意力模块（CoreAttention、MLA 变体）构建
  形如 `b=, s_q=, s_kv=, h_q=, h_kv=, d=, causal=` 的 shape_desc，
  使 flash-attention 效率可按 shape（seq_len 敏感）测量与键控。
- elementwise / norm 类（现为 `default` op）：轻量 `b=, s=, h=`
  描述符；class_key 级调整已覆盖大部分需求，shape 级为可选细化。
- 现有 `accurate_efficient_factor[(op_name, shape_desc)]` 在第 5 级
  原样工作。

## 6. cost spec 注册表与声明式配方

**注册表**（`simumax/core/cost_specs.py`）：把"算子的成本怎么算"
与"算子在树的什么位置"分离。一条 `CostSpec` 绑定：

- fwd / bwd_grad_act / bwd_grad_w 的 `flops(stage)` / `bytes(stage)`
  推导（现有公式原样搬迁），
- `op_name` 绑定（如 `matmul`），
- shape_desc 构建器，
- 默认 class_key。

现有模块类原样注册（零行为变化）；新算子通过注册 spec 获得成本
语义，不必为改成本而继承。

**配方**（model JSON，可选）：用注册模板声明层组合：

```json
"recipe": {
  "stem": {"embedding": "Embedding", "head": "ParallelCE"},
  "blocks": [
    {"template": "DenseLLMBlock", "count": 3},
    {"template": "MoELLMBlock", "count": 58}
  ]
}
```

`ModelConfig.recipe` + 配方驱动的 `LLMModel` 构建路径；无配方时走
现有组合逻辑不变。模板映射到现有模块类——配方是当前
`dense_layers` + `layer_num` 模式的泛化（该模式成为隐式默认配
方）。超出已注册模板的自定义注意力/MLP 仍需 Python 模块，但常见
的"N 个 block + dense 前缀"家族变为纯 JSON 接入。

## 7. 测量回灌闭环

- `miss_efficiency` 按 class_key/path_key 归组（§3）——报告直接点
  名该测哪个算子。
- `simu_tools/efficency_test` 产出按 cost 键归组，生成可粘贴的
  `operator_efficiency` 片段（GEMM/sdp 按 shape 条目，其余按键级
  默认值）。
- 文档展示闭环：跑 → 看 `miss_efficiency` → 真机测 → 贴回
  system.json。

## 8. 文档刷新（决策 2）

- `docs/system.md`（+zh）：`operator_efficiency` 模式、查找链、测量
  闭环。
- `docs/strategy.md`（+zh）：`efficiency_overrides` 字段与优先级。
- `docs/model.md`（+zh）：`recipe` 段与模板表。
- `docs/tutorial.md`（+zh，如有）：一个调整单个算子的完整示例。
- `AGENTS.md`：注册表 / 键 / 配方摘要。

## 9. 分阶段实施

- **Phase 1 键与 overrides**：`cost_op_key`/路径计算、
  `compute_op_accuracy_time` 查找链、system `operator_efficiency`
  + strategy/API `efficiency_overrides`、按键归组的 miss/hit 记录、
  本阶段文档。验证：无键 golden 等价；合成按键覆盖用例。
- **Phase 2 非 GEMM shape**：sdp + elementwise 的 shape_desc 构建
  器、efficiency_test 归组、文档。验证：合成 sdp 按 shape 条目
  优先于默认值。
- **Phase 3 注册表与配方**：`cost_specs.py`、`ModelConfig.recipe`
  + 配方构建路径、dense/MoE 家族模板表、文档。验证：配方构建的
  模型与类构建在 golden 用例上逐位一致；交付一个纯 JSON 变体模型
  作为演示。

每阶段独立可合入，逐阶段提交。

## 10. 影响面

- `simumax/core/config.py`：查找链、`operator_efficiency` /
  `efficiency_overrides` / `recipe` 字段与校验。
- `simumax/core/base_struct.py`：模块键/路径计算、
  `_comp_cost_info_impl` 的键感知查找接线。
- `simumax/core/cost_specs.py`（新，Phase 3）。
- `simumax/core/transformer/dense_module.py` / `moe_module.py` /
  `language_model.py`：spec 注册、shape_desc 构建器、配方构建路径。
- `simumax/core/perf_llm.py`：API overrides 入口。
- `simu_tools/efficency_test/`：产出归组（Phase 2）。
- 文档见 §8。

## 11. 验收标准

- 默认（无键/无配方）结果在 golden 用例上逐位一致。
- 路径键覆盖精确改变目标模块的耗时；优先级 API > strategy >
  system 可演示。
- sdp 按 shape 条目在合成用例中优先于 op 默认值。
- 配方构建的现有模型与类构建结果完全一致。

## 12. 决策记录

1. **键粒度**：类名与路径双粒度并存（用户要路径的清晰归因，同时
   保留类级调整）。
2. **放置**：`operator_efficiency` 放 system.json 作为机器属性；
   strategy/API overrides 做临时调整；改动必须刷新使用文档。
3. **接入范围**：全量声明式配方（不止注册表）。
4. **非 GEMM shape 维度**：必须——sdp 与 elementwise 算子都配
   shape_desc 构建器。
